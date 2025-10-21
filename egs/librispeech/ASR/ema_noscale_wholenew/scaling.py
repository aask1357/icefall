# Copyright    2022  Xiaomi Corp.        (authors: Daniel Povey, Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import collections
import random
from itertools import repeat
from typing import Optional, Tuple, Union

import torch
import torch.backends.cudnn.rnn as rnn
import torch.nn as nn
from torch import _VF, Tensor

from icefall.utils import is_jit_tracing
from chip_fp16 import Q


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)


class ActivationBalancerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        scale_factor: Tensor,
        sign_factor: Optional[Tensor],
        channel_dim: int,
    ) -> Tensor:
        if channel_dim < 0:
            channel_dim += x.ndim
        ctx.channel_dim = channel_dim
        xgt0 = x > 0
        if sign_factor is None:
            ctx.save_for_backward(xgt0, scale_factor)
        else:
            ctx.save_for_backward(xgt0, scale_factor, sign_factor)
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor) -> Tuple[Tensor, None, None, None]:
        if len(ctx.saved_tensors) == 3:
            xgt0, scale_factor, sign_factor = ctx.saved_tensors
            for _ in range(ctx.channel_dim, x_grad.ndim - 1):
                scale_factor = scale_factor.unsqueeze(-1)
                sign_factor = sign_factor.unsqueeze(-1)
            factor = sign_factor + scale_factor * (xgt0.to(x_grad.dtype) - 0.5)
        else:
            xgt0, scale_factor = ctx.saved_tensors
            for _ in range(ctx.channel_dim, x_grad.ndim - 1):
                scale_factor = scale_factor.unsqueeze(-1)
            factor = scale_factor * (xgt0.to(x_grad.dtype) - 0.5)
        neg_delta_grad = x_grad.abs() * factor
        return (
            x_grad - neg_delta_grad,
            None,
            None,
            None,
        )


def _compute_scale_factor(
    x: Tensor,
    channel_dim: int,
    min_abs: float,
    max_abs: float,
    gain_factor: float,
    max_factor: float,
) -> Tensor:
    if channel_dim < 0:
        channel_dim += x.ndim
    sum_dims = [d for d in range(x.ndim) if d != channel_dim]
    x_abs_mean = torch.mean(x.abs(), dim=sum_dims).to(torch.float32)

    if min_abs == 0.0:
        below_threshold = 0.0
    else:
        # below_threshold is 0 if x_abs_mean > min_abs, can be at most max_factor if
        # x_abs)_mean , min_abs.
        below_threshold = ((min_abs - x_abs_mean) * (gain_factor / min_abs)).clamp(
            min=0, max=max_factor
        )

    above_threshold = ((x_abs_mean - max_abs) * (gain_factor / max_abs)).clamp(
        min=0, max=max_factor
    )

    return below_threshold - above_threshold


def _compute_sign_factor(
    x: Tensor,
    channel_dim: int,
    min_positive: float,
    max_positive: float,
    gain_factor: float,
    max_factor: float,
) -> Tensor:
    if channel_dim < 0:
        channel_dim += x.ndim
    sum_dims = [d for d in range(x.ndim) if d != channel_dim]
    proportion_positive = torch.mean((x > 0).to(torch.float32), dim=sum_dims)
    if min_positive == 0.0:
        factor1 = 0.0
    else:
        # 0 if proportion_positive >= min_positive, else can be
        # as large as max_factor.
        factor1 = (
            (min_positive - proportion_positive) * (gain_factor / min_positive)
        ).clamp_(min=0, max=max_factor)

    if max_positive == 1.0:
        factor2 = 0.0
    else:
        # 0 if self.proportion_positive <= max_positive, else can be
        # as large as -max_factor.
        factor2 = (
            (proportion_positive - max_positive) * (gain_factor / (1.0 - max_positive))
        ).clamp_(min=0, max=max_factor)
    sign_factor = factor1 - factor2
    # require min_positive != 0 or max_positive != 1:
    assert not isinstance(sign_factor, float)
    return sign_factor


class ActivationBalancer(torch.nn.Module):
    """
    Modifies the backpropped derivatives of a function to try to encourage, for
    each channel, that it is positive at least a proportion `threshold` of the
    time.  It does this by multiplying negative derivative values by up to
    (1+max_factor), and positive derivative values by up to (1-max_factor),
    interpolated from 1 at the threshold to those extremal values when none
    of the inputs are positive.

    Args:
        num_channels: the number of channels
        channel_dim: the dimension/axis corresponding to the channel, e.g.
            -1, 0, 1, 2; will be interpreted as an offset from x.ndim if negative.
        min_positive: the minimum, per channel, of the proportion of the time
            that (x > 0), below which we start to modify the derivatives.
        max_positive: the maximum, per channel, of the proportion of the time
            that (x > 0), above which we start to modify the derivatives.
        max_factor: the maximum factor by which we modify the derivatives for
            either the sign constraint or the magnitude constraint;
            e.g. with max_factor=0.02, the the derivatives would be multiplied by
            values in the range [0.98..1.02].
        sign_gain_factor: determines the 'gain' with which we increase the
            change in gradient once the constraints on min_positive and max_positive
            are violated.
        scale_gain_factor: determines the 'gain' with which we increase the
            change in gradient once the constraints on min_abs and max_abs
            are violated.
        min_abs:  the minimum average-absolute-value difference from the mean
            value per channel, which we allow, before we start to modify
            the derivatives to prevent this.
        max_abs:  the maximum average-absolute-value difference from the mean
            value per channel, which we allow, before we start to modify
            the derivatives to prevent this.
        min_prob: determines the minimum probability with which we modify the
            gradients for the {min,max}_positive and {min,max}_abs constraints,
            on each forward().  This is done randomly to prevent all layers
            from doing it at the same time.  Early in training we may use
            higher probabilities than this; it will decay to this value.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int,
        min_positive: float = 0.05,
        max_positive: float = 0.95,
        max_factor: float = 0.04,
        sign_gain_factor: float = 0.01,
        scale_gain_factor: float = 0.02,
        min_abs: float = 0.2,
        max_abs: float = 100.0,
        min_prob: float = 0.1,
    ):
        super(ActivationBalancer, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.max_factor = max_factor
        self.min_abs = min_abs
        self.max_abs = max_abs
        self.min_prob = min_prob
        self.sign_gain_factor = sign_gain_factor
        self.scale_gain_factor = scale_gain_factor

        # count measures how many times the forward() function has been called.
        # We occasionally sync this to a tensor called `count`, that exists to
        # make sure it is synced to disk when we load and save the model.
        self.cpu_count = 0
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))

    def forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_scripting() or not x.requires_grad or torch.jit.is_tracing():
            return x

        count = self.cpu_count
        self.cpu_count += 1

        if random.random() < 0.01:
            # Occasionally sync self.cpu_count with self.count.
            # count affects the decay of 'prob'.  don't do this on every iter,
            # because syncing with the GPU is slow.
            self.cpu_count = max(self.cpu_count, self.count.item())
            self.count.fill_(self.cpu_count)

        # the prob of doing some work exponentially decreases from 0.5 till it hits
        # a floor at min_prob (==0.1, by default)
        prob = max(self.min_prob, 0.5 ** (1 + (count / 4000.0)))

        if random.random() < prob:
            sign_gain_factor = 0.5
            if self.min_positive != 0.0 or self.max_positive != 1.0:
                sign_factor = _compute_sign_factor(
                    x,
                    self.channel_dim,
                    self.min_positive,
                    self.max_positive,
                    gain_factor=self.sign_gain_factor / prob,
                    max_factor=self.max_factor,
                )
            else:
                sign_factor = None

            scale_factor = _compute_scale_factor(
                x.detach(),
                self.channel_dim,
                min_abs=self.min_abs,
                max_abs=self.max_abs,
                gain_factor=self.scale_gain_factor / prob,
                max_factor=self.max_factor,
            )
            return ActivationBalancerFunction.apply(
                x,
                scale_factor,
                sign_factor,
                self.channel_dim,
            )
        else:
            return x





def _diag(x: Tensor):  # like .diag(), but works for tensors with 3 dims.
    if x.ndim == 2:
        return x.diag()
    else:
        (batch, dim, dim) = x.shape
        x = x.reshape(batch, dim * dim)
        x = x[:, :: dim + 1]
        assert x.shape == (batch, dim)
        return x


def _whitening_metric(x: Tensor, num_groups: int):
    """
    Computes the "whitening metric", a value which will be 1.0 if all the eigenvalues of
    of the centered feature covariance are the same within each group's covariance matrix
    and also between groups.
    Args:
        x: a Tensor of shape (*, num_channels)
     num_groups:  the number of groups of channels, a number >=1 that divides num_channels
    Returns:
        Returns a scalar Tensor that will be 1.0 if the data is "perfectly white" and
    greater than 1.0 otherwise.
    """
    assert x.dtype != torch.float16
    x = x.reshape(-1, x.shape[-1])
    (num_frames, num_channels) = x.shape
    assert num_channels % num_groups == 0
    channels_per_group = num_channels // num_groups
    x = x.reshape(num_frames, num_groups, channels_per_group).transpose(0, 1)
    # x now has shape (num_groups, num_frames, channels_per_group)
    # subtract the mean so we use the centered, not uncentered, covariance.
    # My experience has been that when we "mess with the gradients" like this,
    # it's better not do anything that tries to move the mean around, because
    # that can easily cause instability.
    x = x - x.mean(dim=1, keepdim=True)
    # x_covar: (num_groups, channels_per_group, channels_per_group)
    x_covar = torch.matmul(x.transpose(1, 2), x)
    x_covar_mean_diag = _diag(x_covar).mean()
    # the following expression is what we'd get if we took the matrix product
    # of each covariance and measured the mean of its trace, i.e.
    # the same as _diag(torch.matmul(x_covar, x_covar)).mean().
    x_covarsq_mean_diag = (x_covar**2).sum() / (num_groups * channels_per_group)
    # this metric will be >= 1.0; the larger it is, the less 'white' the data was.
    metric = x_covarsq_mean_diag / (x_covar_mean_diag**2 + 1.0e-20)
    return metric


class WhiteningPenaltyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: Tensor, num_groups: int, whitening_limit: float, grad_scale: float
    ) -> Tensor:
        ctx.save_for_backward(x)
        ctx.num_groups = num_groups
        ctx.whitening_limit = whitening_limit
        ctx.grad_scale = grad_scale
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor):
        (x_orig,) = ctx.saved_tensors
        with torch.enable_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x_detached = x_orig.to(torch.float32).detach()
                x_detached.requires_grad = True

                metric = _whitening_metric(x_detached, ctx.num_groups)

                (metric - ctx.whitening_limit).relu().backward()
                penalty_grad = x_detached.grad
                scale = ctx.grad_scale * (
                    x_grad.to(torch.float32).norm() / (penalty_grad.norm() + 1.0e-20)
                )
                penalty_grad = penalty_grad * scale
        return x_grad + penalty_grad.to(x_grad.dtype), None, None, None


class Whiten(nn.Module):
    def __init__(
        self,
        num_groups: int,
        whitening_limit: float,
        prob: Union[float, Tuple[float, float]],
        grad_scale: float,
    ):
        """
        Args:
          num_groups: the number of groups to divide the channel dim into before
            whitening.  We will attempt to make the feature covariance
            within each group, after mean subtraction, as "white" as possible,
            while having the same trace across all groups.
         whitening_limit: a value greater than 1.0, that dictates how much
           freedom we have to violate the constraints.  1.0 would mean perfectly
           white, with exactly the same trace across groups; larger values
           give more freedom.  E.g. 2.0.
         prob: the probability with which we apply the gradient modification
           (also affects the grad scale).  May be supplied as a float,
           or as a pair (min_prob, max_prob)

          grad_scale: determines the scale on the gradient term from this object,
            relative to the rest of the gradient on the attention weights.
            E.g. 0.02 (you may want to use smaller values than this if prob is large)
        """
        super(Whiten, self).__init__()
        assert num_groups >= 1
        assert whitening_limit >= 1
        assert grad_scale >= 0
        self.num_groups = num_groups
        self.whitening_limit = whitening_limit
        if isinstance(prob, float):
            assert 0 < prob <= 1
            self.prob = prob
        else:
            (self.min_prob, self.max_prob) = prob
            assert 0 < self.min_prob < self.max_prob <= 1
            self.prob = self.max_prob

        self.grad_scale = grad_scale

    def forward(self, x: Tensor) -> Tensor:
        """
        In the forward pass, this function just returns the input unmodified.
        In the backward pass, it will modify the gradients to ensure that the
        distribution in each group has close to (lambda times I) as the covariance
        after mean subtraction, with the same lambda across groups.
        For whitening_limit > 1, there will be more freedom to violate this
        constraint.

        Args:
           x: the input of shape (*, num_channels)

        Returns:
            x, unmodified.   You should make sure
        you use the returned value, or the graph will be freed
        and nothing will happen in backprop.
        """
        if not x.requires_grad or random.random() > self.prob or self.grad_scale == 0:
            return x
        else:
            if hasattr(self, "min_prob") and random.random() < 0.25:
                # occasionally switch between min_prob and max_prob, based on whether
                # we are above or below the threshold.
                if (
                    _whitening_metric(x.to(torch.float32), self.num_groups)
                    > self.whitening_limit
                ):
                    # there would be a change to the grad.
                    self.prob = self.max_prob
                else:
                    self.prob = self.min_prob

            return WhiteningPenaltyFunction.apply(
                x, self.num_groups, self.whitening_limit, self.grad_scale
            )


class BasicNorm(torch.nn.Module):
    """
    This is intended to be a simpler, and hopefully cheaper, replacement for
    LayerNorm.  The observation this is based on, is that Transformer-type
    networks, especially with pre-norm, sometimes seem to set one of the
    feature dimensions to a large constant value (e.g. 50), which "defeats"
    the LayerNorm because the output magnitude is then not strongly dependent
    on the other (useful) features.  Presumably the weight and bias of the
    LayerNorm are required to allow it to do this.

    So the idea is to introduce this large constant value as an explicit
    parameter, that takes the role of the "eps" in LayerNorm, so the network
    doesn't have to do this trick.  We make the "eps" learnable.

    Args:
       num_channels: the number of channels, e.g. 512.
      channel_dim: the axis/dimension corresponding to the channel,
        interprted as an offset from the input's ndim if negative.
        shis is NOT the num_channels; it should typically be one of
        {-2, -1, 0, 1, 2, 3}.
       eps: the initial "epsilon" that we add as ballast in:
             scale = ((input_vec**2).mean() + epsilon)**-0.5
          Note: our epsilon is actually large, but we keep the name
          to indicate the connection with conventional LayerNorm.
       learn_eps: if true, we learn epsilon; if false, we keep it
         at the initial value.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,  # CAUTION: see documentation.
        eps: float = 0.25,
        learn_eps: bool = True,
    ) -> None:
        super(BasicNorm, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        if learn_eps:
            self.eps = nn.Parameter(torch.tensor(eps).log().detach())
        else:
            self.register_buffer("eps", torch.tensor(eps).log().detach())

    def forward(self, x: Tensor) -> Tensor:
        if not is_jit_tracing():
            assert x.shape[self.channel_dim] == self.num_channels
        scales = (
            torch.mean(x**2, dim=self.channel_dim, keepdim=True) + self.eps.exp()
        ) ** -0.5
        return x * scales


class ScaledLinear(nn.Linear):
    """
    A modified version of nn.Linear where the parameters are scaled before
    use, via:
         weight = self.weight * self.weight_scale.exp()
         bias = self.bias * self.bias_scale.exp()

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
        initial_speed: this affects how fast the parameter will
           learn near the start of training; you can set it to a
           value less than one if you suspect that a module
           is contributing to instability near the start of training.
           Nnote: regardless of the use of this option, it's best to
           use schedulers like Noam that have a warm-up period.
           Alternatively you can set it to more than 1 if you want it to
           initially train faster.   Must be greater than 0.
    """

    def __init__(
        self,
        *args,
        initial_scale: float = 1.0,
        initial_speed: float = 1.0,
        **kwargs,
    ):
        super(ScaledLinear, self).__init__(*args, **kwargs)
        initial_scale = torch.tensor(initial_scale).log()
        self.weight_scale = nn.Parameter(initial_scale.clone().detach())
        if self.bias is not None:
            self.bias_scale = nn.Parameter(initial_scale.clone().detach())
        else:
            self.register_parameter("bias_scale", None)

        self._reset_parameters(
            initial_speed
        )  # Overrides the reset_parameters in nn.Linear
        Q(self)

    def _reset_parameters(self, initial_speed: float):
        std = 0.1 / initial_speed
        a = (3**0.5) * std
        nn.init.uniform_(self.weight, -a, a)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
        fan_in = self.weight.shape[1] * self.weight[0][0].numel()
        scale = fan_in**-0.5  # 1/sqrt(fan_in)
        with torch.no_grad():
            self.weight_scale += torch.tensor(scale / std).log()

    def get_weight(self):
        with torch.no_grad():
            self.weight_scale.clamp_(min=-10.0, max=2.0)
        return self.weight * self.weight_scale.exp()

    def get_bias(self):
        if self.bias is None or self.bias_scale is None:
            return None
        else:
            return self.bias * self.bias_scale.exp()

    def forward(self, input: Tensor) -> Tensor:
        return torch.nn.functional.linear(input, self.get_weight(), self.get_bias())


class ScaledConv1d(nn.Conv1d):
    # See docs for ScaledLinear
    def __init__(
        self,
        *args,
        initial_scale: float = 1.0,
        initial_speed: float = 1.0,
        **kwargs,
    ):
        super(ScaledConv1d, self).__init__(*args, **kwargs)
        initial_scale = torch.tensor(initial_scale).log()

        self.bias_scale: Optional[nn.Parameter]  # for torchscript

        self.weight_scale = nn.Parameter(initial_scale.clone().detach())
        if self.bias is not None:
            self.bias_scale = nn.Parameter(initial_scale.clone().detach())
        else:
            self.register_parameter("bias_scale", None)
        self._reset_parameters(
            initial_speed
        )  # Overrides the reset_parameters in base class
        Q(self)

    def _reset_parameters(self, initial_speed: float):
        std = 0.1 / initial_speed
        a = (3**0.5) * std
        nn.init.uniform_(self.weight, -a, a)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
        fan_in = self.weight.shape[1] * self.weight[0][0].numel()
        scale = fan_in**-0.5  # 1/sqrt(fan_in)
        with torch.no_grad():
            self.weight_scale += torch.tensor(scale / std).log()

    def get_weight(self):
        with torch.no_grad():
            self.weight_scale.clamp_(min=-10.0, max=2.0)
        return self.weight * self.weight_scale.exp()

    def get_bias(self):
        bias = self.bias
        bias_scale = self.bias_scale
        if bias is None or bias_scale is None:
            return None
        else:
            return bias * bias_scale.exp()

    def forward(self, input: Tensor) -> Tensor:
        F = torch.nn.functional
        if self.padding_mode != "zeros":
            return F.conv1d(
                F.pad(
                    input,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                self.get_weight(),
                self.get_bias(),
                self.stride,
                (0,),
                self.dilation,
                self.groups,
            )
        return F.conv1d(
            input,
            self.get_weight(),
            self.get_bias(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ScaledConv2d(nn.Conv2d):
    # See docs for ScaledLinear
    def __init__(
        self,
        *args,
        initial_scale: float = 1.0,
        initial_speed: float = 1.0,
        **kwargs,
    ):
        super(ScaledConv2d, self).__init__(*args, **kwargs)
        initial_scale = torch.tensor(initial_scale).log()
        self.weight_scale = nn.Parameter(initial_scale.clone().detach())
        if self.bias is not None:
            self.bias_scale = nn.Parameter(initial_scale.clone().detach())
        else:
            self.register_parameter("bias_scale", None)
        self._reset_parameters(
            initial_speed
        )  # Overrides the reset_parameters in base class

    def _reset_parameters(self, initial_speed: float):
        std = 0.1 / initial_speed
        a = (3**0.5) * std
        nn.init.uniform_(self.weight, -a, a)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
        fan_in = self.weight.shape[1] * self.weight[0][0].numel()
        scale = fan_in**-0.5  # 1/sqrt(fan_in)
        with torch.no_grad():
            self.weight_scale += torch.tensor(scale / std).log()

    def get_weight(self):
        with torch.no_grad():
            self.weight_scale.clamp_(min=-10.0, max=2.0)
        return self.weight * self.weight_scale.exp()

    def get_bias(self):
        # see https://github.com/pytorch/pytorch/issues/24135
        bias = self.bias
        bias_scale = self.bias_scale
        if bias is None or bias_scale is None:
            return None
        else:
            return bias * bias_scale.exp()

    def _conv_forward(self, input, weight):
        F = torch.nn.functional
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                weight,
                self.get_bias(),
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input,
            weight,
            self.get_bias(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.get_weight())


class DoubleSwishFunction(torch.autograd.Function):
    """
      double_swish(x) = x * torch.sigmoid(x-1)
    This is a definition, originally motivated by its close numerical
    similarity to swish(swish(x)), where swish(x) =  x * sigmoid(x).

    Memory-efficient derivative computation:
     double_swish(x) = x * s, where s(x) = torch.sigmoid(x-1)
     double_swish'(x) = d/dx double_swish(x) =  x * s'(x) + x' * s(x) = x * s'(x) + s(x).
     Now, s'(x) = s(x) * (1-s(x)).
     double_swish'(x) =  x * s'(x) + s(x).
                      =  x * s(x) * (1-s(x)) + s(x).
                     = double_swish(x) * (1-s(x)) + s(x)
     ... so we just need to remember s(x) but not x itself.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        x = x.detach()
        s = torch.sigmoid(x - 1.0)
        y = x * s
        ctx.save_for_backward(s, y)
        return y

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        s, y = ctx.saved_tensors
        return (y * (1 - s) + s) * y_grad


class ScaledEmbedding(nn.Module):
    r"""This is a modified version of nn.Embedding that introduces a learnable scale
    on the parameters.  Note: due to how we initialize it, it's best used with
    schedulers like Noam that have a warmup period.

    It is a simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int, optional): If given, pads the output with the embedding vector at :attr:`padding_idx`
                                         (initialized to zeros) whenever it encounters the index.
        scale_grad_by_freq (boolean, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor.
                                 See Notes for more details regarding sparse gradients.

        initial_speed (float, optional):  This affects how fast the parameter will
           learn near the start of training; you can set it to a value less than
           one if you suspect that a module is contributing to instability near
           the start of training.  Note: regardless of the use of this option,
           it's best to use schedulers like Noam that have a warm-up period.
           Alternatively you can set it to more than 1 if you want it to
           initially train faster.  Must be greater than 0.


    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
                         initialized from :math:`\mathcal{N}(0, 1)`

    Shape:
        - Input: :math:`(*)`, LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`

    .. note::
        Keep in mind that only a limited number of optimizers support
        sparse gradients: currently it's :class:`optim.SGD` (`CUDA` and `CPU`),
        :class:`optim.SparseAdam` (`CUDA` and `CPU`) and :class:`optim.Adagrad` (`CPU`)

    .. note::
        With :attr:`padding_idx` set, the embedding vector at
        :attr:`padding_idx` is initialized to all zeros. However, note that this
        vector can be modified afterwards, e.g., using a customized
        initialization method, and thus changing the vector used to pad the
        output. The gradient for this vector from :class:`~torch.nn.Embedding`
        is always zero.

    Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding = nn.Embedding(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
        >>> embedding(input)
        tensor([[[-0.0251, -1.6902,  0.7172],
                 [-0.6431,  0.0748,  0.6969],
                 [ 1.4970,  1.3448, -0.9685],
                 [-0.3677, -2.7265, -0.1685]],

                [[ 1.4970,  1.3448, -0.9685],
                 [ 0.4362, -0.4004,  0.9400],
                 [-0.6431,  0.0748,  0.6969],
                 [ 0.9124, -2.3616,  1.1151]]])


        >>> # example with padding_idx
        >>> embedding = nn.Embedding(10, 3, padding_idx=0)
        >>> input = torch.LongTensor([[0,2,0,5]])
        >>> embedding(input)
        tensor([[[ 0.0000,  0.0000,  0.0000],
                 [ 0.1535, -2.0309,  0.9315],
                 [ 0.0000,  0.0000,  0.0000],
                 [-0.1655,  0.9897,  0.0635]]])

    """
    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "padding_idx",
        "scale_grad_by_freq",
        "sparse",
    ]

    num_embeddings: int
    embedding_dim: int
    padding_idx: int
    scale_grad_by_freq: bool
    weight: Tensor
    sparse: bool

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        initial_speed: float = 1.0,
    ) -> None:
        super(ScaledEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.scale_grad_by_freq = scale_grad_by_freq

        self.scale = nn.Parameter(torch.zeros(()))  # see reset_parameters()
        self.sparse = sparse

        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters(initial_speed)

    def reset_parameters(self, initial_speed: float = 1.0) -> None:
        std = 0.1 / initial_speed
        nn.init.normal_(self.weight, std=std)
        nn.init.constant_(self.scale, torch.tensor(1.0 / std).log())

        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        F = torch.nn.functional
        with torch.no_grad():
            self.scale.clamp_(min=-10.0, max=2.0)
        scale = self.scale.exp()
        if input.numel() < self.num_embeddings:
            return (
                F.embedding(
                    input,
                    self.weight,
                    self.padding_idx,
                    None,
                    2.0,  # None, 2.0 relate to normalization
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                * scale
            )
        else:
            return F.embedding(
                input,
                self.weight * scale,
                self.padding_idx,
                None,
                2.0,  # None, 2.0 relates to normalization
                self.scale_grad_by_freq,
                self.sparse,
            )

    def extra_repr(self) -> str:
        # s = "{num_embeddings}, {embedding_dim}, scale={scale}"
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        return s.format(**self.__dict__)
