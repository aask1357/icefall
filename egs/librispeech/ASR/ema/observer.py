from typing import Optional, Tuple

import torch
from torch import Tensor
from torchao.quantization.granularity import Granularity
from torchao.quantization.granularity import PerTensor
from torchao.quantization.observer import (
    AffineQuantizedObserverBase,
    AffineQuantizedMSEObserver,
)
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
    _get_reduction_params,
    choose_qparams_affine_with_min_max,
)
from torchao.quantization.utils import get_block_size


class MovingAverageMSEObserver(AffineQuantizedMSEObserver):
    def __init__(
        self,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        granularity: Granularity,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        eps: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: bool = True,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        steps: int = 100,
        gamma: float = 0.99,
    ):
        super().__init__(
            mapping_type,
            target_dtype,
            granularity,
            quant_min,
            quant_max,
            eps,
            scale_dtype,
            zero_point_dtype,
            preserve_zero,
            zero_point_domain,
            steps,
            run_once=False,
        )
        self.gamma = gamma

    def forward(self, x: Tensor) -> Tensor:
        min_val, max_val = self.line_search(x)
        if not hasattr(self, "min_val") or not hasattr(self, "max_val"):
            self.min_val = min_val
            self.max_val = max_val
        else:
            assert self.min_val.shape == min_val.shape, (
                f"Can't update existing min_val - shape mismatch, self.min_val:{self.min_val.shape} != min_val:{min_val.shape}"
            )
            assert self.max_val.shape == max_val.shape, (
                f"Can't update existing max_val - shape mismatch, self.max_val {self.max_val.shape} != max_val:{max_val.shape}"
            )
            min_val = min_val + self.gamma * (self.min_val - min_val)
            max_val = max_val + self.gamma * (self.max_val - max_val)
            self.min_val.copy_(min_val)
            self.max_val.copy_(max_val)
        return x


class MovingAverageMinMaxObserver(AffineQuantizedObserverBase):
    def __init__(
        self,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        granularity: Granularity,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        eps: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: bool = True,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        gamma: float = 0.99,
    ):
        super().__init__(
            mapping_type,
            target_dtype,
            granularity,
            quant_min,
            quant_max,
            eps,
            scale_dtype,
            zero_point_dtype,
            preserve_zero,
            zero_point_domain,
        )
        self.gamma = gamma

    def forward(self, x: Tensor) -> Tensor:
        if x.numel() == 0:
            return x
        x_detached = x.detach()  # avoid keeping autograd tape
        assert self.granularity is not None, "granularity is None"
        block_size = get_block_size(x_detached.shape, self.granularity)
        shape_for_reduction, reduction_dims = _get_reduction_params(
            block_size, x_detached.size()
        )
        x_detached = x_detached.view(shape_for_reduction)
        min_val = torch.amin(x_detached, dim=reduction_dims, keepdim=False)
        max_val = torch.amax(x_detached, dim=reduction_dims, keepdim=False)
        if not hasattr(self, "min_val") or not hasattr(self, "max_val"):
            self.min_val = min_val
            self.max_val = max_val
        else:
            assert self.min_val.shape == min_val.shape, (
                f"Can't update existing min_val - shape mismatch, self.min_val:{self.min_val.shape} != min_val:{min_val.shape}"
            )
            assert self.max_val.shape == max_val.shape, (
                f"Can't update existing max_val - shape mismatch, self.max_val {self.max_val.shape} != max_val:{max_val.shape}"
            )
            min_val = min_val + self.gamma * (self.min_val - min_val)
            max_val = max_val + self.gamma * (self.max_val - max_val)
            self.min_val.copy_(min_val)
            self.max_val.copy_(max_val)
        return x

    def calculate_qparams(self) -> Tuple[Tensor, Tensor]:
        assert hasattr(self, "min_val") and hasattr(self, "max_val"), (
            "Expecting the observer has min_val and max_val, please run the observer before calling calculate_qparams"
        )
        return choose_qparams_affine_with_min_max(
            self.min_val,
            self.max_val,
            self.mapping_type,
            [],  # BlockSize is not needed because the min/max are already reduced
            self.target_dtype,
            self.quant_min,
            self.quant_max,
            self.eps,
            self.scale_dtype,
            self.zero_point_dtype,
            self.preserve_zero,
            self.zero_point_domain,
        )


class HistogramObserver(AffineQuantizedObserverBase):
    def __init__(
        self,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        granularity: Granularity,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        eps: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: bool = True,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        bins: int = 2048,
    ):
        assert granularity == PerTensor(), "HistogramObserver only supports PerTensor granularity"
        super().__init__(
            mapping_type,
            target_dtype,
            granularity,
            quant_min,
            quant_max,
            eps,
            scale_dtype,
            zero_point_dtype,
            preserve_zero,
            zero_point_domain,
        )
        self.dst_nbins = 2 ** torch.iinfo(target_dtype).bits
        self.bins = bins
        self.upsample_rate = (
            16  # used to reduce quantization errors when upscaling histogram
        )

    def _get_norm(
        self, delta_begin: torch.Tensor, delta_end: torch.Tensor, density: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute the norm of the values uniformaly distributed between
        delta_begin and delta_end.
        Currently only L2 norm is supported.

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
        norm = (
            delta_end * delta_end * delta_end - delta_begin * delta_begin * delta_begin
        ) / 3
        return density * norm

    def _compute_quantization_error(self, next_start_bin: int, next_end_bin: int):
        r"""
        Compute the quantization error if we use start_bin to end_bin as the
        min and max to do the quantization.
        """
        bin_width = (self.max_val.item() - self.min_val.item()) / self.bins

        dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
        if dst_bin_width == 0.0:
            return 0.0

        src_bin = torch.arange(self.bins, device=self.histogram.device)
        # distances from the beginning of first dst_bin to the beginning and
        # end of src_bin
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width

        # which dst_bins the beginning and end of src_bin belong to?
        dst_bin_of_begin = torch.clamp(
            torch.div(src_bin_begin, dst_bin_width, rounding_mode="floor"),
            0,
            self.dst_nbins - 1,
        )
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

        dst_bin_of_end = torch.clamp(
            torch.div(src_bin_end, dst_bin_width, rounding_mode="floor"),
            0,
            self.dst_nbins - 1,
        )
        density = self.histogram / bin_width

        norm = torch.zeros(self.bins, device=self.histogram.device)

        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += self._get_norm(
            delta_begin,
            torch.ones(self.bins, device=self.histogram.device) * delta_end,
            density,
        )

        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self._get_norm(
            torch.tensor(-dst_bin_width / 2), torch.tensor(dst_bin_width / 2), density
        )

        dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2

        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += self._get_norm(torch.tensor(delta_begin), delta_end, density)

        return norm.sum().item()

    def _non_linear_param_search(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Non-linear parameter search.

        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
        assert self.histogram.size()[0] == self.bins, "bins mismatch"
        bin_width = (self.max_val - self.min_val) / self.bins

        # cumulative sum
        total = torch.sum(self.histogram).item()
        cSum = torch.cumsum(self.histogram, dim=0)

        stepsize = 1e-5  # granularity
        alpha = 0.0  # lower bound
        beta = 1.0  # upper bound
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float("inf")

        while alpha < beta:
            # Find the next step
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize

            # find the left and right bins between the quantile bounds
            l = start_bin
            r = end_bin
            while l < end_bin and cSum[l] < next_alpha * total:
                l = l + 1
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1

            # decide the next move
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (l - start_bin) > (end_bin - r):
                # move the start bin
                next_start_bin = l
                alpha = next_alpha
            else:
                # move the end bin
                next_end_bin = r
                beta = next_beta

            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue

            # calculate the quantization error using next_start_bin and next_end_bin
            norm = self._compute_quantization_error(next_start_bin, next_end_bin)

            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return new_min, new_max

    def _upscale_histogram(
        self,
        histogram: torch.Tensor,
        orig_min: torch.Tensor,
        orig_max: torch.Tensor,
        update_min: torch.Tensor,
        update_max: torch.Tensor,
    ):
        # this turns the histogram into a more fine-coarsed histogram to reduce
        # bin quantization errors
        histogram = histogram.repeat_interleave(self.upsample_rate) / self.upsample_rate
        bin_size = (orig_max - orig_min) / (self.bins * self.upsample_rate)
        mid_points_histogram = (
            torch.linspace(
                orig_min,
                orig_max,
                self.bins * self.upsample_rate + 1,
                device=orig_min.device,
            )[:-1].to(histogram.device)
            + 0.5 * bin_size
        )
        boundaries_new_histogram = torch.linspace(
            update_min, update_max, self.bins + 1, device=update_min.device
        ).to(histogram.device)
        # this maps the mid-poits of the histogram to the new histogram's space
        bucket_assignments = (
            torch.bucketize(mid_points_histogram, boundaries_new_histogram, right=True)
            - 1
        )
        # this then maps the histogram mid-points in the new space, weighted by the original histogram's values
        # this is just the old histogram in the new histogram's space

        # In case due to numerical issues the values land higher/lower than the maximum/minimum
        bucket_assignments[bucket_assignments >= self.bins] = self.bins - 1
        bucket_assignments[bucket_assignments < 0] = 0

        update_histogram = torch.bincount(
            bucket_assignments, weights=histogram, minlength=self.bins
        )
        return update_histogram

    def _combine_histograms(
        self,
        orig_hist: torch.Tensor,
        orig_min: torch.Tensor,
        orig_max: torch.Tensor,
        update_hist: torch.Tensor,
        update_min: torch.Tensor,
        update_max: torch.Tensor,
    ) -> torch.Tensor:
        # If the new min and max are the same as the current min and max,
        # we can just add the new histogram to the original histogram
        if update_min == orig_min and update_max == orig_max:
            return orig_hist + update_hist

        # If the orig hist only has one value (i.e., the min and max are the same)
        # we can just add it into new histogram
        if orig_min == orig_max:
            bin_value = torch.sum(orig_hist)
            transformed_orig_hist = (
                torch.histc(orig_min, bins=self.bins, min=update_min, max=update_max)  # type: ignore[arg-type]
                * bin_value
            )
            return transformed_orig_hist + update_hist

        # We assume the update_hist is already in the target range, we will map the orig_max to it
        assert update_min <= orig_min
        assert update_max >= orig_max

        # Now we need to turn the old_histogram, into the range of the new histogram
        transformed_orig_hist = self._upscale_histogram(
            orig_hist,
            orig_min,
            orig_max,
            update_min,
            update_max,
        )

        return update_hist + transformed_orig_hist

    def reset_histogram(
        self, x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> None:
        self.min_val = min_val.data
        self.max_val = max_val.data
        assert min_val.numel() == 1 and max_val.numel() == 1, (
            "histogram min/max values must be scalar."
        )
        new_histogram = torch.histc(x, self.bins, min=min_val, max=max_val)  # type: ignore[arg-type]
        self.histogram = new_histogram

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:  # pyre-ignore[14]
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        x_min, x_max = torch.aminmax(x)
        # want to ignore torch.inf since we don't actually
        # want to make our quantization range infinite
        # and in practice those values will be clamped
        if x_min == -torch.inf or x_max == torch.inf:
            warnings.warn("torch.inf detected in input tensor, ignoring input")
            x = x[x.abs() != torch.inf]
            if x.numel() == 0:
                return x_orig
            x_min, x_max = torch.aminmax(x)

        if not hasattr(self, "min_val"):
            self.reset_histogram(x, x_min, x_max)
        else:
            current_min = self.min_val
            current_max = self.max_val
            update_min, update_max = x_min, x_max
            new_min = torch.min(current_min, update_min)
            new_max = torch.max(current_max, update_max)

            # TODO: For some reason, this is required for it to pass torchscript test
            # new_min and new_max should already have requires_grad set to False
            new_min, new_max = new_min.detach(), new_max.detach()
            update_histogram = torch.histc(
                x,
                self.bins,
                min=new_min,  # type: ignore[arg-type]
                max=new_max,  # type: ignore[arg-type]
            ).to(self.histogram.device)
            if new_min == current_min and new_max == current_max:
                combined_histogram = self.histogram + update_histogram
                self.histogram.detach_().resize_(combined_histogram.shape)
                self.histogram.copy_(combined_histogram)
            else:
                combined_histogram = self._combine_histograms(
                    self.histogram,
                    current_min,
                    current_max,
                    update_histogram,
                    new_min,
                    new_max,
                )
                self.histogram.detach_().resize_(combined_histogram.shape)
                self.histogram.copy_(combined_histogram)
                self.min_val.detach_().resize_(new_min.shape)
                self.min_val.copy_(new_min)
                self.max_val.detach_().resize_(new_max.shape)
                self.max_val.copy_(new_max)

        return x_orig

    def calculate_qparams(self) -> Tuple[Tensor, Tensor]:
        assert hasattr(self, "min_val"), (
            "Expecting the observer has min_val and max_val, "
            "please run the observer before calling calculate_qparams"
        )
        assert self.bins == len(self.histogram), (
            "The number of bins in histogram should be equal to the number of bins "
            "supplied while making this observer"
        )
        new_min, new_max = self._non_linear_param_search()
        return choose_qparams_affine_with_min_max(
            new_min,
            new_max,
            self.mapping_type,
            [],  # BlockSize is not needed because the min/max are already reduced
            self.target_dtype,
            self.quant_min,
            self.quant_max,
            self.eps,
            self.scale_dtype,
            self.zero_point_dtype,
            self.preserve_zero,
            self.zero_point_domain,
        )
