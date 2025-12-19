from dataclasses import dataclass
import time
import math
import logging
from lhotse import TorchaudioFbank, TorchaudioFbankConfig
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torchaudio.compliance.kaldi import get_mel_banks

from local.se import Model


def next_power_of_2(x: int) -> int:
    """
    Returns the smallest power of 2 that is greater than x.
    Original source: TorchAudio (torchaudio/compliance/kaldi.py)
    """
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _get_strided(waveform: Tensor, window_size: int, window_shift: int, snip_edges: bool) -> Tensor:
    r"""Given a waveform (2D tensor of size batch_size x num_samples),
    it returns a 3D tensor (batch_size, m, window_size)
    representing how the window is shifted along the waveform. Each row is a frame.
    If snip_edges==False,
    then m = math.round(num_samples / window_shift)
           = (num_samples + (window_shift // 2)) // window_shift

    Args:
        waveform (Tensor): Tensor of size ``num_samples``
        window_size (int): Frame length
        window_shift (int): Frame shift
        snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends.

    Returns:
        Tensor: 2D tensor of size (m, ``window_size``) where each row is a frame
    """
    assert waveform.dim() == 1
    num_samples = waveform.size(0)
    strides = (window_shift * waveform.stride(0), waveform.stride(0))

    if snip_edges:
        if num_samples < window_size:
            return torch.empty((0, 0), dtype=waveform.dtype, device=waveform.device)
        else:
            m = 1 + (num_samples - window_size) // window_shift
    else:
        reversed_waveform = torch.flip(waveform, [0])
        m = (num_samples + (window_shift // 2)) // window_shift
        pad = window_size // 2 - window_shift // 2
        pad_right = reversed_waveform
        if pad > 0:
            # torch.nn.functional.pad returns [2,1,0,1,2] for 'reflect'
            # but we want [2, 1, 0, 0, 1, 2]
            pad_left = reversed_waveform[-pad:]
            waveform = torch.cat((pad_left, waveform, pad_right), dim=0)
        else:
            # pad is negative so we want to trim the waveform at the front
            waveform = torch.cat((waveform[-pad:], pad_right), dim=0)

    sizes = (m, window_size)
    return waveform.as_strided(sizes, strides)


def _get_strided_batch(waveform: Tensor, window_size: int, window_shift: int, snip_edges: bool) -> Tensor:
    r"""Given a waveform (1D tensor of size ``num_samples``), it returns a 2D tensor (m, ``window_size``)
    representing how the window is shifted along the waveform. Each row is a frame.

    Args:
        waveform (Tensor): Tensor of size ``num_samples``
        window_size (int): Frame length
        window_shift (int): Frame shift
        snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends.
            default: False

    Returns:
        Tensor: 2D tensor of size (m, ``window_size``) where each row is a frame
    """
    assert window_size >= window_shift
    B, T = waveform.shape
    num_samples = waveform.size(1)
    if snip_edges:
        if T < window_size:
            return torch.empty((B, 0, 0), dtype=waveform.dtype, device=waveform.device)
        else:
            m = 1 + (T - window_size) // window_shift
    else:
        m = (num_samples + (window_shift // 2)) // window_shift
        pad = m * window_shift + (window_size - window_shift) - num_samples
        if pad > 0:
            # shahn) Just reflect padding. Don't need to insist on Kaldi-style.
            pad_left = pad // 2
            pad_right = pad - pad_left
            waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right), mode='reflect')
        else:
            # pad is negative so we want to trim the waveform at the front
            waveform = waveform[:, -pad:]

    sizes = (B, m, window_size)
    strides = (waveform.size(1), window_shift, 1)

    return waveform.as_strided(sizes, strides)


def _get_strided_batch_fuck(waveform: Tensor, window_size: int, window_shift: int, snip_edges: bool) -> Tensor:
    r"""Given a waveform (1D tensor of size ``num_samples``), it returns a 2D tensor (m, ``window_size``)
    representing how the window is shifted along the waveform. Each row is a frame.

    Args:
        waveform (Tensor): Tensor of size ``num_samples``
        window_size (int): Frame length
        window_shift (int): Frame shift
        snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends.
            default: False

    Returns:
        Tensor: 2D tensor of size (m, ``window_size``) where each row is a frame
    """
    assert window_size >= window_shift
    B, T = waveform.shape
    num_samples = waveform.size(1)
    if snip_edges:
        if T < window_size:
            return torch.empty((B, 0, 0), dtype=waveform.dtype, device=waveform.device)
        else:
            m = 1 + (T - window_size) // window_shift
    else:
        reversed_waveform = torch.flip(waveform, [1])
        m = (T + window_shift - 1) // window_shift
        tmp = (num_samples - window_size + window_shift - num_samples)
        pad = tmp // window_shift * window_shift + window_size - num_samples
        _pad_left = pad // 2
        _pad_right = pad - _pad_left
        pad_right = reversed_waveform[:, :_pad_right]
        # torch.nn.functional.pad returns [2,1,0,1,2] for 'reflect'
        # but we want [2, 1, 0, 0, 1, 2]
        pad_left = reversed_waveform[:, -_pad_left:]
        waveform = torch.cat((pad_left, waveform, pad_right), dim=1)

    sizes = (B, m, window_size)
    strides = (waveform.size(1), window_shift, 1)

    return waveform.as_strided(sizes, strides)


def Q(x: Tensor) -> Tensor:
    x = x.to(torch.float16)
    mag = x.abs()
    return torch.where(mag < 2**-14, torch.zeros_like(x), x).float()


@dataclass
class CustomFbankConfig(TorchaudioFbankConfig):
    snip_edges: bool = False
    version = "abs" # [mag|power|abs]
                    # mag: sqrt(r^2+i^2), power: r^2+i^2, abs: |r|+|i|
    sampling_rate: int = 16_000
    # sampling_rate: int = 8_000
    # frame_length: float = 0.05
    # frame_shift: float = 0.02
    # num_mel_bins: int = 62
    downsample: bool = False
    # lpf_taps: int = 128
    # lpf_cutoff: float = 0.475
    # lpf_beta: float = 8.0
    # low_freq: float = 20.0
    # high_freq: float = 3800.0


class CustomFbank(TorchaudioFbank):
    ''' default Fbank: log(mel(real^2 + img^2).clamp_min(EPS))
    custom Fbank: {2*log(mel(sqrt(real^2 + img^2)).clamp_min(EPS))}.clamp_min(EPS)
    1. Use magnitude instead of power
    2. multiply 2 and clamp so that the dynamic range is similar to that of default Fbank
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        eps = torch.tensor(torch.finfo(torch.float).eps)
        self.eps = eps.item() if self.config.version=="power" else eps.sqrt().item()
        
        self.win_size = int(self.config.frame_length * self.config.sampling_rate)
        self.hop_size = int(self.config.frame_shift * self.config.sampling_rate)
        if self.config.round_to_power_of_two:
            self.n_fft = next_power_of_2(self.win_size)
        else:
            self.n_fft = self.win_size
        
        self.lpf = torch.tensor(0)
        if self.config.downsample:
            from scipy import signal
            with np.errstate(invalid='ignore'):
                lpf = signal.firwin(
                    self.config.lpf_taps,
                    self.config.lpf_cutoff,
                    window=('kaiser', self.config.lpf_beta),
                    pass_zero='lowpass'
                )
                self.lpf = Q(torch.tensor(lpf, dtype=torch.float32)).view(1, 1, -1)
        
        if self.config.window_type == "povey":
            window = torch.hann_window(self.win_size, periodic=False).pow(0.85).unsqueeze(0)
        else:
            window = getattr(torch, self.config.window_type)(win_size).unsqueeze(0)
        k = torch.arange(self.n_fft//2, dtype=torch.float32).unsqueeze(1)
        n = torch.arange(self.win_size, dtype=torch.float32).unsqueeze(0)
        idx = 2 * math.pi / self.n_fft * k * n
        ft_weight = torch.cat([torch.cos(idx), torch.sin(idx)], dim=0)
        self.ft_weight = Q((window * ft_weight))
        
        mel_fbank = get_mel_banks(
            num_bins=self.config.num_mel_bins,
            window_length_padded=self.n_fft,
            sample_freq=self.config.sampling_rate,
            low_freq=self.config.low_freq,
            high_freq=self.config.high_freq,
            vtln_low=self.config.vtln_low,
            vtln_high=self.config.vtln_high,
            vtln_warp_factor=self.config.vtln_warp,
        )[0]
        self.mel_fbank = Q(mel_fbank)
        self._device = self.mel_fbank.device

        self.se = Model().to(self._device)
        state_dict = torch.load(
            "/home/shahn/Documents/trainer/logs/gw_se/spec/v1.1/00500.pth",
            map_location=self._device
        )["model"]
        self.se.load_state_dict(state_dict)
    
    def to(self, device):
        self._device = torch.device(device)
        self.lpf = self.lpf.to(device)
        self.ft_weight = self.ft_weight.to(device)
        self.mel_fbank = self.mel_fbank.to(device)
        self.se = self.se.to(device)
        return self
    
    def float(self):
        self.ft_weight = self.ft_weight.float()
        self.mel_fbank = self.mel_fbank.float()
        self.se = self.se.float()
        return self

    def extract(self, samples, sampling_rate=None):
        # samples: [T_wav] or [1, T_wav] -> [T_wav]
        y = Q(torch.from_numpy(samples)).view(-1)
        
        # downsample
        if self.config.downsample:
            n_pad = (self.config.lpf_taps - 2) // 2
            y = Q(F.conv1d(y.view(1, 1, -1), self.lpf, None, stride=2, padding=n_pad).view(-1))
        
        # dither
        if self.config.dither > 0.0:
            y = Q(y + torch.randn_like(y) * self.config.dither)
        
        y = _get_strided(y, self.win_size, self.hop_size, self.config.snip_edges)   # [T_mel, win_size]
        # remove DC
        if self.config.remove_dc_offset:
            row_means = Q(torch.mean(y, dim=1, keepdim=True))
            y = Q(y - row_means)
        
        # pre-emphasis
        if self.config.preemphasis_coefficient != 0.0:
            y_prev = F.pad(y.unsqueeze(0), (1, 0), mode="replicate").squeeze(0)
            y = Q(y - self.config.preemphasis_coefficient * y_prev[:, :-1])
        
        # window + stft
        y = Q(F.linear(y.float(), self.ft_weight, None))    # [T_mel, n_fft] (real: n_fft//2, imag: n_fft//2 -> concatenated)
        
        # magnitude
        if self.config.version == "power":
            y = Q(Q(y.view(-1, 2, self.n_fft//2).square()).sum(dim=1))
        elif self.config.version == "mag":
            y = Q(y.view(-1, 2, self.n_fft//2).square().sum(dim=1).sqrt())
        elif self.config.version == "abs":
            y = Q(y.view(-1, 2, self.n_fft//2).abs().sum(dim=1))
        
        # mel filterbank
        if sampling_rate == self.config.sampling_rate or sampling_rate is None:
            mel_fbank = self.mel_fbank
        else:
            mel_fbank = Q(get_mel_banks(
                num_bins=self.config.num_mel_bins,
                window_length_padded=self.n_fft,
                sample_freq=sampling_rate,
                low_freq=self.config.low_freq,
                high_freq=self.config.high_freq,
                vtln_low=self.config.vtln_low,
                vtln_high=self.config.vtln_high,
                vtln_warp_factor=self.config.vtln_warp,
            )[0])
        y = Q(F.linear(y, mel_fbank))  # [T_mel, num_mels]
        
        # log
        y = Q(y.clamp_min(self.eps).log())
        if self.config.version != "power":
            y *= 2
        
        return y.numpy()

    def extract_batch(self, samples, sampling_rate=None, lengths=None):
        st = time.perf_counter()
        if lengths is None:
            # samples: List[Tensor]
            if samples[0].dim() > 1:
                samples = [s.view(-1) for s in samples]
            _lengths = [
                (s.size(0) + self.hop_size//2) // self.hop_size for s in samples
            ]
            samples = torch.nn.utils.rnn.pad_sequence(
                samples, batch_first=True, padding_value=0.0)
        sample_device = samples.device
        samples = samples.to(self._device)
        # samples: [B, T_wav]
        y = Q(samples)
        
        # dither
        if self.config.dither > 0.0:
            y = Q(y + torch.randn_like(y) * self.config.dither)
        
        # se
        farend = torch.zeros_like(y)
        y = self.se(farend, y)  # [B, T_spec, F, 2]
        y = y.transpose(2, 3)   # [B, T_spec, 2, F]
        y = y.contiguous()

        # magnitude
        B = y.size(0)
        if self.config.version == "power":
            y = Q(Q(y.view(B, -1, 2, self.n_fft//2).square()).sum(dim=2))
        elif self.config.version == "mag":
            y = Q(y.view(B, -1, 2, self.n_fft//2).square().sum(dim=2).sqrt())
        elif self.config.version == "abs":
            y = Q(y.view(B, -1, 2, self.n_fft//2).abs().sum(dim=2))

        # mel filterbank
        if sampling_rate == self.config.sampling_rate or sampling_rate is None:
            mel_fbank = self.mel_fbank
        else:
            mel_fbank = Q(get_mel_banks(
                num_bins=self.config.num_mel_bins,
                window_length_padded=self.n_fft,
                sample_freq=sampling_rate,
                low_freq=self.config.low_freq,
                high_freq=self.config.high_freq,
                vtln_low=self.config.vtln_low,
                vtln_high=self.config.vtln_high,
                vtln_warp_factor=self.config.vtln_warp,
            )[0]).to(y.device)
        y = Q(F.linear(y, mel_fbank))  # [B, T_mel, num_mels]
        
        # log
        y = Q(y.clamp_min(self.eps).log())
        if self.config.version != "power":
            y *= 2
        
        y = y.to(sample_device)
        if lengths is None:
            y = [y[i, :l] for i, l in enumerate(_lengths)]
        # logging.info(f"extract_batch: {time.perf_counter()-st:.2f} sec")
        return y


if __name__=="__main__":
    extractor = CustomFbank(CustomFbankConfig(num_mel_bins=80))
    samples = np.zeros(16000, dtype=np.float32)
    features = extractor.extract(samples, 16000)
