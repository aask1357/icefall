import torch

NUM_EXPONENT_BITS = 4
NUM_MANTISSA_BITS = 3
@torch.no_grad()
def q_fp8(weight: torch.Tensor, layer: str) -> None:
    """Quantize the weights of a layer to FP8 format in-place.

    Args:
        weight (Tensor): The weight to quantize.
    """
    # max_power_of_2 = torch.tensor([7], dtype=weight.dtype, device=weight.device)    # FP8 standard
    # threshold = max(
    #     torch.quantile(weight.data, 0.9999),
    #     -torch.quantile(-weight.data, 0.9999)
    # )
    # max_power_of_2 = threshold.log2().floor()
    max_power_of_2 = weight.abs().max().log2().floor() + 3
    print(f"{layer}: {int(max_power_of_2.item())}")
    max_value = 2 ** max_power_of_2 * (2 - 1/2**NUM_MANTISSA_BITS)
    min_value = 2 ** (max_power_of_2 - 2**NUM_EXPONENT_BITS + 1)
    weight.clamp_(min=-max_value, max=max_value)

    sign = weight.sign()
    weight_abs = weight.abs().clamp_min(min_value)
    exponent = weight_abs.log2().floor()
    mantissa = ((weight_abs / (2 ** exponent) - 1) * 2**NUM_MANTISSA_BITS).round()

    assert torch.all(mantissa >= 0) and torch.all(mantissa <= 2**NUM_MANTISSA_BITS), mantissa
    weight.data.copy_(sign * (2 ** exponent) * (1 + mantissa / 2**NUM_MANTISSA_BITS))
