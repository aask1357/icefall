import torch


FP16 = False


def q(x: torch.Tensor) -> torch.Tensor:
    if not FP16:
        return x
    x = x.to(torch.float16)
    mag = x.abs()
    return torch.where(mag < 2**-14, torch.zeros_like(x), x).float()


def hook(module: torch.nn.Module, args, output: torch.Tensor) -> torch.Tensor:
    output = q(output)
    return output


def Q(m: torch.nn.Module) -> torch.nn.Module:
    if not FP16:
        return m
    m.register_forward_hook(hook)
    return m


if __name__=="__main__":
    conv = torch.nn.Conv1d(1, 1, 3)
    conv.bias.data.zero_()
    x = torch.randn(1, 1, 3) / 2**-10
    with torch.no_grad():
        y = conv(x)
        print(f"output: {y}")
        print(f"quantized output: {q(y)}")
    conv = Q(conv)
    with torch.no_grad():
        y = conv(x)
        print(f"output after hook: {y}")