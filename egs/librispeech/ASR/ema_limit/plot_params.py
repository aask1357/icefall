from typing import Union, Dict
import torch
import numpy as np


def join(name1: str, name2: str):
    if name1 != "":
        return f"{name1}.{name2}"
    else:
        return name2


def plot_params(
    dict_to_plot: dict,
    model: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
    model_name: str = ""
) -> None:
    if isinstance(model, dict):
        for key, value in model.items():
            plot_params(dict_to_plot, key, join(model_name, key))
        return
     
    if hasattr(model, "module"):
        model = model.module

    for param_name, param in model.named_parameters():
        if param.numel() == 0:
            continue
        dict_to_plot[f"param/{join(model_name, param_name)}"] = param.data.detach().float().cpu().numpy()
    for buffer_name, buffer in model.named_buffers():
        if buffer.numel() == 0:
            continue
        if torch.any(torch.isnan(buffer.data)) or torch.any(torch.isinf(buffer.data)):
            continue
        dict_to_plot[f"buffer/{join(model_name, buffer_name)}"] = buffer.data.detach().float().cpu().numpy()


if __name__=="__main__":
    model = torch.nn.Linear(10, 10)
    dict_to_plot = {}
    plot_params(dict_to_plot, model)
    print(dict_to_plot)