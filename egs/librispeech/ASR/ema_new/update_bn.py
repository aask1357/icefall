import logging

import torch
from torch import nn

from asr_datamodule import AsrDataModule


@torch.no_grad()
def update_bn(model: nn.Module, args, sp) -> None:
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum
    
    if not momenta:
        return
    
    # prepare train dataset
    datamodule = AsrDataModule(args)
    # train_cuts = datamodule.train_clean_100_cuts()
    # train_cuts += datamodule.train_clean_360_cuts()
    # train_cuts += datamodule.train_other_500_cuts()
    train_cuts = datamodule.ksponspeech_train_cuts()
    train_cuts += datamodule.zeroth_train_cuts()
    
    def remove_short_and_long_utt(c):
        # Keep only utterances with duration between 1 second and 20 seconds
        #
        # Caution: There is a reason to select 20.0 here. Please see
        # ../local/display_manifest_statistics.py
        #
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        if c.duration < 0.9 or c.duration > 31.0:
            logging.warning(
                f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            )
            return False

        return True
    train_cuts = train_cuts.filter(remove_short_and_long_utt)
    
    train_dl = datamodule.train_dataloaders(train_cuts)
    
    # save model config
    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    # update BN statistics
    MAX_TIME = 10   # hours
    device = next(model.parameters()).device
    time = 0.0
    for batch in train_dl:
        feature = batch["inputs"].to(device)
        assert feature.ndim == 3    # at entry, feature is (N, T, C)
        N, T, C = feature.shape
        time += N * T * 10 / 1000   # 10ms
        feature_lens = batch["supervisions"]["num_frames"].to(device)
        with torch.cuda.amp.autocast(enabled=True):
            _ = model(
                x=feature,
                x_lens=feature_lens,
                warmup=1.0
            )
        print(f"\r{time/60/60:6.2f}/{MAX_TIME} hours", end="", flush=True)
        if time > MAX_TIME * 60 * 60:    # 100 hours
            print("")
            break

    # restore model config
    model.train(was_training)
    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
