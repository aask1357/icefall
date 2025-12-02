# Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
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


import argparse
import inspect
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import torch
from lhotse import CutSet, load_manifest, load_manifest_lazy, validate
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import (  # noqa F401 For AudioSamples
    AudioSamples,
    OnTheFlyFeatures,
)
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from icefall.utils import str2bool
from custom_fbank import CustomFbank as Fbank, CustomFbankConfig as FbankConfig
# from lhotse import Fbank, FbankConfig


def filter_ipa(ipa: str) -> str:
    ipa = ipa.replace("tʃhʲ", "tʃh")
    ipa = ipa.replace("tɕ", "J").replace("dʑ", "J").replace("tʃh", "C")
    ipa = ipa.replace("kh", "K").replace("th", "T").replace("ph", "P")
    return ipa


class ASRDataset(K2SpeechRecognitionDataset):
    def __init__(self, *args, cutset_text: str = "text", **kwargs):
        super().__init__(*args, **kwargs)
        self.cutset_text = cutset_text

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        batch = super().__getitem__(cuts)

        if self.cutset_text == "text":
            return batch

        text_type = self.cutset_text[len("custom."):]
        _filter_ipa = (text_type == "ipa_filtered")
        if _filter_ipa:
            text_type = "ipa"

        text_list = []
        for cut in batch["supervisions"]["cut"]:
            text = cut.supervisions[0].custom[text_type]
            if _filter_ipa:
                text = filter_ipa(text)
            text_list.append(text)
        batch["supervisions"][self.cutset_text] = text_list
        return batch


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class AsrDataModule:
    """
    DataModule for k2 ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).

    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,
    - on-the-fly feature extraction

    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--data-libri-train",
            type=str2bool,
            default=True,
            help="""When enabled, use LibriSpeech for training.""",
        )
        group.add_argument(
            "--full-libri",
            type=str2bool,
            default=True,
            help="""Used only when --mini-libri is False.When enabled,
            use 960h LibriSpeech. Otherwise, use 100h subset.""",
        )
        group.add_argument(
            "--mini-libri",
            type=str2bool,
            default=False,
            help="True for mini librispeech",
        )
        group.add_argument(
            "--data-libri-dev-clean",
            type=str2bool,
            default=True,
            help="""When enabled, use LibriSpeech-dev-clean for validation.""",
        )
        group.add_argument(
            "--data-libri-dev-other",
            type=str2bool,
            default=True,
            help="""When enabled, use LibriSpeech-dev-other for validation.""",
        )
        group.add_argument(
            "--data-libri-test-clean",
            type=str2bool,
            default=False,
            help="""When enabled, use LibriSpeech-test-clean for test.""",
        )
        group.add_argument(
            "--data-libri-test-other",
            type=str2bool,
            default=False,
            help="""When enabled, use LibriSpeech-test-other for test.""",
        )
        group.add_argument(
            "--data-ksponspeech-train",
            type=str2bool,
            default=False,
            help="When enabled, use kpsonspeech for training.",
        )
        group.add_argument(
            "--data-ksponspeech-dev",
            type=str2bool,
            default=False,
            help="When enabled, use kpsonspeech-dev.",
        )
        group.add_argument(
            "--data-ksponspeech-eval-clean",
            type=str2bool,
            default=False,
            help="When enabled, use kpsonspeech-eval-clean.",
        )
        group.add_argument(
            "--data-ksponspeech-eval-other",
            type=str2bool,
            default=False,
            help="When enabled, use kpsonspeech-eval-clean.",
        )
        group.add_argument(
            "--data-ksponspeech-enhanced-train",
            type=str2bool,
            default=False,
            help="When enabled, use kpsonspeech-enhanced for training.",
        )
        group.add_argument(
            "--data-ksponspeech-enhanced-dev",
            type=str2bool,
            default=False,
            help="When enabled, use kpsonspeech-enhanced-dev.",
        )
        group.add_argument(
            "--data-ksponspeech-enhanced-eval-clean",
            type=str2bool,
            default=False,
            help="When enabled, use kpsonspeech-enhanced-eval-clean.",
        )
        group.add_argument(
            "--data-ksponspeech-enhanced-eval-other",
            type=str2bool,
            default=False,
            help="When enabled, use kpsonspeech-enhanced-eval-other.",
        )
        group.add_argument(
            "--data-zeroth-train",
            type=str2bool,
            default=False,
            help="When enabled, use Zeroth for training.",
        )
        group.add_argument(
            "--data-zeroth-test",
            type=str2bool,
            default=False,
            help="When enabled, use Zeroth for validation.",
        )
        group.add_argument(
            "--data-command-kid-train",
            type=str2bool,
            default=False,
            help="When enabled, use Command-Kid for training.",
        )
        group.add_argument(
            "--data-command-nor-train",
            type=str2bool,
            default=False,
            help="When enabled, use Command-Normal for training.",
        )
        group.add_argument(
            "--data-command-old-train",
            type=str2bool,
            default=False,
            help="When enabled, use Command-Old for training.",
        )
        group.add_argument(
            "--data-freetalk-kid-train",
            type=str2bool,
            default=False,
            help="When enabled, use Freetalk-Kid for training.",
        )
        group.add_argument(
            "--data-freetalk-nor-train",
            type=str2bool,
            default=False,
            help="When enabled, use Freetalk-Normal for training.",
        )
        group.add_argument(
            "--data-reazonspeech-medium-train",
            type=str2bool,
            default=False,
            help="When enabled, use ReazonSpeech-medium for training.",
        )
        group.add_argument(
            "--data-reazonspeech-medium-dev",
            type=str2bool,
            default=False,
            help="When enabled, use ReazonSpeech-dev for validation.",
        )
        group.add_argument(
            "--data-reazonspeech-medium-test",
            type=str2bool,
            default=False,
            help="When enabled, use ReazonSpeech-test for validation.",
        )
        group.add_argument(
            "--data-reazonspeech-large-train",
            type=str2bool,
            default=False,
            help="When enabled, use ReazonSpeech for training.",
        )
        group.add_argument(
            "--data-reazonspeech-large-dev",
            type=str2bool,
            default=False,
            help="When enabled, use ReazonSpeech-dev for validation.",
        )
        group.add_argument(
            "--data-reazonspeech-large-test",
            type=str2bool,
            default=False,
            help="When enabled, use ReazonSpeech-test for validation.",
        )
        group.add_argument(
            "--data-jsut-basic5000",
            type=str2bool,
            default=False,
            help="When enabled, use JSUT-basic5000 for validation.",
        )
        group.add_argument(
            "--data-jsut-basic5000-sudachi",
            type=str2bool,
            default=False,
            help="When enabled, use JSUT-basic5000-Sudachi for validation.",
        )
        group.add_argument(
            "--data-tedxjp-10k",
            type=str2bool,
            default=False,
            help="When enabled, use TEDxJP-10K-v1.1 for validation.",
        )

        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/fbank"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=200.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="When enabled, the batches will come from buckets of "
            "similar duration (saves padding frames).",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=30,
            help="The number of buckets for the DynamicBucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="When enabled, utterances (cuts) will be concatenated "
            "to minimize the amount of padding.",
        )
        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Determines the maximum duration of a concatenated cut "
            "relative to the duration of the longest cut in a batch.",
        )
        group.add_argument(
            "--gap",
            type=float,
            default=1.0,
            help="The amount of padding (in seconds) inserted between "
            "concatenated cuts. This padding is filled with noise when "
            "noise augmentation is used.",
        )
        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
            help="When enabled, use on-the-fly cut mixing and feature "
            "extraction. Will drop existing precomputed feature manifests "
            "if available.",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="When enabled (=default), the examples will be "
            "shuffled for each epoch.",
        )
        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=True,
            help="Whether to drop last batch. Used by sampler.",
        )
        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['supervisions']['cut'] with the cuts that "
            "were used to construct it.",
        )

        group.add_argument(
            "--num-workers",
            type=int,
            default=2,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )

        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=True,
            help="When enabled, use SpecAugment for training dataset.",
        )

        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="Used only when --enable-spec-aug is True. "
            "It specifies the factor for time warping in SpecAugment. "
            "Larger values mean more warping. "
            "A value less than 1 means to disable time warp.",
        )

        group.add_argument(
            "--enable-musan",
            type=str2bool,
            default=True,
            help="When enabled, select noise from MUSAN and mix it"
            "with training dataset. ",
        )

        group.add_argument(
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
            help="AudioSamples or PrecomputedFeatures",
        )
    
        parser.add_argument(
            "--cutset-text",
            type=str,
            default="text",
            help="text|custom.ipa|custom.cjj|custom.ipa_filtered"
        )

        group.add_argument(
            "--min-utt-duration",
            type=float,
            default=0.9,
            help="Minimum utterance duration in seconds.",
        )

        group.add_argument(
            "--max-utt-duration",
            type=float,
            default=31.0,
            help="Maximum utterance duration in seconds.",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        transforms = []
        if self.args.enable_musan:
            logging.info("Enable MUSAN")
            logging.info("About to get Musan cuts")
            cuts_musan = load_manifest(self.args.manifest_dir / "musan_cuts.jsonl.gz")
            transforms.append(
                CutMix(cuts=cuts_musan, p=0.5, snr=(10, 20), preserve_id=True)
            )
        else:
            logging.info("Disable MUSAN")

        if self.args.concatenate_cuts:
            logging.info(
                f"Using cut concatenation with duration factor "
                f"{self.args.duration_factor} and gap {self.args.gap}."
            )
            # Cut concatenation should be the first transform in the list,
            # so that if we e.g. mix noise in, it will fill the gaps between
            # different utterances.
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        input_transforms = []
        if self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info(f"Time warp factor: {self.args.spec_aug_time_warp_factor}")
            # Set the value of num_frame_masks according to Lhotse's version.
            # In different Lhotse's versions, the default of num_frame_masks is
            # different.
            num_frame_masks = 10
            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]
            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2
            logging.info(f"Num frame mask: {num_frame_masks}")
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )
        else:
            logging.info("Disable SpecAugment")

        logging.info("About to create train dataset")
        train = ASRDataset(
            input_strategy=eval(self.args.input_strategy)(),
            cut_transforms=transforms,
            input_transforms=input_transforms,
            return_cuts=self.args.return_cuts,
            cutset_text=self.args.cutset_text,
        )

        if self.args.on_the_fly_feats:
            # NOTE: the PerturbSpeed transform should be added only if we
            # remove it from data prep stage.
            # Add on-the-fly speed perturbation; since originally it would
            # have increased epoch size by 3, we will apply prob 2/3 and use
            # 3x more epochs.
            # Speed perturbation probably should come first before
            # concatenation, but in principle the transforms order doesn't have
            # to be strict (e.g. could be randomized)
            # transforms = [PerturbSpeed(factors=[0.9, 1.1], p=2/3)] + transforms   # noqa
            # Drop feats to be on the safe side.
            train = ASRDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                input_transforms=input_transforms,
                return_cuts=self.args.return_cuts,
                cutset_text=self.args.cutset_text,
            )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                drop_last=self.args.drop_last,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )
        logging.info("About to create train dataloader")

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        transforms = []
        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        logging.info("About to create dev dataset")
        if self.args.on_the_fly_feats:
            validate = ASRDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                return_cuts=self.args.return_cuts,
                cutset_text=self.args.cutset_text,
            )
        else:
            validate = ASRDataset(
                cut_transforms=transforms,
                return_cuts=self.args.return_cuts,
                cutset_text=self.args.cutset_text,
            )
        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.info("About to create dev dataloader")
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=2,
            persistent_workers=False,
        )

        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        test = ASRDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
            if self.args.on_the_fly_feats
            else eval(self.args.input_strategy)(),
            return_cuts=self.args.return_cuts,
            cutset_text=self.args.cutset_text,
        )
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    @lru_cache()
    def train_clean_5_cuts(self) -> CutSet:
        logging.info("mini_librispeech: About to get train-clean-5 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-clean-5.jsonl.gz"
        )

    @lru_cache()
    def train_clean_100_cuts(self) -> CutSet:
        logging.info("About to get train-clean-100 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-clean-100.jsonl.gz"
        )

    @lru_cache()
    def train_clean_360_cuts(self) -> CutSet:
        logging.info("About to get train-clean-360 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-clean-360.jsonl.gz"
        )

    @lru_cache()
    def train_other_500_cuts(self) -> CutSet:
        logging.info("About to get train-other-500 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-other-500.jsonl.gz"
        )

    @lru_cache()
    def train_all_shuf_cuts(self) -> CutSet:
        logging.info(
            "About to get the shuffled train-clean-100, \
            train-clean-360 and train-other-500 cuts"
        )
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-all-shuf.jsonl.gz"
        )

    @lru_cache()
    def dev_clean_2_cuts(self) -> CutSet:
        logging.info("mini_librispeech: About to get dev-clean-2 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_dev-clean-2.jsonl.gz"
        )

    @lru_cache()
    def dev_clean_cuts(self) -> CutSet:
        logging.info("About to get dev-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_dev-clean.jsonl.gz"
        )

    @lru_cache()
    def dev_other_cuts(self) -> CutSet:
        logging.info("About to get dev-other cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_dev-other.jsonl.gz"
        )

    @lru_cache()
    def test_clean_cuts(self) -> CutSet:
        logging.info("About to get test-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_test-clean.jsonl.gz"
        )

    @lru_cache()
    def test_other_cuts(self) -> CutSet:
        logging.info("About to get test-other cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_test-other.jsonl.gz"
        )

    @lru_cache()
    def ksponspeech_train_cuts(self) -> CutSet:
        logging.info("About to get ksponspeech-train cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "ksponspeech_cuts_train.jsonl.gz"
        ).modify_ids(lambda id: f"ksp-train-{id}")

    @lru_cache()
    def ksponspeech_dev_cuts(self) -> CutSet:
        logging.info("About to get ksponspeech-dev cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "ksponspeech_cuts_dev.jsonl.gz"
        ).modify_ids(lambda id: f"ksp-dev-{id}")

    @lru_cache()
    def ksponspeech_eval_clean_cuts(self) -> CutSet:
        logging.info("About to get ksponspeech-eval-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "ksponspeech_cuts_eval_clean.jsonl.gz"
        ).modify_ids(lambda id: f"ksp-eval-clean-{id}")

    @lru_cache()
    def ksponspeech_eval_other_cuts(self) -> CutSet:
        logging.info("About to get ksponspeech-eval-other cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "ksponspeech_cuts_eval_other.jsonl.gz"
        ).modify_ids(lambda id: f"ksp-eval-other-{id}")

    @lru_cache()
    def ksponspeech_enhanced_train_cuts(self) -> CutSet:
        logging.info("About to get ksponspeech-enhanced-train cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "ksponspeech_enhanced_cuts_train.jsonl.gz"
        ).modify_ids(lambda id: f"ksp-enh-train-{id}")

    @lru_cache()
    def ksponspeech_enhanced_dev_cuts(self) -> CutSet:
        logging.info("About to get ksponspeech-enhanced-dev cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "ksponspeech_enhanced_cuts_dev.jsonl.gz"
        ).modify_ids(lambda id: f"ksp-enh-dev-{id}")

    @lru_cache()
    def ksponspeech_enhanced_eval_clean_cuts(self) -> CutSet:
        logging.info("About to get ksponspeech-enhanced-eval-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "ksponspeech_enhanced_cuts_eval_clean.jsonl.gz"
        ).modify_ids(lambda id: f"ksp-enh-eval-clean-{id}")

    @lru_cache()
    def ksponspeech_enhanced_eval_other_cuts(self) -> CutSet:
        logging.info("About to get ksponspeech-enhanced-eval-other cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "ksponspeech_enhanced_cuts_eval_other.jsonl.gz"
        ).modify_ids(lambda id: f"ksp-enh-eval-other-{id}")

    @lru_cache()
    def zeroth_train_cuts(self) -> CutSet:
        logging.info("About to get zeroth-train cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "zeroth_wav_cuts_train.jsonl.gz"
        ).modify_ids(lambda id: f"zeroth-train-{id}")

    @lru_cache()
    def zeroth_test_cuts(self) -> CutSet:
        logging.info("About to get zeroth-test cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "zeroth_wav_cuts_test.jsonl.gz"
        ).modify_ids(lambda id: f"zeroth-test-{id}")

    @lru_cache()
    def command_nor_train_cuts(self) -> CutSet:
        logging.info("About to get command-nor-train cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "command_nor_cuts_train.jsonl.gz"
        ).modify_ids(lambda id: f"cmd-nor-{id}")

    @lru_cache()
    def command_kid_train_cuts(self) -> CutSet:
        logging.info("About to get command-kid-train cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "command_kid_cuts_train.jsonl.gz"
        ).modify_ids(lambda id: f"cmd-kid-{id}")

    @lru_cache()
    def command_old_train_cuts(self) -> CutSet:
        logging.info("About to get command-old-train cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "command_old_cuts_train.jsonl.gz"
        ).modify_ids(lambda id: f"cmd-old-{id}")

    @lru_cache()
    def freetalk_nor_train_cuts(self) -> CutSet:
        logging.info("About to get freetalk-nor-train cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "freetalk_nor_cuts_train.jsonl.gz"
        ).modify_ids(lambda id: f"freetalk-nor-{id}")

    @lru_cache()
    def freetalk_kid_train_cuts(self) -> CutSet:
        logging.info("About to get freetalk-kid-train cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "freetalk_kid_cuts_train.jsonl.gz"
        ).modify_ids(lambda id: f"freetalk-kid-{id}")

    @lru_cache()
    def reazonspeech_medium_train_cuts(self) -> CutSet:
        logging.info("About to get reazonspeech-medium-train cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "reazonspeech_medium_cuts_train.jsonl.gz"
        ).modify_ids(lambda id: f"reazonspeech-medium-train-{id}")

    @lru_cache()
    def reazonspeech_medium_dev_cuts(self) -> CutSet:
        logging.info("About to get reazonspeech-medium-dev cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "reazonspeech_medium_cuts_dev.jsonl.gz"
        ).modify_ids(lambda id: f"reazonspeech-medium-dev-{id}")

    @lru_cache()
    def reazonspeech_medium_test_cuts(self) -> CutSet:
        logging.info("About to get reazonspeech-medium-test cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "reazonspeech_medium_cuts_test.jsonl.gz"
        ).modify_ids(lambda id: f"reazonspeech-medium-test-{id}")

    @lru_cache()
    def reazonspeech_large_train_cuts(self) -> CutSet:
        logging.info("About to get reazonspeech-large-train cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "reazonspeech_large_cuts_train.jsonl.gz"
        ).modify_ids(lambda id: f"reazonspeech-large-train-{id}")

    @lru_cache()
    def reazonspeech_large_dev_cuts(self) -> CutSet:
        logging.info("About to get reazonspeech-large-dev cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "reazonspeech_large_cuts_dev.jsonl.gz"
        ).modify_ids(lambda id: f"reazonspeech-large-dev-{id}")

    @lru_cache()
    def reazonspeech_large_test_cuts(self) -> CutSet:
        logging.info("About to get reazonspeech-large-test cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "reazonspeech_large_cuts_test.jsonl.gz"
        ).modify_ids(lambda id: f"reazonspeech-large-test-{id}")

    @lru_cache()
    def jsut_basic5000_cuts(self) -> CutSet:
        logging.info("About to get jsut-basic5000 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "jsut_basic5000_cuts.jsonl.gz"
        ).modify_ids(lambda id: f"jsut-basic5000-{id}")

    @lru_cache()
    def jsut_basic5000_sudachi_cuts(self) -> CutSet:
        logging.info("About to get jsut-basic5000-sudachi cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "jsut_basic5000_sudachi_cuts.jsonl.gz"
        ).modify_ids(lambda id: f"jsut-basic5000-sudachi-{id}")

    @lru_cache()
    def tedxjp_10k_cuts(self) -> CutSet:
        logging.info("About to get TEDxJP-10K-v1.1 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "tedxjp_10k_v1.1_cuts.jsonl.gz"
        ).modify_ids(lambda id: f"tedxjp-10k-v1.1-{id}")

    # freetalk_old -> PCM data -> not implemented yet.
    # @lru_cache()
    # def freetalk_old_train_cuts(self) -> CutSet:
    #     logging.info("About to get freetalk-old-train cuts")
    #     return load_manifest_lazy(
    #         self.args.manifest_dir / "freetalk_old_cuts_train.jsonl.gz"
    #     )

    def get_train_dataloader(self, args, sp, checkpoints=None) -> DataLoader:
        train_cuts = CutSet()
        if args.data_libri_train:
            train_cuts += self.train_clean_100_cuts()
            if args.full_libri:
                train_cuts += self.train_clean_360_cuts()
                train_cuts += self.train_other_500_cuts()
        if args.data_ksponspeech_train:
            train_cuts += self.ksponspeech_train_cuts()
        if args.data_ksponspeech_enhanced_train:
            train_cuts += self.ksponspeech_enhanced_train_cuts()
        if args.data_zeroth_train:
            train_cuts += self.zeroth_train_cuts()
        if args.data_command_kid_train:
            train_cuts += self.command_kid_train_cuts()
        if args.data_command_nor_train:
            train_cuts += self.command_nor_train_cuts()
        if args.data_command_old_train:
            train_cuts += self.command_old_train_cuts()
        if args.data_freetalk_kid_train:
            train_cuts += self.freetalk_kid_train_cuts()
        if args.data_freetalk_nor_train:
            train_cuts += self.freetalk_nor_train_cuts()
        if args.data_reazonspeech_medium_train:
            train_cuts += self.reazonspeech_medium_train_cuts()
        if args.data_reazonspeech_large_train:
            train_cuts += self.reazonspeech_large_train_cuts()
        def remove_short_and_long_utt(c):
            # Keep only utterances with duration between 1 second and 20 seconds
            #
            # Caution: There is a reason to select 20.0 here. Please see
            # ../local/display_manifest_statistics.py
            #
            # You should use ../local/display_manifest_statistics.py to get
            # an utterance duration distribution for your dataset to select
            # the threshold
            if c.duration < args.min_utt_duration or c.duration > args.max_utt_duration:
                logging.warning(
                    f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
                )
                return False

            # In pruned RNN-T, we require that T >= S
            # where T is the number of feature frames after subsampling
            # and S is the number of tokens in the utterance

            # In ./lstm.py, the conv module uses the following expression
            # for subsampling
            if c.num_frames is not None:
                T = ((c.num_frames - 3) // 2 - 1) // 2
            else:
                T = int(c.duration / FbankConfig.frame_shift)

            if args.cutset_text == "text":
                text = c.supervisions[0].text
            elif args.cutset_text.startswith("custom."):
                text_type = args.cutset_text[7:]
                _filter_ipa = (text_type == "ipa_filtered")
                if _filter_ipa:
                    text_type = "ipa"
                text = c.supervisions[0].custom[text_type]
                if _filter_ipa:
                    text = filter_ipa(text)
            else:
                raise ValueError(f"Unsupported cutset_text: {args.cutset_text}")
            tokens = sp.encode(text, out_type=str)

            if T < len(tokens):
                logging.warning(
                    f"Exclude cut with ID {c.id} from training. "
                    f"Number of frames (before subsampling): {c.num_frames}. "
                    f"Number of frames (after subsampling): {T}. "
                    f"Text: {text}. "
                    f"Tokens: {tokens}. "
                    f"Number of tokens: {len(tokens)}"
                )
                return False
            return True
        train_cuts = train_cuts.filter(remove_short_and_long_utt)

        if getattr(args, "start_batch", 0) > 0 and checkpoints and "sampler" in checkpoints:
            # We only load the sampler's state dict when it loads a checkpoint
            # saved in the middle of an epoch
            sampler_state_dict = checkpoints["sampler"]
        else:
            sampler_state_dict = None
        return self.train_dataloaders(train_cuts, sampler_state_dict=sampler_state_dict)

    def get_valid_dataloader_dict(self, args) -> Dict[str, DataLoader]:
        valid_cuts = dict()
        if args.data_libri_dev_clean:
            valid_cuts["librispeech-dev-clean"] = self.dev_clean_cuts()
        if args.data_libri_dev_other:
            valid_cuts["librispeech-dev-other"] = self.dev_other_cuts()
        if args.data_ksponspeech_dev:
            valid_cuts["ksponspeech-dev"] = self.ksponspeech_dev_cuts()
        if args.data_ksponspeech_enhanced_dev:
            valid_cuts["ksponspeech-enhanced-dev"] = self.ksponspeech_enhanced_dev_cuts()
        if args.data_reazonspeech_medium_dev:
            valid_cuts["reazonspeech-medium-dev"] = self.reazonspeech_medium_dev_cuts()
        if args.data_reazonspeech_large_dev:
            valid_cuts["reazonspeech-large-dev"] = self.reazonspeech_large_dev_cuts()
        if args.data_jsut_basic5000:
            valid_cuts["jsut-basic5000"] = self.jsut_basic5000_cuts()
        if args.data_jsut_basic5000_sudachi:
            valid_cuts["jsut-basic5000-sudachi"] = self.jsut_basic5000_sudachi_cuts()
        if args.data_tedxjp_10k:
            valid_cuts["tedxjp-10k-v1.1"] = self.tedxjp_10k_cuts()
        for cuts in valid_cuts.values():
            validate(cuts)
        return dict(
            (name, self.valid_dataloaders(cuts)) for name, cuts in valid_cuts.items()
        )

    def get_test_dataloader_dict(self, args) -> Dict[str, DataLoader]:
        test_cuts = dict()
        if args.data_libri_test_clean:
            test_cuts["librispeech-test-clean"] = self.test_clean_cuts()
        if args.data_libri_test_other:
            test_cuts["librispeech-test-other"] = self.test_other_cuts()
        if args.data_ksponspeech_eval_clean:
            test_cuts["ksponspeech-eval-clean"] = self.ksponspeech_eval_clean_cuts()
        if args.data_ksponspeech_eval_other:
            test_cuts["ksponspeech-eval-other"] = self.ksponspeech_eval_other_cuts()
        if args.data_ksponspeech_enhanced_eval_clean:
            test_cuts[
                "ksponspeech-enhanced-eval-clean"
            ] = self.ksponspeech_enhanced_eval_clean_cuts()
        if args.data_ksponspeech_enhanced_eval_other:
            test_cuts[
                "ksponspeech-enhanced-eval-other"
            ] = self.ksponspeech_enhanced_eval_other_cuts()
        if args.data_zeroth_test:
            test_cuts["zeroth-test"] = self.zeroth_test_cuts()
        return dict(
            (name, self.test_dataloaders(cuts)) for name, cuts in test_cuts.items()
        )
