# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING, II, OmegaConf

from fairseq.data import MelAudioDataset
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.data.text_compressor import TextCompressionLevel

from . import FairseqTask, register_task


logger = logging.getLogger(__name__)


@dataclass
class InferredW2vConfig:
    # The following are needed to precompute mask and mask channel indices
    #   before model's forward.
    mask_length: Optional[int] = II("model.mask_length")
    mask_prob: Optional[float] = II("model.mask_prob")
    mask_selection: Optional[str] = II("model.mask_selection")
    mask_other: Optional[float] = II("model.mask_other")
    no_mask_overlap: Optional[bool] = II("model.no_mask_overlap")
    mask_min_space: Optional[int] = II("model.mask_min_space")
    mask_channel_length: Optional[int] = II("model.mask_channel_length")
    mask_channel_prob: Optional[float] = II("model.mask_channel_prob")
    mask_channel_selection: Optional[str] = II("model.mask_channel_selection")
    mask_channel_other: Optional[float] = II("model.mask_channel_other")
    no_mask_channel_overlap: Optional[bool] = II("model.no_mask_channel_overlap")
    mask_channel_min_space: Optional[int] = II("model.mask_channel_min_space")

    conv_feature_layers: Optional[str] = II("model.conv_feature_layers")
    encoder_embed_dim: Optional[int] = II("model.encoder_embed_dim")


@dataclass
class STFTAudioPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load, used for fine-tuning"},
    )
    wave2graph: bool = field(
        default=False,
        metadata={"help": "if set, load mfcc and correlation for wave2graph"}
    )
    adresso_pretrain: bool = field(
        default=False,
        metadata={"help": "if set, load pretrained of adresso"}
    )
    specaug: bool = field(
        default=False,
        metadata={"help": "if set, use SpecAugment for train set"},
    )
    profiling: bool = field(
        default=False,
        metadata={"help": "if set, add profiling branch to downstream model"},
    )
    profiles_path: Optional[str] = field(
        default=None, metadata={"help": "path to the Users Profiles for a dataset"}
    )
    profile_sort: bool = field(
        default = False,
        metadata={"help": "if set, sort and create batches of samples from the same profile ID"}
    )
    # binarized_dataset: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "if true, loads binarized dataset (useful for very large datasets). "
    #         "See examples/wav2vec/scripts/binarize_manifest.sh"
    #     },
    # )
    # sample_rate: int = field(
    #     default=16_000,
    #     metadata={
    #         "help": "target sample rate. audio files will be up/down sampled to this rate"
    #     },
    # )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False, metadata={"help": "pad shorter samples instead of cropping"}
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to skip small examples"}
    )
    num_mels: int = field(
        default=128, metadata={"help": "number of mel frequency bands"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={"help": "number of buckets"},
    )
    precompute_mask_indices: bool = field(
        default=False,
        metadata={
            "help": "flag to compute mask indices in data preparation.",
        },
    )

    inferred_w2v_config: Optional[InferredW2vConfig] = field(
        default=None,
        metadata={
            "help": "wav2vec 2.0 masking arguments used to pre-compute masks (required for TPU)",
        },
    )

    tpu: bool = II("common.tpu")
    text_compression_level: ChoiceEnum([x.name for x in TextCompressionLevel]) = field(
        default="none",
        metadata={
            "help": "compression level for texts (e.g. audio filenames, "
            "target texts): none/low/high (default: none). "
        },
    )


@register_task("stft_audio_pretraining", dataclass=STFTAudioPretrainingConfig)
class STFTAudioPretrainingTask(FairseqTask):
    """ """

    cfg: STFTAudioPretrainingConfig

    @classmethod
    def setup_task(cls, cfg: STFTAudioPretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def _get_mask_precompute_kwargs(self, cfg):
        if self.cfg.precompute_mask_indices or self.cfg.tpu:
            assert (
                cfg.inferred_w2v_config is not None
            ), "inferred_w2v_config must be set"
            return OmegaConf.to_container(
                cfg.inferred_w2v_config, resolve=True, enum_to_str=True
            )
        else:
            return {}

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == "ctc"

        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )
        manifest_path = os.path.join(data_path, "{}.tsv".format(split))

        self.datasets[split] = MelAudioDataset(
            manifest_path=manifest_path,
            # sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.min_sample_size,
            num_mels=self.cfg.num_mels,
            pad=task_cfg.labels is not None or task_cfg.enable_padding,
            normalize=task_cfg.normalize,
            num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
            compute_mask_indices=(self.cfg.precompute_mask_indices or self.cfg.tpu),
            text_compression_level=text_compression_level,
            profiling=self.cfg.profiling,
            specaug = self.cfg.specaug if split == "train" else False,
            profile_sort = self.cfg.profile_sort,
            wave2graph = self.cfg.wave2graph,
            adresso_pretrain = self.cfg.adresso_pretrain,
            **self._get_mask_precompute_kwargs(task_cfg),
        )

        if self.cfg.tpu and task_cfg.inferred_w2v_config.mask_channel_prob == 0.0:
            logger.info(
                "Pretraining on TPUs may suffer convergence "
                "issues when training with `mask_channel_prob` value of "
                "0. You may want to set this to a low value close to 0."
            )

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize

    def build_model(self, model_cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(model_cfg, from_checkpoint)

        actualized_cfg = getattr(model, "cfg", None)
        if actualized_cfg is not None:
            if hasattr(actualized_cfg, "w2v_args"):
                model_cfg.w2v_args = actualized_cfg.w2v_args

        return model
