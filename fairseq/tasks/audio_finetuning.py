# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import torch
import json

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any

from fairseq.data import AddTargetDataset, Dictionary, encoders
from fairseq.tasks.stft_audio_pretraining import STFTAudioPretrainingConfig, STFTAudioPretrainingTask
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

from . import register_task
from .. import utils
from ..logging import metrics


logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        # print(self.dictionary.encode_line(
        #     label, append_eos=False, add_if_not_exist=False
        # ).type(torch.LongTensor))
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        ).type(torch.LongTensor) - 4


def label_len_fn(label):
    return len(label.split(" "))


@dataclass
class AudioFinetuningConfig(STFTAudioPretrainingConfig):
    auto_encoder: bool = field(
        default=False,
        metadata={"help": "if set, add auto encoder branch to downstream model"},
    )
    sup_contrast: bool = field(
        default=False,
        metadata={"help": "if set, add supervised contrastive loss"},
    )
    # Options for reporting WER metrics during validation. Only applicable to
    # Seq2Seq models during fine-tuning
    eval_wer: bool = field(
        default=False, metadata={"help": "compute WER for Seq2Seq models"}
    )
    eval_wer_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(),
        metadata={"help": "beam search config for evaluating wer during training"},
    )
    eval_wer_tokenizer: Any = field(
        default=None,
        metadata={"help": "tokenizer config for evaluating wer during training"},
    )
    eval_wer_post_process: str = field(
        default="letter",
        metadata={
            "help": "remove BPE tokens before scoring (can be sentencepiece, letter, and more)"
        },
    )
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_detok: Optional[str] = field(
        default=None,
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); "
            "required if using --eval-bleu; use 'space' to disable "
            "detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: str = field(
        default="{}", metadata={"help": "args for building the tokenizer, if needed"}
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None, metadata={"help": "remove BPE before computing BLEU"}
    )
    eval_bleu_args: str = field(
        default="{}",
        metadata={
            "help": "generation args for BLUE scoring, e.g., "
            '\'{"beam": 4, "lenpen": 0.6}\''
        },
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    autoregressive: bool = field(
        default=False,
        metadata={
            "help": "required for autoregressive decoders (like seq2seq models); "
            "adds 'prev_output_tokens' to input and appends eos to target"
        },
    )


@register_task("audio_finetuning", dataclass=AudioFinetuningConfig)
class AudioFinetuningTask(STFTAudioPretrainingTask):
    """ """

    cfg: AudioFinetuningConfig

    def __init__(
        self,
        cfg: AudioFinetuningConfig,
    ):
        super().__init__(cfg)

        dict_path = os.path.join(self.cfg.data, f"dict.{self.cfg.labels}.txt")
        self.label_vocab = Dictionary.load(dict_path)
        # print(self.label_vocab.bos_index, self.label_vocab.pad_index, self.label_vocab.eos_index, self.label_vocab.unk_index)

    def load_dataset(
        self, split: str, task_cfg: AudioFinetuningConfig = None, **kwargs
    ):
        super().load_dataset(split, task_cfg, **kwargs)

        task_cfg = task_cfg or self.cfg
        assert task_cfg.labels is not None
        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )
        data_path = self.cfg.data
        label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
        # if 'notest' in label_path: label_path = label_path.replace('valid', 'test')
        skipped_indices = getattr(self.datasets[split], "skipped_indices", set())
        text_compressor = TextCompressor(level=text_compression_level)

        labels = []
        with open(label_path, "r") as f:
            # for line in f:
            #     label = line.strip()
            #     labels.append(torch.LongTensor([self.label_vocab.add_symbol(label)]))
            labels = [
                text_compressor.compress(l.strip())
                for i, l in enumerate(f)
                if i not in skipped_indices
            ]

        assert len(labels) == len(self.datasets[split]), (
            f"labels length ({len(labels)}) and dataset length "
            f"({len(self.datasets[split])}) do not match"
        )

        process_label = LabelEncoder(self.target_dictionary)

        self.datasets[split] = AddTargetDataset(
            self.datasets[split],
            labels,
            pad=self.target_dictionary.pad(),
            eos=self.target_dictionary.eos(),
            batch_targets=True,
            process_label=process_label,
            label_len_fn=label_len_fn,
            add_to_input=task_cfg.get("autoregressive", False),
            text_compression_level=text_compression_level,
        )

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.label_vocab

