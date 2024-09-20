# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
from json import decoder
import logging
import math
import re
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import II, MISSING, open_dict

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer
from fairseq.tasks import FairseqTask

from fairseq.EncoderContrastive import AutoEncoder

from fairseq.models.wav2vec.effnet import EffNetMean
from fairseq.models.wav2vec.resnet import ResNet

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to wav2vec 2.0 model"}
    )
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside wav2vec 2.0 model"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside wav2vec 2.0 model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside wav2vec 2.0 model"
        },
    )
    conv_feature_layers: Optional[str] = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": (
                "string describing convolutional feature extraction "
                "layers in form of a python list that contains "
                "[(dim, kernel_size, stride), ...]"
            ),
        },
    )
    encoder_embed_dim: Optional[int] = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask (normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    require_same_masks: bool = field(
        default=True,
        metadata={
            "help": "whether to number of masked timesteps must be the same across all "
            "examples in a batch"
        },
    )
    mask_dropout: float = field(
        default=0.0,
        metadata={"help": "percent of masks to unmask for each sample"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )
    feature_grad_mult: float = field(
        default=0.0, metadata={"help": "reset feature grad mult in wav2vec 2.0 to this"}
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a layer in wav2vec 2.0"}
    )
    mask_channel_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    mask_channel_before: bool = False
    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    # this holds the loaded wav2vec args
    w2v_args: Any = None
    offload_activations: bool = field(
        default=False, metadata={"help": "offload_activations"}
    )
    min_params_to_wrap: int = field(
        default=int(1e8),
        metadata={
            "help": "minimum number of params for a layer to be wrapped with FSDP() when "
            "training with --ddp-backend=fully_sharded. Smaller values will "
            "improve memory efficiency, but may make torch.distributed "
            "communication less efficient due to smaller input sizes. This option "
            "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
            "--offload-activations are passed."
        },
    )

    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )
    ddp_backend: str = II("distributed_training.ddp_backend")

@dataclass
class RoPADetConfig(BaseConfig):
    tensor_fusion: bool = field(
        default=False, metadata={"help": "use tensor fusion layer"}
    )
    profile_extractor_path: str = field(
        default=MISSING, metadata={"help": "path to wav2vec 2.0 model"}
    )
    npn_option: int = field(
        default=2, metadata={"help": "Option for NPN, including 1, 2, and 3"}
    )
    batch_mask: bool = field(
        default=True, metadata={"help": "use profile batch mask for NPN contrastive loss"}
    )
    use_attention: bool = field(
        default=False, metadata={"help": "whether to use attention or just concatenate"}
    )
    use_profile_attention: bool = field(
        default=False, metadata={"help": "whether to use attention between samples for profile"}
    )
    use_cossim_attention: bool = field(
        default=False, metadata={"help": "whether to use cos sim attention between samples for profile"}
    )
    use_dot_softmax_attention: bool = field(
        default=False, metadata={"help": "whether to use dot softmax attention between samples for profile"}
    )
    clf_hidden_dim: int = field(
        default=64, metadata={'help': 'classifier head hidden dimension'}
    )
    clf_dropout_rate: float = field(
        default=0.1, metadata={'help': 'classifier head dropout rate'}
    )
    clf_output_dim: int = field(
        default=2, metadata={'help': 'classifier head output dimension'}
    )
    decoder_embed_dim: int = field(
        default=256, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num of decoder layers"})
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    autoregressive: bool = II("task.autoregressive")
    num_classes: int = field(
        default=2,
        metadata={
            "help": "number of output classes"
        }
    )


@register_model("MultiBranch", dataclass=RoPADetConfig)
class MultiBranch(BaseFairseqModel):
    def __init__(self, encoder_one, encoder_two, decoder):
        super().__init__()

        self.encoder_one = encoder_one
        self.encoder_two = encoder_two
        self.decoder = decoder

    @classmethod
    def build_model(cls, cfg: RoPADetConfig, task: FairseqTask):
        """Build a new model instance."""

        encoder_one = cls.build_encoder(cfg)
        encoder_two = cls.build_encoder(cfg)

        decoder = cls.build_decoder(cfg, task)

        return MultiBranch(encoder_one, encoder_two, decoder)

    @classmethod
    def build_encoder(cls, cfg: BaseConfig):
        return RoPADetEncoder(cfg)

    @classmethod
    def build_decoder(cls, cfg: RoPADetConfig, task: FairseqTask):
        model = torch.nn.Sequential(
            torch.nn.Linear(cfg.decoder_embed_dim * 2, cfg.clf_hidden_dim*2),
            # torch.nn.BatchNorm1d(cfg.clf_hidden_dim*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=cfg.clf_dropout_rate),
            torch.nn.Linear(cfg.clf_hidden_dim*2, cfg.clf_output_dim),
            # torch.nn.Linear(cfg.decoder_embed_dim, cfg.clf_output_dim),
        )

        return model

    def forward(self, **kwargs):
        encoder_one_out = self.encoder_one(**kwargs)
        encoder_two_out = self.encoder_two(**kwargs)

        # In case batch size == 1, add a dimension for batch
        if len(encoder_one_out['encoder_out'].shape) == 1:
            encoder_one_out['encoder_out'] = encoder_one_out['encoder_out'].unsqueeze(dim=0)

        if len(encoder_two_out['encoder_out'].shape) == 1:
            encoder_two_out['encoder_out'] = encoder_two_out['encoder_out'].unsqueeze(dim=0)

        decoder_input = torch.cat((encoder_one_out['encoder_out'], encoder_two_out['encoder_out']), dim=1)

        decoder_out = self.decoder(decoder_input)

        return decoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict


@register_model("onlyNeuProNet", dataclass=RoPADetConfig)
class onlyNeuProNet(BaseFairseqModel):
    def __init__(self, linear_clf, profile_extractor, users_profile, npn_option, auto_encoder, sup_contrast, use_attention, use_profile_attention, use_cossim_attention, use_dot_softmax_attention):
        super().__init__()

        self.linear_clf = linear_clf
        self.profile_extractor = profile_extractor
        self.users_profile = users_profile
        self.npn_option = npn_option
        self.use_attention = use_attention
        self.use_profile_attention = use_profile_attention
        self.use_cossim_attention = use_cossim_attention
        self.use_dot_softmax_attention = use_dot_softmax_attention

        self.auto_encoder = auto_encoder
        if self.auto_encoder:
            self.ae = AutoEncoder(input_dim=192)
            self.ae2hidden = nn.Linear(8, 384)
            self.cnn2hidden = nn.Linear(384,384)

        self.sup_contrast = sup_contrast

    @classmethod
    def build_model(cls, cfg: RoPADetConfig, task: FairseqTask):
        profile_extractor_cfg = cfg.copy()
        profile_extractor_cfg.w2v_path = profile_extractor_cfg.profile_extractor_path
        profile_extractor = RoPADetEncoder(profile_extractor_cfg)

        linear_clf = torch.nn.Sequential(
            torch.nn.Linear(cfg.decoder_embed_dim, cfg.clf_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=cfg.clf_dropout_rate),
            torch.nn.Linear(cfg.clf_hidden_dim, cfg.clf_output_dim)
        )

        return onlyNeuProNet(linear_clf, profile_extractor, True, cfg.npn_option, task.cfg.auto_encoder, task.cfg.sup_contrast, cfg.use_attention, cfg.use_profile_attention, cfg.use_cossim_attention, cfg.use_dot_softmax_attention)

    def forward(self, **kwargs):
        pre_embed = self.profile_extractor(**kwargs)['encoder_out']

        # In case batch size == 1, add a dimension for batch
        if len(pre_embed.shape) == 1:
            pre_embed = pre_embed.unsqueeze(dim=0)

        profiles = []
        # Create profile embedding using softmax attention pooling:
        for i in range(kwargs['source'].shape[0]):
            curr_profile_idx = np.where(np.array(kwargs['profile']) == kwargs["profile"][i])[0]
            curr_profiles = pre_embed[curr_profile_idx,...]

            dot_product = torch.matmul(curr_profiles, pre_embed[i])
            exp_sim = F.softmax(dot_product / 1.0)
            exp_sim = exp_sim.unsqueeze(dim=1)
            curr_profiles = exp_sim * curr_profiles
            curr_profile = torch.sum(curr_profiles, dim=0).squeeze()

            profiles.append(curr_profile)

        profiles_tensor = torch.stack(profiles).to(pre_embed.get_device())

        out = self.linear_clf(profiles_tensor)
        return out


@dataclass
class CNN_NPNConfig(RoPADetConfig):
    cnn_arch: str = field(
        default="EffNet", metadata={"help": "CNN architecture for downstream branch"}
    )
    

@register_model("CNN_NPN", dataclass=CNN_NPNConfig)
class CNN_NPN(BaseFairseqModel):
    def __init__(self, encoder, decoder, profile_extractor, users_profile, npn_option, auto_encoder, sup_contrast, use_attention, use_profile_attention, use_cossim_attention, use_dot_softmax_attention, cfg):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.profile_extractor = profile_extractor
        self.users_profile = users_profile
        self.npn_option = npn_option
        self.use_attention = use_attention
        self.use_profile_attention = use_profile_attention
        self.use_cossim_attention = use_cossim_attention
        self.use_dot_softmax_attention = use_dot_softmax_attention

        self.auto_encoder = auto_encoder
        if self.auto_encoder:
            self.ae = AutoEncoder(input_dim=192)
            self.ae2hidden = nn.Linear(8, 384)
            self.cnn2hidden = nn.Linear(384,384)

        self.sup_contrast = sup_contrast
        self.cfg = cfg

    @classmethod
    def build_model(cls, cfg: RoPADetConfig, task: FairseqTask):
        """Build a new model instance."""

        encoder = cls.build_encoder(cfg)

        decoder = cls.build_decoder(cfg, task)

        if task.cfg.profiling:
            profile_extractor = cls.build_profile_extractor(cfg)
            users_profile = True
        else:
            profile_extractor = None
            users_profile = None

        return CNN_NPN(encoder, decoder, profile_extractor, users_profile, cfg.npn_option, task.cfg.auto_encoder, task.cfg.sup_contrast, cfg.use_attention, cfg.use_profile_attention, cfg.use_cossim_attention, cfg.use_dot_softmax_attention, cfg)

    @classmethod
    def build_encoder(cls, cfg: BaseConfig):
        if cfg.cnn_arch == 'EffNet':
            return EffNetMean(label_dim=256, level=0, pretrain=False).to('cuda')
        elif cfg.cnn_arch == 'ResNet':
            return ResNet(pretrain=False).to('cuda')

    @classmethod
    def build_profile_extractor(cls, cfg: RoPADetConfig):
        profile_extractor_cfg = cfg.copy()
        # profile_extractor_cfg.freeze_finetune_updates = 500000
        profile_extractor_cfg.w2v_path = profile_extractor_cfg.profile_extractor_path
        profile_extractor_cfg.no_pretrained_weights = False
        return RoPADetEncoder(profile_extractor_cfg)

    @classmethod
    def build_decoder(cls, cfg: RoPADetConfig, task: FairseqTask):
        # return TransformerDecoder(cfg, tgt_dict, embed_tokens)
        if task.cfg.profiling:
            if cfg.use_attention:
                model = torch.nn.Sequential(
                    torch.nn.Linear(cfg.decoder_embed_dim, cfg.clf_hidden_dim),
                    # torch.nn.BatchNorm1d(cfg.clf_hidden_dim*2),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=cfg.clf_dropout_rate),
                    torch.nn.Linear(cfg.clf_hidden_dim, cfg.clf_output_dim),
                    # torch.nn.Linear(cfg.decoder_embed_dim, cfg.clf_output_dim),
                )
            elif cfg.tensor_fusion:
                model = InfoSharing(cfg)
            else:
                model = torch.nn.Sequential(
                    torch.nn.Linear(cfg.decoder_embed_dim * 2, cfg.clf_hidden_dim*2),
                    # torch.nn.BatchNorm1d(cfg.clf_hidden_dim*2),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=cfg.clf_dropout_rate),
                    torch.nn.Linear(cfg.clf_hidden_dim*2, cfg.clf_output_dim),
                    # torch.nn.Linear(cfg.decoder_embed_dim, cfg.clf_output_dim),
                )
        else:
            model = torch.nn.Sequential(
                # torch.nn.Linear(cfg.decoder_embed_dim, cfg.clf_hidden_dim),
                # # torch.nn.BatchNorm1d(cfg.clf_hidden_dim),
                # torch.nn.ReLU(),
                # ## torch.nn.Tanh(),
                # torch.nn.Dropout(p=cfg.clf_dropout_rate),
                # torch.nn.Linear(cfg.clf_hidden_dim, cfg.clf_output_dim),

                # torch.nn.Linear(cfg.decoder_embed_dim, cfg.clf_hidden_dim),
                # torch.nn.ReLU(),
                # torch.nn.Dropout(p=cfg.clf_dropout_rate),
                # torch.nn.Linear(cfg.clf_hidden_dim, cfg.clf_hidden_dim),
                # torch.nn.ReLU(),

                torch.nn.Linear(cfg.decoder_embed_dim, cfg.clf_output_dim),
            )
        return model

    def forward(self, **kwargs):
        # start = time.time()
        encoder_out = self.encoder(kwargs['source'])
        # print(f"ENCODER TIME: {time.time() - start}s")
        # print("PADDING MASK SHAPE: ", kwargs['padding_mask'].shape)

        # In case batch size == 1, add a dimension for batch
        if len(encoder_out.shape) == 1:
            encoder_out = encoder_out.unsqueeze(dim=0)

        if self.use_cossim_attention:
            encoder_out = F.normalize(encoder_out, dim=1)

        if self.users_profile:
            # NOTE: NPN OPTION 1
            if self.npn_option == 1:
                lst_samples = [[kwargs['dataset'][idx] for idx in profile_group] for profile_group in kwargs['profile_group']]
                batches = [kwargs['dataset'].collater(samples) for samples in lst_samples]
                profiles = []

                for batch in batches:
                    batch = utils.move_to_cuda(batch)
                    curr_profiles = self.profile_extractor(**batch["net_input"])['encoder_out']

                    # Average when there are more than one sample contribute to profile
                    if len(curr_profiles.shape) == 2:
                        curr_profile = torch.mean(curr_profiles, dim=0).squeeze()
                    else:
                        curr_profile = curr_profiles
                    profiles.append(curr_profile)

                profiles_tensor = torch.stack(profiles).to(encoder_out.get_device())

            elif self.npn_option == 2:
                all_profiles = self.profile_extractor(**kwargs)['encoder_out']
                profiles = []
                # profiles_before_attn_pool = []
                for i in range(kwargs['source'].shape[0]):
                    curr_profile_idx = np.where(np.array(kwargs['profile']) == kwargs["profile"][i])[0]
                    curr_profiles = all_profiles[curr_profile_idx,...]

                    # Average when there are more than one sample contribute to profile
                    if len(curr_profiles.shape) == 2:
                        if self.use_profile_attention:
                            # print("BEFORE: ", curr_profiles.shape)
                            # print("AFTER: ", F.softmax(curr_profiles, dim=1).shape)
                            curr_profiles = curr_profiles + encoder_out[i] * F.softmax(curr_profiles, dim=1)
                        if self.use_cossim_attention:
                            curr_profiles = F.normalize(curr_profiles, dim=1)
                            cos_sim = torch.nn.CosineSimilarity(dim=1)(encoder_out[i],curr_profiles)
                            cos_sim = cos_sim.unsqueeze(dim=1)
                            curr_profiles = cos_sim * curr_profiles
                        if self.use_dot_softmax_attention:
                            # curr_profiles = F.normalize(curr_profiles, dim=1)
                            # curr_index_encoder_out = encoder_out[i].detach()
                            # dot_product = torch.matmul(curr_profiles, curr_index_encoder_out)
                            dot_product = torch.matmul(curr_profiles, encoder_out[i])
                            exp_sim = F.softmax(dot_product / 5.0)
                            exp_sim = exp_sim.unsqueeze(dim=1)
                            # curr_profile_before_attn_pool = all_profiles[i].clone().squeeze()
                            curr_profiles = exp_sim * curr_profiles
                        # else:
                        if self.use_dot_softmax_attention:
                            curr_profile = torch.sum(curr_profiles, dim=0).squeeze()
                        else:
                            curr_profile = torch.mean(curr_profiles, dim=0).squeeze()
                    else:
                        curr_profile = curr_profiles
                    profiles.append(curr_profile)
                    # profiles_before_attn_pool.append(curr_profile_before_attn_pool)
                    # profiles_before_attn_pool.append(curr_profile)

                profiles_tensor = torch.stack(profiles).to(encoder_out.get_device())
                # profiles_before_attn_pool_tensor = torch.stack(profiles_before_attn_pool).to(encoder_out.get_device())

            if self.use_attention:
                encoder_out = F.normalize(encoder_out, dim=1)
            # profiles_tensor = F.normalize(profiles_tensor, dim=1)
                # print("BEFORE: ", profiles_tensor.shape, torch.std_mean(profiles_tensor))
                # print("AFTER: ", F.softmax(profiles_tensor, dim=1).shape, torch.std_mean(F.softmax(profiles_tensor, dim=1)))
                decoder_input = encoder_out['encoder_out'] + encoder_out['encoder_out'] * F.softmax(profiles_tensor, dim=1)
            elif self.use_cossim_attention:
                profiles_tensor = F.normalize(profiles_tensor, dim=1)
                decoder_input = torch.cat((encoder_out, profiles_tensor), dim=1)
            else:
                decoder_input = torch.cat((encoder_out, profiles_tensor), dim=1)
        else:
            decoder_input = encoder_out

        if self.auto_encoder:
            # source: (batch_size, num_mels, num_timesteps)
            # mask: (batch_size, num_mels, num_timesteps)
            x = kwargs['source']
            mask = kwargs['padding_mask']
            ae_input = torch.sum(x, dim=2)/(mask.shape[-1]-torch.sum(mask, dim=2))
            ae_bottleneck = self.ae.bottleneck(self.ae.encoder(ae_input))

            ae_bottleneck = F.normalize(ae_bottleneck, dim=1)
            decoder_input = F.normalize(decoder_input, dim=1)

            # ae_hidden_output = self.hidden(self.ae2hidden(ae_bottleneck))
            # cnn_hidden_output = self.hidden(self.cnn2hidden(decoder_input))
            # NOTE: Try without the additional hidden layers
            ae_hidden_output = self.ae2hidden(ae_bottleneck)
            cnn_hidden_output = self.cnn2hidden(decoder_input)
            ae_hidden_output = F.relu(ae_hidden_output)
            cnn_hidden_output = F.relu(cnn_hidden_output)
            ae_output = self.ae.output(self.ae.decoder(ae_bottleneck))
            return ae_input, ae_hidden_output, cnn_hidden_output, ae_output, self.decoder(decoder_input)

        if self.sup_contrast:
            return F.normalize(decoder_input, dim=1), self.decoder(decoder_input)

        if self.cfg.tensor_fusion:
            decoder_out = self.decoder(profiles_tensor, encoder_out)
        else:
            decoder_out = self.decoder(decoder_input)
        # return encoder_out.detach().cpu().numpy(), profiles_tensor.detach().cpu().numpy(), decoder_out, profiles_before_attn_pool_tensor.detach().cpu().numpy()
        return decoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict


class InfoSharing(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.post_fusion=torch.nn.Sequential(
                        torch.nn.Linear((int(cfg.decoder_embed_dim/8)) * (int(cfg.decoder_embed_dim/8)), cfg.clf_hidden_dim*2),
                        # torch.nn.BatchNorm1d(cfg.clf_hidden_dim*2),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(p=cfg.clf_dropout_rate),
                        torch.nn.Linear(cfg.clf_hidden_dim*2, cfg.clf_output_dim),
                    )
        self.profile_embed=torch.nn.Linear(cfg.decoder_embed_dim, int(cfg.decoder_embed_dim/8)).to('cuda')
        self.backbone_embed=torch.nn.Linear(cfg.decoder_embed_dim, int(cfg.decoder_embed_dim/8)).to('cuda')

    def forward(self, profiles_tensor, encoder_out):
        profile_embed = self.profile_embed(profiles_tensor)
        backbone_embed = self.backbone_embed(encoder_out)

        # _profile_embed = torch.cat((torch.autograd.Variable(torch.ones(profile_embed.shape[0], 1).type(torch.cuda.FloatTensor), requires_grad=False), profile_embed), dim=1)
        # _backbone_embed = torch.cat((torch.autograd.Variable(torch.ones(backbone_embed.shape[0], 1).type(torch.cuda.FloatTensor), requires_grad=False), backbone_embed), dim=1)

        _profile_embed = profile_embed
        _backbone_embed = backbone_embed

        fusion_tensor = torch.bmm(_profile_embed.unsqueeze(2), _backbone_embed.unsqueeze(1))
        fusion_tensor = fusion_tensor.view(fusion_tensor.shape[0],-1)
        decoder_out = self.post_fusion(fusion_tensor)

        # return decoder_out, profiles_tensor, encoder_out, fusion_tensor
        # return decoder_out, profile_embed, backbone_embed, fusion_tensor
        return decoder_out


@register_model("RoPADet", dataclass=RoPADetConfig)
class RoPADet(BaseFairseqModel):
    def __init__(self, encoder, decoder, profile_extractor, users_profile, npn_option, auto_encoder, sup_contrast, use_attention, use_profile_attention, use_cossim_attention, use_dot_softmax_attention, cfg=None):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.profile_extractor = profile_extractor
        self.users_profile = users_profile
        self.npn_option = npn_option
        self.use_attention = use_attention
        self.use_profile_attention = use_profile_attention
        self.use_cossim_attention = use_cossim_attention
        self.use_dot_softmax_attention = use_dot_softmax_attention

        self.auto_encoder = auto_encoder
        if self.auto_encoder:
            self.ae = AutoEncoder(input_dim=192)
            self.ae2hidden = nn.Linear(8, 384)
            self.cnn2hidden = nn.Linear(384,384)

        self.sup_contrast = sup_contrast
        self.cfg = cfg

    @classmethod
    def build_model(cls, cfg: RoPADetConfig, task: FairseqTask):
        """Build a new model instance."""

        encoder = cls.build_encoder(cfg)

        decoder = cls.build_decoder(cfg, task)

        if task.cfg.profiling:
            profile_extractor = cls.build_profile_extractor(cfg)
            users_profile = True
        else:
            profile_extractor = None
            users_profile = None

        return RoPADet(encoder, decoder, profile_extractor, users_profile, cfg.npn_option, task.cfg.auto_encoder, task.cfg.sup_contrast, cfg.use_attention, cfg.use_profile_attention, cfg.use_cossim_attention, cfg.use_dot_softmax_attention, cfg)

    @classmethod
    def build_encoder(cls, cfg: BaseConfig):
        return RoPADetEncoder(cfg)

    @classmethod
    def build_profile_extractor(cls, cfg: RoPADetConfig):
        profile_extractor_cfg = cfg.copy()
        # profile_extractor_cfg.freeze_finetune_updates = 500_000
        profile_extractor_cfg.w2v_path = profile_extractor_cfg.profile_extractor_path
        profile_extractor_cfg.no_pretrained_weights = False
        return RoPADetEncoder(profile_extractor_cfg)

    @classmethod
    def build_decoder(cls, cfg: RoPADetConfig, task: FairseqTask):
        # return TransformerDecoder(cfg, tgt_dict, embed_tokens)
        if task.cfg.profiling:
            if cfg.use_attention:
                model = torch.nn.Sequential(
                    # torch.nn.Linear(cfg.decoder_embed_dim, cfg.clf_hidden_dim),
                    # # torch.nn.BatchNorm1d(cfg.clf_hidden_dim*2),
                    # torch.nn.ReLU(),
                    # torch.nn.Dropout(p=cfg.clf_dropout_rate),
                    # torch.nn.Linear(cfg.clf_hidden_dim, cfg.clf_output_dim),
                    torch.nn.Linear(cfg.decoder_embed_dim*2, cfg.clf_output_dim),
                )
            elif cfg.tensor_fusion:
                # model = {
                #     "post_fusion" : torch.nn.Sequential(
                #         torch.nn.Linear((int(cfg.decoder_embed_dim/4) + 1) * (int(cfg.decoder_embed_dim/4) + 1), cfg.clf_hidden_dim*2),
                #         # torch.nn.BatchNorm1d(cfg.clf_hidden_dim*2),
                #         torch.nn.ReLU(),
                #         torch.nn.Dropout(p=cfg.clf_dropout_rate),
                #         torch.nn.Linear(cfg.clf_hidden_dim*2, cfg.clf_output_dim),
                #     ).to('cuda'),
                #     "profile_embed": torch.nn.Linear(cfg.decoder_embed_dim, int(cfg.decoder_embed_dim/4)).to('cuda'),
                #     "backbone_embed": torch.nn.Linear(cfg.decoder_embed_dim, int(cfg.decoder_embed_dim/4)).to('cuda'),
                # }
                model = InfoSharing(cfg)
            else:
                model = torch.nn.Sequential(
                    # torch.nn.Linear(cfg.decoder_embed_dim * 2, cfg.clf_hidden_dim*2),
                    # # torch.nn.BatchNorm1d(cfg.clf_hidden_dim*2),
                    # torch.nn.ReLU(),
                    # torch.nn.Dropout(p=cfg.clf_dropout_rate),
                    # torch.nn.Linear(cfg.clf_hidden_dim*2, cfg.clf_output_dim),
                    torch.nn.Linear(cfg.decoder_embed_dim*2, cfg.clf_output_dim),
                )
        else:
            model = torch.nn.Sequential(
                # torch.nn.Linear(cfg.decoder_embed_dim, cfg.clf_hidden_dim),
                # # torch.nn.BatchNorm1d(cfg.clf_hidden_dim),
                # torch.nn.ReLU(),
                # ## torch.nn.Tanh(),
                # torch.nn.Dropout(p=cfg.clf_dropout_rate),
                # torch.nn.Linear(cfg.clf_hidden_dim, cfg.clf_output_dim),

                # torch.nn.Linear(cfg.decoder_embed_dim, cfg.clf_hidden_dim),
                # torch.nn.ReLU(),
                # torch.nn.Dropout(p=cfg.clf_dropout_rate),
                # torch.nn.Linear(cfg.clf_hidden_dim, cfg.clf_hidden_dim),
                # torch.nn.ReLU(),

                torch.nn.Linear(cfg.decoder_embed_dim, cfg.clf_output_dim),
            )
        return model

    def forward(self, **kwargs):
        # start = time.time()
        encoder_out = self.encoder(**kwargs)
        # print(f"ENCODER TIME: {time.time() - start}s")
        # print("PADDING MASK SHAPE: ", kwargs['padding_mask'].shape)

        # In case batch size == 1, add a dimension for batch
        if len(encoder_out['encoder_out'].shape) == 1:
            encoder_out['encoder_out'] = encoder_out['encoder_out'].unsqueeze(dim=0)

        # if self.use_cossim_attention or self.use_dot_softmax_attention:
        #NOTE: test norm, also line 691
        # if self.use_cossim_attention:
        #     encoder_out['encoder_out'] = F.normalize(encoder_out['encoder_out'], dim=1)
        # if self.use_cossim_attention or self.use_dot_softmax_attention:
        #     encoder_out['encoder_out'] = F.normalize(encoder_out['encoder_out'], dim=1)
        if self.users_profile:
            # NOTE: NPN OPTION 1
            if self.npn_option == 1:
                lst_samples = [[kwargs['dataset'][idx] for idx in profile_group] for profile_group in kwargs['profile_group']]
                batches = [kwargs['dataset'].collater(samples) for samples in lst_samples]
                profiles = []

                for batch in batches:
                    batch = utils.move_to_cuda(batch)
                    curr_profiles = self.profile_extractor(**batch["net_input"])['encoder_out']

                    # Average when there are more than one sample contribute to profile
                    if len(curr_profiles.shape) == 2:
                        curr_profile = torch.mean(curr_profiles, dim=0).squeeze()
                    else:
                        curr_profile = curr_profiles
                    profiles.append(curr_profile)

                profiles_tensor = torch.stack(profiles).to(encoder_out['encoder_out'].get_device())

            elif self.npn_option == 2:
                all_profiles = self.profile_extractor(**kwargs)['encoder_out']
                profiles = []
                profiles_before_attn_pool = []
                # print("PROFILE: ", np.array(kwargs['profile']).shape)
                # print("ALL PROFILE: ", all_profiles.shape)
                # print("SOURCE: ", kwargs['source'].shape)
                # print("ENCODER_OUT: ", encoder_out['encoder_out'].shape)
                for i in range(kwargs['source'].shape[0]):
                    curr_profile_idx = np.where(np.array(kwargs['profile']) == kwargs["profile"][i])[0]
                    curr_profiles = all_profiles[curr_profile_idx,...]

                    # Average when there are more than one sample contribute to profile
                    if len(curr_profiles.shape) == 2:
                        if self.use_profile_attention:
                            # print("BEFORE: ", curr_profiles.shape)
                            # print("AFTER: ", F.softmax(curr_profiles, dim=1).shape)
                            curr_profiles = curr_profiles + encoder_out['encoder_out'][i] * F.softmax(curr_profiles, dim=1)
                        if self.use_cossim_attention:
                            curr_profiles = F.normalize(curr_profiles, dim=1)
                            cos_sim = torch.nn.CosineSimilarity(dim=1)(encoder_out['encoder_out'][i],curr_profiles)
                            cos_sim = cos_sim.unsqueeze(dim=1)
                            curr_profiles = (1 - cos_sim) * curr_profiles
                        if self.use_dot_softmax_attention:
                            # curr_profiles = F.normalize(curr_profiles, dim=1)
                            dot_product = torch.matmul(curr_profiles, encoder_out['encoder_out'][i])
                            exp_sim = F.softmax(dot_product / 1.0)
                            exp_sim = exp_sim.unsqueeze(dim=1)
                            curr_profile_before_attn_pool = all_profiles[i].clone().squeeze()
                            curr_profiles = exp_sim * curr_profiles
                        # else:
                        if self.use_dot_softmax_attention:
                            curr_profile = torch.sum(curr_profiles, dim=0).squeeze()
                        else:
                            curr_profile = torch.mean(curr_profiles, dim=0).squeeze()
                    else:
                        curr_profile = curr_profiles
                    profiles.append(curr_profile)
                    profiles_before_attn_pool.append(curr_profile_before_attn_pool)

                profiles_tensor = torch.stack(profiles).to(encoder_out['encoder_out'].get_device())
                profiles_before_attn_pool_tensor = torch.stack(profiles_before_attn_pool).to(encoder_out['encoder_out'].get_device())

            if self.use_attention:
                encoder_out['encoder_out'] = F.normalize(encoder_out['encoder_out'], dim=1)
            # profiles_tensor = F.normalize(profiles_tensor, dim=1)
                print("BEFORE: ", profiles_tensor.shape, torch.std_mean(profiles_tensor))
                print("AFTER: ", F.softmax(profiles_tensor, dim=1).shape, torch.std_mean(F.softmax(profiles_tensor, dim=1)))
                decoder_input = encoder_out['encoder_out'] + encoder_out['encoder_out'] * F.softmax(profiles_tensor, dim=1)
            else:
                decoder_input = torch.cat((encoder_out['encoder_out'], profiles_tensor), dim=1)
        else:
            decoder_input = encoder_out['encoder_out']

        if self.auto_encoder:
            # source: (batch_size, num_mels, num_timesteps)
            # mask: (batch_size, num_mels, num_timesteps)
            x = kwargs['source']
            mask = kwargs['padding_mask']
            ae_input = torch.sum(x, dim=2)/(mask.shape[-1]-torch.sum(mask, dim=2))
            ae_bottleneck = self.ae.bottleneck(self.ae.encoder(ae_input))

            ae_bottleneck = F.normalize(ae_bottleneck, dim=1)
            decoder_input = F.normalize(decoder_input, dim=1)

            # ae_hidden_output = self.hidden(self.ae2hidden(ae_bottleneck))
            # cnn_hidden_output = self.hidden(self.cnn2hidden(decoder_input))
            # NOTE: Try without the additional hidden layers
            ae_hidden_output = self.ae2hidden(ae_bottleneck)
            cnn_hidden_output = self.cnn2hidden(decoder_input)
            ae_hidden_output = F.relu(ae_hidden_output)
            cnn_hidden_output = F.relu(cnn_hidden_output)
            ae_output = self.ae.output(self.ae.decoder(ae_bottleneck))
            return ae_input, ae_hidden_output, cnn_hidden_output, ae_output, self.decoder(decoder_input)

        if self.sup_contrast:
            return F.normalize(decoder_input, dim=1), self.decoder(decoder_input)

        if self.cfg.tensor_fusion:
            decoder_out = self.decoder(profiles_tensor, encoder_out['encoder_out'])
        else:
            decoder_out = self.decoder(decoder_input)
        # return encoder_out['encoder_out'].detach().cpu().numpy(), profiles_tensor.detach().cpu().numpy(), decoder_out, profiles_before_attn_pool_tensor.detach().cpu().numpy()
        # return backbone_embed.detach().cpu().numpy(), profile_embed.detach().cpu().numpy(), fused_tensor.detach().cpu().numpy(), decoder_out
        return decoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict


class RoPADetEncoder(FairseqEncoder):
    def __init__(self, cfg: BaseConfig, output_size=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "require_same_masks": getattr(cfg, "require_same_masks", True),
            "pct_holes": getattr(cfg, "mask_dropout", 0),
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_before": cfg.mask_channel_before,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "checkpoint_activations": cfg.checkpoint_activations,
            "offload_activations": cfg.offload_activations,
            "min_params_to_wrap": cfg.min_params_to_wrap,
        }

        if cfg.w2v_args is None:
            try:
                state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            except Exception as _:
                # TODO: remove this
                # state = checkpoint_utils.load_checkpoint_to_cpu('/media/data/tungtk2/fairseq/outputs/2022-07-25/00-11-55/checkpoints/checkpoint_best.pt', arg_overrides)
                try:
                    state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path.replace('../../../', '/home/tungtk2/fairseq/'), arg_overrides)
                except Exception as _:
                    state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path.replace('/media/SSD/tungtk2/fairseq/', '/home/tungtk2/fairseq/'), arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args

            logger.info(w2v_args)

        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        model_normalized = w2v_args.task.get(
            "normalize", w2v_args.model.get("normalize", False)
        )
        assert cfg.normalize == model_normalized, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        if hasattr(cfg, "checkpoint_activations") and cfg.checkpoint_activations:
            with open_dict(w2v_args):
                w2v_args.model.checkpoint_activations = cfg.checkpoint_activations

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)
        # TODO: possible changes here
        if w2v_args.task._name == 'stft_audio_pretraining':
            model.remove_pretraining_modules()
            # print("NEW LOADING METHOD, PRINTING MODEL TYPE IN EACH STEP")
            # print("MODEL STYLE IN STEP 1: ", type(model))
            # model = model.encoder
            # print("MODEL STYLE IN STEP 2: ", type(model))
            # model = model.w2v_model
            # print("MODEL STYLE IN STEP 3: ", type(model))
        #     pass
        # else:
        # print("TYPE OF MODEL 3: ", type(model2))
        # print("TASK AND MODEL: ", w2v_args.task, w2v_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        if w2v_args.task._name == 'audio_finetuning':
            self.w2v_model = model.encoder.w2v_model
            d = 256
        else:
            self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        targ_d = None
        self.proj = None

        if output_size is not None:
            targ_d = output_size
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            targ_d = cfg.decoder_embed_dim

        if targ_d is not None:
            self.proj = nn.Linear(d, targ_d)

        self.apply_batch_mask = cfg.batch_mask

    def load_model_weights(self, state, model, cfg):
        if cfg.ddp_backend == "fully_sharded":
            from fairseq.distributed import FullyShardedDataParallel

            for name, module in model.named_modules():
                if "encoder.layers" in name and len(name.split(".")) == 3:
                    # Only for layers, we do a special handling and load the weights one by one
                    # We dont load all weights together as that wont be memory efficient and may
                    # cause oom
                    new_dict = {
                        k.replace(name + ".", ""): v
                        for (k, v) in state["model"].items()
                        if name + "." in k
                    }
                    assert isinstance(module, FullyShardedDataParallel)
                    with module.summon_full_params():
                        module.load_state_dict(new_dict, strict=True)
                    module._reset_lazy_init()

            # Once layers are loaded, filter them out and load everything else.
            r = re.compile("encoder.layers.\d.")
            filtered_list = list(filter(r.match, state["model"].keys()))

            new_big_dict = {
                k: v for (k, v) in state["model"].items() if k not in filtered_list
            }

            model.load_state_dict(new_big_dict, strict=False)
        else:
            if "_ema" in state["model"]:
                del state["model"]["_ema"]
            # if cfg.w2v_args.task._name == 'audio_finetuning':
            #     print("NEW LOADING, MODEL TYPE: ", type(model))
            #     new_dict = {
            #         k.replace('encoder.', ''): v
            #         for (k, v) in state['model'].items()
            #         if k.startswith('encoder.')
            #     }
            # # print("MODEL STATE: ", state["model"].keys())
            # # print("MODEL: ", model)
            
            #     model.load_state_dict(new_dict, strict=True)
            # else:
            model.load_state_dict(state["model"], strict=True)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)

            x = res["x"]
            padding_mask = res["padding_mask"]

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if padding_mask is not None:
            x[padding_mask] = 0
            ntokens = torch.sum(~padding_mask, dim=1, keepdim=True)
            x = torch.sum(x, dim=1) /ntokens.type_as(x)
        else:
            x = torch.mean(x, dim=1)

        x=x.squeeze() # B x 1 x C -> B x C
        # add back batch dim in case batch size == 1
        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)

        return {
            "encoder_out": x,  # B x C
            "padding_mask": padding_mask,  # B x T,
            "layer_results": res["layer_results"],
        }

    def contrastive_loss(self, **kwargs):
        # print(f"GROUP LEN: {len(kwargs['profile_group'])}")
        # for i, group in enumerate(kwargs['profile_group']):
        #     print(f"SAMPLE #{i}: {len(group)} samples in profile.")
        # profiles = kwargs['profile']
        # profile_group = kwargs['profile_group']
        # flatten_profile_group_keys = np.array(profile_group.keys())
        # flatten_profile_group_values = np.array(profile_group.values())
        # padding_mask = kwargs['padding_mask']
        final_loss = 0.0

        # source = kwargs['source']
        # batch_size = source.shape[0]
        # for i in range(batch_size):
        #     idx_to_sample = np.where(flatten_profile_group_keys != profiles[i])
        #     idxs = np.random.choice(idx_to_sample, size=batch_size, replace=False)
        #     flatten_profile_group_values[idxs]
        if self.apply_batch_mask:
            # unique_profiles = sorted(set(kwargs["profile"]), key=lambda x: int(x))
            unique_profiles = sorted(set(kwargs["profile"]), key=lambda x: str(x))
            unique_profiles_dct = {k:v for v, k in enumerate(unique_profiles)}
            batch_mask = [unique_profiles_dct[p] for p in kwargs["profile"]]
            w2v_args = {
                "source": kwargs["source"],
                "padding_mask": kwargs["padding_mask"],
                "batch_mask": batch_mask
            }
        else:
            w2v_args = {
                "source": kwargs["source"],
                "padding_mask": kwargs["padding_mask"],
            }
        with contextlib.ExitStack():
            net_output = self.w2v_model(**w2v_args)
        logits = self.w2v_model.get_logits(net_output).float()
        target = self.w2v_model.get_targets(None, net_output)
            # logits = logits.reshape(-1, batch_size, logits.shape[1])
            # target = target.reshape(-1, batch_size)
            # logits = logits[:,0,:].squeeze()
            # target = target[:,0].squeeze()
            # weights = self.w2v_model.get_target_weights(target, net_output)
            # if torch.is_tensor(weights):
            #     weights = weights.float()
        final_loss = F.cross_entropy(
                                    logits, target, reduction='sum'
                                    )
                # final_loss += loss
        # final_loss /= batch_size
        return final_loss

    def forward_torchscript(self, net_input):
        if torch.jit.is_scripting():
            return self.forward(net_input["source"], net_input["padding_mask"])
        else:
            return self.forward_non_torchscript(net_input)

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out["padding_mask"].index_select(
                0, new_order
            )
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict
