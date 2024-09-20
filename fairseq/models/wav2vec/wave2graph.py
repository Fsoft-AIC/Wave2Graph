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
from fairseq.models.wav2vec.gnns import GCN, GCN_v2, custom_GCN, GAT

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
class GraphNetworkConfig(FairseqDataclass):
    adj_threshold: float = field(
        default=-1.0, metadata={"help": "threshold to cut off adjacency matrix"}
    )
    graph_network: str = field(
        default='GCN', metadata={"help": "architecture of graph network"}
    )
    graph_num_layers: int = field(
        default=3, metadata={"help": "number of layers of graph network"}
    )
    graph_input_dim: int = field(
        default=800, metadata={"help": "input dim of graph network"}
    )
    graph_hidden_dim: int = field(
        default=2048, metadata={"help": "hidden dim of graph network"}
    )
    graph_output_dim: int = field(
        default=256, metadata={"help": "output dim of graph network"}
    )
    graph_clf_output_dim: int = field(
        default=4, metadata={"help": "num class"}
    )
    graph_use_norm: bool = field(
        default=True, metadata={"help": "use normalization for graph network"}
    )
    graph_dropout_rate: float = field(
        default=0.5, metadata={"help": "dropout rate for graph network"}
    )
    graph_grad_mul: float = field(
        default=1.0, metadata={"help": "learning rate multiplier"}
    )
    graph_n_hidden_dim: int = field(
        default=1, metadata={"help": "num of hidden dim"}
    )
    inner_dropout: float = field(
        default=0.0, metadata={"help": "inner attention dropout rate for GAT"}
    )
    add_self_loops: bool = field(
        default=True, metadata={"help": "add self loops for GAT"}
    )
    gat_n_head: int = field(
        default=1, metadata={"help": "number of attention heads for GAT"}
    )
    cut_off: bool = field(
        default=False, metadata={"help": "cut off input dim"}
    )
    cut_time: bool = field(
        default=False, metadata={"help": "cut off temporal input dim"}
    )
    batch_norm: bool = field(
        default=False, metadata={"help": "use batch norm"}
    )
    act: str = field(
        default='elu', metadata={"help": "activation function of graph network"}
    )
    

@dataclass
class Wave2GraphConfig(BaseConfig, GraphNetworkConfig):
    adresso_fusion_mode: int = field(
        default=0, metadata={"help": "mode for fusion on adresso dataset"}
    )
    tensor_fusion: bool = field(
        default=False, metadata={"help": "use tensor fusion layer"}
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


@register_model("GraphOnlyNet", dataclass=GraphNetworkConfig)
class GraphOnlyNet(BaseFairseqModel):
    def __init__(self, graph_network, decoder, cfg=None):
        super().__init__()
        self.graph_network = graph_network
        self.decoder = decoder
        self.cfg = cfg

    @classmethod
    def build_model(cls, cfg: GraphNetworkConfig, task: FairseqTask):
        # graph_network = GCN(nfeat=cfg.graph_input_dim, hidden_dim=cfg.graph_hidden_dim, nclass=cfg.graph_output_dim, dropout=cfg.graph_dropout_rate, grad_mul=cfg.graph_grad_mul)
        if cfg.graph_network == 'GCN':
            #TODO: set params here
            graph_network = GCN(nfeat=cfg.graph_input_dim, hidden_dim=cfg.graph_hidden_dim, nclass=cfg.graph_output_dim, dropout=cfg.graph_dropout_rate, grad_mul=cfg.graph_grad_mul)
        elif cfg.graph_network == 'GCN_v2':
            #TODO: set params here
            # import pdb; pdb.set_trace()
            graph_network = GCN_v2(nfeat=cfg.graph_input_dim, hidden_dim=cfg.graph_hidden_dim, nclass=cfg.graph_output_dim, dropout=cfg.graph_dropout_rate, grad_mul=cfg.graph_grad_mul, n_hidden_dim=cfg.graph_n_hidden_dim, adj_thresh=cfg.adj_threshold)
        elif cfg.graph_network == 'GAT':
            graph_network = GAT(
                nfeat=cfg.graph_input_dim, 
                hidden_dim=cfg.graph_hidden_dim, 
                nclass=cfg.graph_output_dim, 
                dropout=cfg.graph_dropout_rate, 
                grad_mul=cfg.graph_grad_mul, 
                n_hidden_dim=cfg.graph_n_hidden_dim, 
                adj_thresh=cfg.adj_threshold, 
                inner_dropout=cfg.inner_dropout, 
                add_self_loops=cfg.add_self_loops, 
                n_head=cfg.gat_n_head,
                batch_norm=cfg.batch_norm,
                act=cfg.act,
            )

        decoder = cls.build_decoder(cfg, task)

        return GraphOnlyNet(graph_network, decoder, cfg)

    @classmethod
    def build_decoder(cls, cfg: GraphNetworkConfig, task: FairseqTask):
        model = torch.nn.Sequential(
            torch.nn.Linear(cfg.graph_output_dim, cfg.graph_clf_output_dim),
        )
        return model

    def forward(self, **kwargs):
        if self.cfg.cut_off == True:
            # import pdb; pdb.set_trace()
            kwargs['mfcc'] = kwargs['mfcc'][:,4:,:]
            kwargs['corr'] = kwargs['corr'][:,4:,4:]
        if self.cfg.cut_time == True:
            kwargs['mfcc'] = kwargs['mfcc'][:,:,128:128+256]

        graph_out = self.graph_network(kwargs['mfcc'], kwargs['corr'])
        if len(graph_out.shape) == 1:
            graph_out = graph_out.unsqueeze(dim=0)

        decoder_out = self.decoder(graph_out)

        # return graph_out.detach().cpu().numpy(), graph_out.detach().cpu().numpy(), decoder_out
        return decoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict


@register_model("Wave2Graph", dataclass=Wave2GraphConfig)
class Wave2Graph(BaseFairseqModel):
    def __init__(self, encoder, decoder, graph_network, cfg=None):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.graph_network = graph_network
        self.cfg = cfg

    @classmethod
    def build_model(cls, cfg: Wave2GraphConfig, task: FairseqTask):
        """Build a new model instance."""

        encoder = cls.build_encoder(cfg)

        decoder = cls.build_decoder(cfg, task)

        graph_network = cls.build_graph_network(cfg)

        return Wave2Graph(encoder, decoder, graph_network, cfg)

    @classmethod
    def build_encoder(cls, cfg: BaseConfig):
        return RoPADetEncoder(cfg)

    @classmethod
    def build_graph_network(cls, cfg: Wave2GraphConfig):
        if cfg.graph_network == 'GCN':
            return GCN(nfeat=cfg.graph_input_dim, hidden_dim=cfg.graph_hidden_dim, nclass=cfg.graph_output_dim, dropout=cfg.graph_dropout_rate, grad_mul=cfg.graph_grad_mul)
        elif cfg.graph_network == 'custom_GCN':
            return custom_GCN(nfeat=cfg.graph_input_dim, hidden_dim=cfg.graph_hidden_dim, nclass=cfg.graph_output_dim, dropout=cfg.graph_dropout_rate, grad_mul=cfg.graph_grad_mul)
        elif cfg.graph_network == 'GCN_v2':
            return GCN_v2(nfeat=cfg.graph_input_dim, hidden_dim=cfg.graph_hidden_dim, nclass=cfg.graph_output_dim, dropout=cfg.graph_dropout_rate, grad_mul=cfg.graph_grad_mul, n_hidden_dim=cfg.graph_n_hidden_dim, adj_thresh=cfg.adj_threshold)
        elif cfg.graph_network == 'GAT':
            return GAT(nfeat=cfg.graph_input_dim, hidden_dim=cfg.graph_hidden_dim, nclass=cfg.graph_output_dim, dropout=cfg.graph_dropout_rate, grad_mul=cfg.graph_grad_mul, n_hidden_dim=cfg.graph_n_hidden_dim, adj_thresh=cfg.adj_threshold, inner_dropout=cfg.inner_dropout, add_self_loops=cfg.add_self_loops, n_head=cfg.gat_n_head)

    @classmethod
    def build_decoder(cls, cfg: Wave2GraphConfig, task: FairseqTask):
        model = torch.nn.Sequential(
            # torch.nn.Linear(cfg.decoder_embed_dim * 2, cfg.clf_hidden_dim*2),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(p=cfg.clf_dropout_rate),
            # torch.nn.Linear(cfg.clf_hidden_dim*2, cfg.clf_output_dim),
            torch.nn.Linear(cfg.decoder_embed_dim*2, cfg.clf_output_dim),
        )
        return model

    def forward(self, **kwargs):
        encoder_out = self.encoder(**kwargs)

        # In case batch size == 1, add a dimension for batch
        if len(encoder_out['encoder_out'].shape) == 1:
            encoder_out['encoder_out'] = encoder_out['encoder_out'].unsqueeze(dim=0)

        # print("INPUT: ", kwargs['source'].mean(), kwargs['source'].std(), kwargs['mfcc'].mean(), kwargs['mfcc'].std())
        graph_out = self.graph_network(kwargs['mfcc'], kwargs['corr'])
        if len(graph_out.shape) == 1:
            graph_out = graph_out.unsqueeze(dim=0)

        decoder_input = torch.cat((encoder_out['encoder_out'], graph_out), dim=1)
        decoder_out = self.decoder(decoder_input)
        # print("OUTPUT: ", encoder_out.mean(), encoder_out.std(), graph_out.mean(), graph_out.std())

        # return encoder_out['encoder_out'].detach().cpu().numpy(), graph_out.detach().cpu().numpy(), decoder_out
        return decoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict


@register_model("Wave2GraphADReSSo", dataclass=Wave2GraphConfig)
class Wave2GraphADReSSo(BaseFairseqModel):
    def __init__(self, encoder, decoder, graph_network, cfg=None):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.graph_network = graph_network
        self.cfg = cfg

    @classmethod
    def build_model(cls, cfg: Wave2GraphConfig, task: FairseqTask):
        """Build a new model instance."""

        encoder = cls.build_encoder(cfg)

        decoder = cls.build_decoder(cfg, task)

        graph_network = cls.build_graph_network(cfg)

        return Wave2GraphADReSSo(encoder, decoder, graph_network, cfg)

    @classmethod
    def build_encoder(cls, cfg: BaseConfig):
        return RoPADetEncoder(cfg)

    @classmethod
    def build_graph_network(cls, cfg: Wave2GraphConfig):
        if cfg.graph_network == 'GCN':
            #TODO: set params here
            model = GCN(nfeat=cfg.graph_input_dim, hidden_dim=cfg.graph_hidden_dim, nclass=cfg.graph_output_dim, dropout=cfg.graph_dropout_rate, grad_mul=cfg.graph_grad_mul)
        elif cfg.graph_network == 'GCN_v2':
            #TODO: set params here
            model= GCN_v2(nfeat=cfg.graph_input_dim, hidden_dim=cfg.graph_hidden_dim, nclass=cfg.graph_output_dim, dropout=cfg.graph_dropout_rate, grad_mul=cfg.graph_grad_mul, n_hidden_dim=cfg.graph_n_hidden_dim, adj_thresh=cfg.adj_threshold)
        elif cfg.graph_network == 'GAT':
            model = GAT(
                nfeat=cfg.graph_input_dim, 
                hidden_dim=cfg.graph_hidden_dim, 
                nclass=cfg.graph_output_dim, 
                dropout=cfg.graph_dropout_rate, 
                grad_mul=cfg.graph_grad_mul, 
                n_hidden_dim=cfg.graph_n_hidden_dim, 
                adj_thresh=cfg.adj_threshold, 
                inner_dropout=cfg.inner_dropout, 
                add_self_loops=cfg.add_self_loops, 
                n_head=cfg.gat_n_head
            )
        # import pdb; pdb.set_trace()
        if cfg.graph_input_dim == 800:
            if cfg.graph_network == 'GAT':
                state = torch.load('/cm/shared/tungtk2/fairseq_out/outputs/2023-05-31/17-00-00/checkpoints/checkpoint_best.pt')
            else:
                # state = torch.load('/home/tungtk2/fairseq/outputs/2023-03-20/20-24-31/checkpoints/checkpoint_best.pt')
                state = torch.load('/home/tungtk2/fairseq/outputs/2023-03-21/06-23-51/checkpoints/checkpoint_best.pt')
            dct = {k[k.find('.')+1:]:v for (k,v) in state['model'].items() if k.startswith('graph_network')}
            model.load_state_dict(dct)

        return model

    @classmethod
    def build_decoder(cls, cfg: Wave2GraphConfig, task: FairseqTask):
        if cfg.adresso_fusion_mode == 0:
            hidden_dim = 256*2 + 192
        elif cfg.adresso_fusion_mode in [1, 3]:
            hidden_dim = 256 + 192
        elif cfg.adresso_fusion_mode == 2:
            hidden_dim = 256 * 2
        elif cfg.adresso_fusion_mode in [4, 5]:
            hidden_dim = 256
        elif cfg.adresso_fusion_mode == 6:
            hidden_dim = 192
        elif cfg.adresso_fusion_mode == 7:
            hidden_dim = cfg.graph_output_dim + 192 + 192 + 1582
        elif cfg.adresso_fusion_mode == 8:
            hidden_dim = 192 + 192 + 1582
        model = torch.nn.Sequential(
            # torch.nn.Linear(cfg.decoder_embed_dim * 2, cfg.clf_hidden_dim*2),
            # torch.nn.ReLU(),
            torch.nn.Dropout(p=cfg.clf_dropout_rate),
            # torch.nn.Linear(cfg.clf_hidden_dim*2, cfg.clf_output_dim),
            torch.nn.Linear(hidden_dim, cfg.clf_output_dim),
        )
        if cfg.adresso_fusion_mode in [7, 8]:
            model = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=cfg.clf_dropout_rate),
                torch.nn.Linear(512, cfg.clf_output_dim),
                # torch.nn.Linear(hidden_dim, cfg.clf_output_dim),
            )
        return model

    def forward(self, **kwargs):
        MEL_LEN = 313
        MFCC_LEN = 801
        NUM_K = 5

        encoder_out = []

        for i, source in enumerate(kwargs['source']):
            size = source.shape[-1]
            target_size = MEL_LEN*NUM_K
            diff = size - target_size
            if diff < 0:
                source = torch.cat(
                    [source, source.new_full((source.shape[0], 128, -diff,), 0.0)],
                    dim=1
                )
            source = source.reshape(NUM_K, 128, MEL_LEN)
            padding_mask = (
                torch.BoolTensor(source.shape).fill_(False)
            )
            padding_mask[-1,:, diff:] = True
            out = self.encoder(source, padding_mask)
            encoder_out.append(torch.mean(out['encoder_out'], dim=0))

        # encoder_out = self.encoder(**kwargs)
        encoder_out = torch.stack(encoder_out)

        # In case batch size == 1, add a dimension for batch
        if len(encoder_out.shape) == 1:
            encoder_out = encoder_out.unsqueeze(dim=0)


        N_MFCC = kwargs['mfcc'].shape[1]
        if self.cfg.graph_input_dim == 800:
            kwargs['mfcc'] = kwargs['mfcc'].reshape(kwargs['mfcc'].shape[0]*NUM_K, N_MFCC, 801)[...,:800]
        else:
            kwargs['mfcc'] = kwargs['mfcc'].reshape(kwargs['mfcc'].shape[0]*NUM_K, N_MFCC, 801)
        kwargs['corr'] = kwargs['corr'].reshape(kwargs['corr'].shape[0]*NUM_K, N_MFCC, N_MFCC)
        graph_out = self.graph_network(kwargs['mfcc'], kwargs['corr'])
        graph_out = graph_out.reshape(-1, 5, self.cfg.graph_output_dim)
        graph_out = torch.mean(graph_out, dim =1)

        if len(graph_out.shape) == 1:
            graph_out = graph_out.unsqueeze(dim=0)
        if self.cfg.adresso_fusion_mode == 0:
            decoder_input = torch.cat((encoder_out, graph_out, kwargs['pretrain']), dim=1)
        elif self.cfg.adresso_fusion_mode == 1:
            decoder_input = torch.cat((graph_out, kwargs['pretrain']), dim=1)
        elif self.cfg.adresso_fusion_mode == 2:
            decoder_input = torch.cat((encoder_out, graph_out), dim=1)
        elif self.cfg.adresso_fusion_mode == 3:
            decoder_input = torch.cat((encoder_out, kwargs['pretrain']), dim=1)
        elif self.cfg.adresso_fusion_mode == 4:
            decoder_input = encoder_out
        elif self.cfg.adresso_fusion_mode == 5:
            decoder_input = graph_out
        elif self.cfg.adresso_fusion_mode == 6:
            decoder_input = kwargs['pretrain']
        elif self.cfg.adresso_fusion_mode == 7:
            decoder_input = torch.cat((graph_out, kwargs['pretrain'], kwargs['w2v_pretrain'], kwargs['os_pretrain']), dim=1)
        elif self.cfg.adresso_fusion_mode == 8:
            decoder_input = torch.cat((kwargs['pretrain'], kwargs['w2v_pretrain'], kwargs['os_pretrain']), dim=1)
        decoder_out = self.decoder(decoder_input)

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
