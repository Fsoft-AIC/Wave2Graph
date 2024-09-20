# Wave2Graph
The official repository for paper *Wave2Graph: Integrating spectral features and correlations for graph-based learning in sound waves*.

Accepted to the AI Open journal.

## Overview
>This paper investigates a novel graph-based representation of sound waves inspired by the physical phenomenon of correlated vibrations. We propose a Wave2Graph framework for integrating multiple acoustic representations, including the spectrum of frequencies and correlations, into various neural computing architectures to achieve new state-of-the-art performances in sound classification. The capability and reliability of our end-to-end framework are evidently demonstrated in voice pathology for low-cost and non-invasive mass-screening of medical conditions, including respiratory illnesses and Alzheimer's Dementia. We conduct extensive experiments on multiple public benchmark datasets (ICBHI and ADReSSo) and our real-world dataset (IJSound: Respiratory disease detection using coughs and breaths). Wave2Graph framework consistently outperforms previous state-of-the-art methods with a large magnitude, up to 7.65% improvement, promising the usefulness of graph-based representation in signal processing and machine learning.

## About this implementation

This source code is based on the [fairseq toolkit](https://github.com/facebookresearch/fairseq). The implementation of Wave2Graph is added to the framework at [Wave2Graph](fairseq/models/wav2vec/wave2graph.py).


## Requirements and Installation
Please follow the instructions to [install the framework](https://github.com/facebookresearch/fairseq#getting-started).

Additionally, install [librosa]():
```
pip install librosa soundfile
```

## Training

### Preparing data

Create corresponding metadata (including: sample spectrum file path, number of frequency bands, number of time steps, label, profile id (optional)) for your dataset.

Place the meta information inside *data* directory.

### Fine-tuning

Example script for training Wave2Graph with transformer-based backbone on IJSound_{breathe_mouth}:
```
fairseq-hydra-train task.data=data/notest_IJSound_breathe_mouth_seed2022_fold0 model.w2v_path=${pretrained_model_checkpoint} common.seed=2022 optimization.max_epoch=200 criterion.class_weights=[1.0,1.0] model.graph_network=GAT +model.graph_input_dim=1024 +model.graph_hidden_dim=1024 +model.graph_n_hidden_dim=0 model.decoder_embed_dim=256 +model.graph_output_dim=256 +model.gat_n_head=1 +model.adj_threshold=0.8 --config-dir examples/wav2vec/config/finetuning --config-name ijsound_wave2graph
```

## Evaluation

```
fairseq-validate --path {trained_model_checkpoint} --task audio_finetuning data/notest_IJSound_breathe_mouth_seed2022_fold0 --valid-subset valid --batch-size 16
```

## Data availability

Our private dataset, IJSound, is available upon request for research purposes. Please send your information, including details about research usage and affiliations to hn@cs.ucc.ie.
