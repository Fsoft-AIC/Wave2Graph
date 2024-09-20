#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from json import encoder
import logging
import os
import sys
from argparse import Namespace
from itertools import chain

import torch
from omegaconf import DictConfig

from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import metrics, progress_bar
from fairseq.utils import reset_logging
import numpy as np

import wandb

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")


run = None
def get_run(run_name):
    api = wandb.Api()

    # runs = api.runs(f'snp-robustness/RoPADet_vocalsound_multibranch')
    # runs = api.runs(f'snp-robustness/RoPADet_vocalsound_npn_option2')
    # runs = api.runs(f'snp-robustness/RoPADet_vocalsound_npn_only')
    # runs = api.runs(f'snp-robustness/RoPADet_vocalsound_cnn')
    # runs = api.runs(f'snp-robustness/RoPADet_urbansound8k_cnn')
    # runs = api.runs(f'snp-robustness/RoPADet_icbhi_npn_option2')
    # runs = api.runs(f'snp-robustness/wav2vec2_icbhi')
    # runs = api.runs(f'snp-robustness/RoPADet_urbansound8k_npn_option2')
    # runs = api.runs(f'snp-robustness/wave2graph_ICBHI')
    # runs = api.runs(f'snp-robustness/wave2graph_adresso')
    runs = api.runs(f'tktung/wave2graph_IJSound')
    # runs = api.runs(f'tktung/RoPADet_IJSound')
    # runs = api.runs(f'snp-robustness/graphonly_IJSound')
    for curr_run in runs:
        if curr_run.name == run_name:
            run = curr_run
            break

    return run


def update_run(run, k, v):
    if (isinstance(run.summary, wandb.old.summary.Summary) and k not in run.summary):
        run.summary._root_set(run.summary._path, [(k, {})])
    run.summary[k] = v


def main(cfg: DictConfig, override_args=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    reset_logging()

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    if cfg.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(saved_cfg)

    # Build criterion
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()


    # np.save(dAVED WEIGHT")
    # return

    for subset in cfg.dataset.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1, task_cfg=saved_cfg.task)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            # required_batch_size_multiple=1,
            seed=cfg.common.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        )

        log_outputs = []
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            # print("BATCH SIZE: ", sample['net_input']['source'].shape)
            sample['net_input']['dataset'] = task.datasets[subset]
            _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
            progress.log(log_output, step=i)
            log_outputs.append(log_output)

        if data_parallel_world_size > 1:
            log_outputs = distributed_utils.all_gather_list(
                log_outputs,
                max_size=cfg.common.all_gather_list_size,
                group=distributed_utils.get_data_parallel_group(),
            )
            log_outputs = list(chain.from_iterable(log_outputs))

        targets = [log.get("targets") for log in log_outputs]
        # # print(targets)
        flat_targets = [item for target in targets for item in target]
        # encoder_out = [log.get("encoder_out") for log in log_outputs]
        # # profiles_tensor = [log.get("profiles_tensor") for log in log_outputs]
        # # # profiles_before_attn_pool_tensor = [log.get('profiles_before_attn_pool_tensor') for log in log_outputs]
        # graph_out = [log.get("graph_out") for log in log_outputs]
        # # fused_tensor  = [log.get('fused_tensor') for log in log_outputs]

        # import numpy as np
        # print(np.concatenate(encoder_out).shape)
        # # print(np.concatenate(profiles_tensor).shape)
        
        # np.save(f'encoder_out_{subset}.npy', np.concatenate(encoder_out))
        # # np.save(f'profiles_tensor_{subset}.npy', np.concatenate(profiles_tensor))
        # # # np.save(f'profiles_before_attn_pool_tensor_{subset}.npy', np.concatenate(profiles_before_attn_pool_tensor))
        # np.save(f'graph_out_{subset}.npy', np.concatenate(graph_out))
        # # np.save(f'fused_tensor_{subset}.npy', np.concatenate(fused_tensor))

        # np.save(f'labels_{subset}.npy', np.array(flat_targets))
        # profiles = [log.get("profile") for log in log_outputs]
        # flat_profiles = [item for profile in profiles for item in profile]
        # np.save(f'profiles_{subset}.npy', np.array(flat_profiles))
        predicts = [log.get("predicts") for log in log_outputs]
        # print(predicts)
        flat_predicts = []
        for predict in predicts:
            if type(predict) != list:
                flat_predicts.append(predict)
            else:
                flat_predicts.extend(predict)

        with metrics.aggregate() as agg:
            task.reduce_metrics(log_outputs, criterion)
            log_output = agg.get_smoothed_values()

        progress.print(log_output, tag=subset, step=i)

        from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_fscore_support, confusion_matrix, accuracy_score
        update_run(run, f'eval', dict())

        if len(cfg.criterion.class_weights) != 2:
            auc = roc_auc_score(flat_targets, flat_predicts, multi_class='ovr')
        elif cfg.criterion.ovr_onehot:
            brier_score = brier_score_loss(flat_targets, final_predicts)
            print(f'Brier Score : {brier_score:12.4f}')
            auc = roc_auc_score(flat_targets, flat_predicts, average='micro')
            run.summary[f'eval']['brier_score'] = round(brier_score, 4)
        else:
            brier_score = brier_score_loss(flat_targets, final_predicts)
            print(f'Brier Score : {brier_score:12.4f}')
            auc = roc_auc_score(flat_targets, flat_predicts)
            run.summary[f'eval']['brier_score'] = round(brier_score, 4)
        print(f"AUC score: {auc:12.4f}")
        run.summary[f'eval']['auc'] = round(auc, 4)
        with open('results.txt', 'a') as f:
            f.write(f"{auc}\n")

        if cfg.checkpoint.best_checkpoint_metric == 'icbhi':
            def get_score(hits, counts):
                se = (hits[1] + hits[2] + hits[3]) / (counts[1] + counts[2] + counts[3])
                sp = hits[0] / counts[0]
                print("SENSE: ", se)
                print("SPEC: ", sp)
                run.summary[f'eval']['SENSE'] = round(se, 4)
                run.summary[f'eval']['SPEC'] = round(sp, 4)
                sc = (se+sp) / 2.0
                return sc

            final_predicts_onehot = flat_predicts
            final_predicts = [predict.index(max(predict)) for predict in flat_predicts]

            class_hits = [0.0, 0.0, 0.0, 0.0] # normal, crackle, wheeze, both
            class_counts = [0.0, 0.0, 0.0+1e-7, 0.0+1e-7] # normal, crackle, wheeze, both
            for idx in range(len(flat_targets)):
                class_counts[flat_targets[idx]] += 1.0
                if final_predicts[idx] == flat_targets[idx]:
                    class_hits[flat_targets[idx]] += 1.0
            
            from sklearn.metrics import confusion_matrix
            print(class_counts)
            print(confusion_matrix(flat_targets, final_predicts))

            icbhi_score = get_score(class_hits, class_counts)
            auc_ovr = roc_auc_score(flat_targets, final_predicts_onehot, multi_class='ovr')
            auc_ovo = roc_auc_score(flat_targets, final_predicts_onehot, multi_class='ovo')
            print(f"ICBHI score: {icbhi_score:12.4f}")
            print(f"AUC OVR: {auc_ovr:12.4f}")
            print(f"AUC OVO: {auc_ovo:12.4f}")
            run.summary[f'eval']['ICBHI'] = round(icbhi_score, 4)
            run.summary[f'eval']['AUC_OVR'] = round(auc_ovr, 4)
            run.summary[f'eval']['AUC_OV)'] = round(auc_ovo, 4)

        # import pdb
        # pdb.set_trace()
        try:
            final_predicts = [predict.index(max(predict)) for predict in flat_predicts]
        except Exception as _:
            flat_predicts = [[1-i, i] for i in flat_predicts]
            final_predicts = [predict.index(max(predict)) for predict in flat_predicts]
        confidence_scores = [predict[target] for predict, target in zip (flat_predicts, flat_targets)]
        run.summary[f'eval']['confidence scores'] = list(confidence_scores)

        cm1 = confusion_matrix(flat_targets, final_predicts)
        print('\nConfusion Matrix : \n', cm1)
        run.summary[f'eval']['cm'] = cm1
        precision, recall, f1, _ = precision_recall_fscore_support(flat_targets, final_predicts, average='macro')
        acc = accuracy_score(flat_targets, final_predicts)
        print(f'Precision  : {precision:12.4f}')
        
        print(f'Recall      : {recall:12.4f}')
        
        print(f'F1 Score    : {f1:12.4f}')

        print(f'Accuracy    : {acc:12.4f}')

        run.summary[f'eval']['precision'] = round(precision, 4)
        run.summary[f'eval']['recall'] = round(recall, 4)
        run.summary[f'eval']['f1_score'] = round(f1, 4)
        run.summary[f'eval']['accuracy'] = round(acc, 4)

        run.summary.update()
        # import pdb; pdb.set_trace()
        mc_nemar = [1 if pred==label else 0 for pred, label in zip(final_predicts, flat_targets)]
        # np.save(run_name.replace('/','-') + '.npy', mc_nemar)

        # for i in range(len(flat_targets)):
        #     if flat_targets[i] != final_predicts[i] and flat_targets[i] == 7:
        #         print('Index: ', i)
        #         print('Profile ID: ', flat_profiles[i])
        #         print("Predicted class: ", final_predicts[i])


def cli_main():
    global run
    global run_name

    parser = options.get_validation_parser()
    args = options.parse_args_and_arch(parser)

    path = args.path
    start_id = path.find('202')
    end_id = path.find('/checkpoints/checkpoint')
    run_name = path[start_id:end_id]

    run = get_run(run_name)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(
        convert_namespace_to_omegaconf(args), main, override_args=override_args
    )


if __name__ == "__main__":
    cli_main()
