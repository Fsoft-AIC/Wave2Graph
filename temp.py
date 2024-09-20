import os

for i, val in enumerate(['2023-01-05/15-33-02', '2023-01-05/17-41-05', '2023-01-05/19-59-17']):
    os.system(f'CUDA_VISIBLE_DEVICES="3" fairseq-validate --path /home/tungtk2/fairseq/outputs/{val}/checkpoints/checkpoint_best.pt --task audio_finetuning /home/tungtk2/fairseq/data/vocalsound --valid-subset te_female --batch-size 64')