# data2vec

data2vec is a framework for self-supervised representation learning for images, speech, and text as described in data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language (Baevski et al., 2022). The algorithm uses the same learning mechanism for different modalities. You can read more about this work here [arxiv](https://arxiv.org/abs/2202.03555) and [fairseq repo](https://github.com/pytorch/fairseq/tree/main/examples/data2vec)

For details about how to setup your BEIT environment, please refer the original README [here](README_Original.md). Below you can find the necessary commands to reproduce the vision results reported in [data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language
](https://arxiv.org/abs/2202.03555)

## Model Checkpoints

Pretrained Model | Version | Link
|---|---|---|
data2vec ViT-B | 800 epochs pretrained | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec/data2vec_vision/base_800/checkpoint-799.pth)
data2vec ViT-L | 800 epochs pretrained | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec/data2vec_vision/large_800/checkpoint-799.pth)
data2vec ViT-L | 1600 epochs pretrained | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec/data2vec_vision/large_1600/checkpoint-799.pth)
data2vec ViT-B | Finetuned | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec/data2vec_vision/finetuned_base/checkpoint-99/mp_rank_00_model_states.pt)
data2vec ViT-L | Finetuned | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec/data2vec_vision/finetuned_large/checkpoint-49/mp_rank_00_model_states.pt)

## VIT-B Pretraining and Finetuning

Command to pretrain the ViT-B model for 800 epochs
```

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=16 run_cyclical.py \
        --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} --log_dir ${OUTPUT_DIR} --num_mask_patches 120 \
        --model beit_base_patch16_224 \
        --seed 0 \
        --target_layers [6,7,8,9,10,11] \
        --ema_decay 0.9998 --ema_start_at 0 --ema_decay_init 0.999 \
        --batch_size 128 --lr 2e-3 --warmup_epochs 10 --epochs 800 \
        --clip_grad 3.0 --drop_path 0.25 --layer_scale_init_value 1e-4 \
        --layer_results 'end' \
        --var_w0 0.0 --var_w1 0.0 \
        --max_mask_patches_per_block 196 --min_mask_patches_per_block 16 \
        --l1_beta=2.0 \
        --weight_decay 0.05 \
        --imagenet_default_mean_and_std --dist_url $dist_url --loss_scale -1 --mask_dropout_prob -1.0 \
        --post_target_layer_norm --world_size 16 --attn_drop_rate 0.05 


```

Command to finetune the ViT-B model
```

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
        --model beit_base_patch16_224 \
        --finetune $CHECKPOINT \
        --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} --log_dir ${OUTPUT_DIR} --batch_size 128 --lr 4e-3 --update_freq 1 \
        --warmup_epochs 10 --epochs 100 --layer_decay 0.65 --drop_path 0.2 --drop 0.0 \
        --weight_decay 0.0 --mixup 0.8 --cutmix 1.0 --enable_deepspeed --nb_classes 1000 \
        --target_layer -1 --world_size 8 --dist_url $dist_url 

```

## VIT-L Pretraining and Finetuning

Command to pretrain the ViT-L model for 800 epochs
```

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=64 run_cyclical.py \
        --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} --log_dir ${OUTPUT_DIR} --num_mask_patches 120 \
        --model beit_large_patch16_224 \
        --seed 0 \
        --target_layers [18,19,20,21,22,23] \
        --ema_decay 0.9998 --ema_start_at 0 \
        --batch_size 64 --lr 1e-3 --warmup_epochs 80 --epochs 800 \
        --clip_grad 3.0 --drop_path 0.2 --layer_scale_init_value 1e-5 \
        --layer_results 'end' \
        --l1_beta=2 \
	--var_w0 0.0 --var_w1 0.0 --var_margin0 0.5 \
	--max_mask_patches_per_block 196 --min_mask_patches_per_block 16 \
        --imagenet_default_mean_and_std --dist_url $dist_url --world_size 64 \
         --post_target_layer_norm  --attn_drop_rate 0.15


```

You further pretrain the ViT-L model for another 800 epochs with constant ema decay
```

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=64 run_cyclical.py \
        --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} --log_dir ${OUTPUT_DIR} --num_mask_patches 120 \
        --model beit_large_patch16_224 \
        --seed 0 \
        --target_layers [18,19,20,21,22,23] \
        --ema_decay 0.9999 --ema_start_at 0 --ema_decay_init 0.999 \
        --batch_size 64 --lr 1e-3 --warmup_epochs 40 --epochs 800 \
        --clip_grad 3.0 --drop_path 0.2 --layer_scale_init_value 1e-5 \
        --layer_results 'end' \
        --l1_beta=2 \
	--var_w0 0.0 --var_w1 0.0 --var_margin0 0.5 \
	--max_mask_patches_per_block 196 --min_mask_patches_per_block 16 \
        --imagenet_default_mean_and_std --dist_url $dist_url --world_size 64 \
         --post_target_layer_norm  --attn_drop_rate 0.15 \
         --seed_model {PATH_TO_800EPOCH_MODEL}

```

Command to finetune the ViT-L model
```

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=16 run_cyclical.py \
        --model beit_large_patch16_224 \
        --finetune $CHECKPOINT \
        --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} --log_dir ${OUTPUT_DIR} --batch_size 64 --lr 5e-3 --update_freq 1 \
        --warmup_epochs $WARMUP --epochs 50 --layer_decay 0.65 --drop_path 0.25 --drop 0.0 \
        --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 --enable_deepspeed --nb_classes 1000 --seed 0\
        --target_layer -1 --world_size 16 --dist_url $dist_url --attn_drop_rate 0.0


```


## LICENSE
Data2Vec is licensed under CC-BY-NC, however portions of the project are available under separate license terms: Unilm is licensed under the MIT license.

## CITATION
If you find this repository useful, please consider citing our work:

```
@misc{https://doi.org/10.48550/arxiv.2202.03555,
  doi = {10.48550/ARXIV.2202.03555},
  url = {https://arxiv.org/abs/2202.03555},
  author = {Baevski, Alexei and Hsu, Wei-Ning and Xu, Qiantong and Babu, Arun and Gu, Jiatao and Auli, Michael},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
