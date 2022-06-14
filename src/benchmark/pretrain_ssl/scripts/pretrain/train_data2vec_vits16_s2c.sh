OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 pretrain_data2vec.py \
        --data_path '/p/scratch/hai_ssl4eo/data/ssl4eo_s12/ssl4eo_250k_s2c_uint8.lmdb'\
        --output_dir '/p/project/hai_ssl4eo/nassim/data2vec/experiments/data2vec/pretrain/output' --log_dir '/p/project/hai_ssl4eo/nassim/data2vec/experiments/data2vec/pretrain/logs' \
        --num_mask_patches 120 \
        --aug_level 2 \
        --model beit_small_patch16_224 \
        --seed 0 \
        --target_layers [6,7,8,9,10,11] \
        --ema_decay 0.9998 --ema_start_at 0 --ema_decay_init 0.999 \
        --batch_size 64 --lr 1e-3 --warmup_epochs 1 --epochs 2 \
        --clip_grad 3.0 --drop_path 0.25 --layer_scale_init_value 1e-4 \
        --layer_results 'end' \
        --var_w0 0.0 --var_w1 0.0 \
        --max_mask_patches_per_block 196 --min_mask_patches_per_block 16 \
        --l1_beta=2.0 \
        --weight_decay 0.05 \
        --imagenet_default_mean_and_std --dist_url 'tcp://localhost:10001' --loss_scale -1 --mask_dropout_prob -1.0 \
        --post_target_layer_norm --world_size 1 --attn_drop_rate 0.05 
        
        