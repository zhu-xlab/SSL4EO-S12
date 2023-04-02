python main_oscd.py \
--data_dir /p/project/hai_ssl4eo/wang_yi/data/oscd \
--resnet_type 50 \
--init_type random \
--n_channels 13 \
--n_epochs 100 \
--learning_rate 0.001 \
--value_discard False \
--patch_size 96 \
--batch_size 8 \
--m_threshold 0.5 \
--result_dir results \
#--ckp_path /p/project/hai_ssl4eo/wang_yi/SSL4EO-S12/src/benchmark/pretrain_ssl/checkpoints/moco/SeCo_B12_rn50_224/checkpoint_0099.pth.tar
#--ckp_path /p/project/hai_ssl4eo/wang_yi/SSL4EO-S12/src/benchmark/pretrain_ssl/checkpoints/moco/SEN12MS_B13_rn50_224/checkpoint_0099.pth.tar
#--ckp_path /p/project/hai_ssl4eo/wang_yi/ssl4eo-s12-dataset/src/benchmark/fullset_temp/checkpoints/pretrained-weights/B13_rn50_moco_0099.pth.tar


