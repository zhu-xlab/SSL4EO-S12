python validate_oscd.py \
--data_dir /p/project/hai_ssl4eo/wang_yi/data/oscd \
--resnet_type 50 \
--init_type random \
--n_channels 12 \
--value_discard False \
--patch_size 96 \
--batch_size 274 \
--m_threshold 0.5 \
--result_dir results \
--ckp_resume /p/project/hai_ssl4eo/wang_yi/SSL4EO-S12/src/benchmark/transfer_change_detection/results/ckps/seco/99-0.15-0.27.ckpt \
#--ckp_resume /p/project/hai_ssl4eo/wang_yi/SSL4EO-S12/src/benchmark/transfer_change_detection/results/ckps/sen12ms/99-0.15-0.25.ckpt
#--ckp_resume /p/project/hai_ssl4eo/wang_yi/SSL4EO-S12/src/benchmark/transfer_change_detection/results/ckps/ssl4eo/99-0.15-0.29.ckpt
#--ckp_resume /p/project/hai_ssl4eo/wang_yi/SSL4EO-S12/src/benchmark/transfer_change_detection/results/ckps/random/99-0.15-0.19.ckpt \
