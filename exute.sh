#/bin/sh
cd /nfs/private/workspace/fire_normal
nvidia-smi
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/luban/anaconda3/envs/torch10/bin/python main.py --result_path ft0_dataset_all_se_resnet_no_normalize --dataset all --n_classes 2 --ft_begin_index 0 --model se_resnet --model_depth 18 --batch_size 512 --n_threads 8 --checkpoint 1 --no_val

