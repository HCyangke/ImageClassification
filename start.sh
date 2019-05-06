#/bin/sh
cd /nfs/private/workspace/clean
nvidia-smi
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/luban/anaconda3/envs/torch10/bin/python focal_loss_adam.py --result_path ft4_dataset_clean_with_normalize_resnet_random_crop_fire_normal_clean_adam_focalloss --dataset all --n_classes 2 --ft_begin_index 4 --model resnet --model_depth 18 --batch_size 512 --n_threads 8 --checkpoint 1 --no_val

