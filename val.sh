#/bin/sh
cd /nfs/private/workspace/clean
nvidia-smi
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/luban/anaconda3/envs/torch10/bin/python for_recall.py --result_path test --dataset test --n_classes 2 --ft_begin_index 4 --model resnet --model_depth 18 --batch_size 128 --n_threads 4 --checkpoint 1 --no_train --resume_path ft4_dataset_clean_with_normalize_resnet_random_crop/save_159.pth
