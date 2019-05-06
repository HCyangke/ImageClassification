#/bin/sh
cd /nfs/private/workspace/fire_smoke_detection
nvidia-smi
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/luban/anaconda3/envs/torch10/bin/python main.py --result_path lr0.01_ft4_r18_finetune_all --dataset custom --n_classes 3 --ft_begin_index 0 --model resnet --model_depth 18 --batch_size 512 --n_threads 4 --checkpoint 1
