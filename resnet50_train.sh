#/bin/sh
cd /nfs/private/workspace/3DResNet/
nvidia-smi
CUDA_VISIBLE_DEVICES=0 /home/luban/anaconda3/envs/torch10/bin/python main.py --root_path ~/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json --result_path resnet50_ucf101 --dataset ucf101 --n_classes 400 --n_finetune_classes 101 --pretrain_path pretrained/resnet-50-kinetics.pth --ft_begin_index 4 --model resnet --model_depth 50 --resnet_shortcut B --batch_size 32 --n_threads 4 --checkpoint 1 --n_val_sample 1
