```bibtex
@inproceedings{hara3dcnns,
  author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh},
  title={Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={6546--6555},
  year={2018},
}
```

clone from https://github.com/kenshohara/3D-ResNets-PyTorch.git

## Pre-trained models

Pre-trained models are available [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing).  
All models are trained on Kinetics.  
ResNeXt-101 achieved the best performance in our experiments. (See [paper](https://arxiv.org/abs/1711.09577) in details.)

```misc
resnet-18-kinetics.pth: --model resnet --model_depth 18 --resnet_shortcut A
resnet-34-kinetics.pth: --model resnet --model_depth 34 --resnet_shortcut A
resnet-34-kinetics-cpu.pth: CPU ver. of resnet-34-kinetics.pth
resnet-50-kinetics.pth: --model resnet --model_depth 50 --resnet_shortcut B
resnet-101-kinetics.pth: --model resnet --model_depth 101 --resnet_shortcut B
resnet-152-kinetics.pth: --model resnet --model_depth 152 --resnet_shortcut B
resnet-200-kinetics.pth: --model resnet --model_depth 200 --resnet_shortcut B
preresnet-200-kinetics.pth: --model preresnet --model_depth 200 --resnet_shortcut B
wideresnet-50-kinetics.pth: --model wideresnet --model_depth 50 --resnet_shortcut B --wide_resnet_k 2
resnext-101-kinetics.pth: --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32
densenet-121-kinetics.pth: --model densenet --model_depth 121
densenet-201-kinetics.pth: --model densenet --model_depth 201
```

Some of fine-tuned models on UCF-101 and HMDB-51 (split 1) are also available.

```misc
resnext-101-kinetics-ucf101_split1.pth: --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32
resnext-101-64f-kinetics-ucf101_split1.pth: --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration 64
resnext-101-kinetics-hmdb51_split1.pth: --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32
resnext-101-64f-kinetics-hmdb51_split1.pth: --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration 64
```
Train ResNets-34 on the Kinetics dataset (400 classes) with 4 CPU threads (for data loading).  
Batch size is 128.  
Save models at every 5 epochs.
All GPUs is used for the training.
If you want a part of GPUs, use ```CUDA_VISIBLE_DEVICES=...```.

```bash
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --model resnet \
--model_depth 34 --n_classes 400 --batch_size 128 --n_threads 4 --checkpoint 5
```

Continue Training from epoch 101. (~/data/results/save_100.pth is loaded.)

```bash
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_100.pth \
--model_depth 34 --n_classes 400 --batch_size 128 --n_threads 4 --checkpoint 5
```

Fine-tuning conv5_x and fc layers of a pretrained model (~/data/models/resnet-34-kinetics.pth) on UCF-101.

```bash
python main.py --root_path ~/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --n_classes 400 --n_finetune_classes 101 \
--pretrain_path models/resnet-34-kinetics.pth --ft_begin_index 4 \
--model resnet --model_depth 34 --resnet_shortcut A --batch_size 128 --n_threads 4 --checkpoint 5
```
