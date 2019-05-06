#coding=utf-8
import csv
import random

split_spatio=0.8
fire_data=[]
smoke_data=[]
normal_data=[]

with open('list.csv') as f:
    # for line in f:
    #     n_frame=int(line.split(',')[-1])
    #     if n_frame>1000:
    #         print(line)
    lines=csv.reader(f)
    for line in lines:
        label=line[0]
        images_name=line[1]
        path=line[2]
        if label=='fire':
            fire_data.append([label,images_name,path])
        if label=='smoke':
            smoke_data.append([label,images_name,path])
        if label=='normal':
            normal_data.append([label,images_name,path])
print('fire_data',fire_data)
print('smoke_data',smoke_data)
print('normal_data',normal_data)

val_data=[]
train_data=[]

random.shuffle(fire_data)
random.shuffle(smoke_data)
random.shuffle(normal_data)

train_val=int(split_spatio*len(fire_data))
train_data.extend(fire_data[:train_val])
val_data.extend(fire_data[train_val:])

train_val=int(split_spatio*len(smoke_data))
train_data.extend(smoke_data[:train_val])
val_data.extend(smoke_data[train_val:])

train_val=int(split_spatio*len(normal_data))
train_data.extend(normal_data[:train_val])
val_data.extend(normal_data[train_val:])

with open('train.csv','w') as f:
    writer = csv.writer(f)
    writer.writerows(train_data)
with open('val.csv','w') as f:
    writer = csv.writer(f)
    writer.writerows(val_data)
