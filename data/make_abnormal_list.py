#coding=utf-8
import glob
import csv
import os
import os.path
from subprocess import call
fire_data=glob.glob("/nfs/project/surveillance/smoke_fire_detection/clean/fire/*")
smoke_data=glob.glob("/nfs/project/surveillance/smoke_fire_detection/clean/smoke/*")
normal_data=glob.glob("/nfs/project/surveillance/smoke_fire_detection/clean/normal/*")
data_lsit=[]
'''
for data in graffiti_data:
    # print(drip_data)
    # data = ''.join(data)
    data=data.split('/')[-1]
    video_name=data.split('.')[0]
    #变成帧，加入到dataß
    dest_dir=os.path.join('../abnormal_action_images_data/graffiti',video_name)
    if not os.path.exists(dest_dir):
        call(['mkdir',dest_dir])
    dest=os.path.join(dest_dir,'%05d.jpg')
    src = 'graffiti/'+data
    call(["ffmpeg", "-i", src, dest])
    n_frames=len(glob.glob(dest_dir+'/*'))
    data_lsit.append(['graffiti',data,n_frames])
'''
for data in fire_data:
    path=data
    data=data.split('/')[-1]
    image_name=data.split('.')[0]
    data_lsit.append(['fire',image_name,path])

for data in smoke_data:
    path=data
    data=data.split('/')[-1]
    image_name=data.split('.')[0]
    data_lsit.append(['smoke',image_name,path])

for data in normal_data:
    path=data
    data=data.split('/')[-1]
    image_name=data.split('.')[0]
    data_lsit.append(['normal',image_name,path])

with open('list.csv', 'w') as fout:
    writer = csv.writer(fout)
    writer.writerows(data_lsit)
# print(data_lsit)

