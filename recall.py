import numpy as np
import csv
from subprocess import call

result_file="top3.npy"
annotation_file='data/test_list.csv'

results = np.load(result_file)

class Logger(object):

    def __init__(self, path, header='分类出错的视频'):
        self.log_file = open(path, 'w')
        #self.logger = csv.writer(self.log_file, delimiter='\t')
        #self.log_file.write(header+'\n')
        #self.logger.writerow(header)
        #self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        self.log_file.write(values+'\n')
        self.log_file.flush()

label2index={}
label2index['fire']=0
label2index['normal']=1
label2index['smoke']=2
label2index[0]='fire'
label2index[1]='normal'
label2index[2]='smoke'
image_list=[]
log=Logger('error.txt')
with open(annotation_file,'r') as myFile:  
    lines=csv.reader(myFile)  
    for line in lines:  
        label=line[0]#.split(';')
        name=line[1]
        # video_name=video_name.split('.')[0]
        path=line[2]
        image_list.append(line)

if results.shape[0] != len(image_list):
    print("numpy的大小和video sample的数量不一致")

count = 0
totle = 0
fire_totle = 0
fire_count = 0
presion_count = 0
presion_totle = 0
for i in range(len(image_list)):
    score = results[i]
    #print(score)
    index = 0
    #for j in range(score.size):
    #    if score[index] < score[j]:
    #        index = j
    if score <= 0.5:
        index = 1
    totle += 1
    if image_list[i][0]=='fire':
        fire_totle += 1
    #print(image_list[i][0])
    if index == 0:
        presion_totle += 1
    if index == label2index[image_list[i][0]]:
        #print('true')
        if image_list[i][0]=='fire':
            fire_count += 1
            presion_count += 1
        count += 1
    else:
        #print('false')
        log.log(image_list[i][0] + '+' + image_list[i][2])

print('data totle: ',totle)
print('data count: ',count)   
print('data acc: ',count/totle)

print('data fire_totle: ',fire_totle)
print('data fire_count: ',fire_count)   
print('recall acc: ',fire_count/fire_totle)

print('presion fire_totle: ',presion_totle)
print('precosion true fire_count: ',presion_count)
print('precision: ',presion_count/presion_totle)
#print(results[1].shape)
