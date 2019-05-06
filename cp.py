from subprocess import call
import os
with open('right.txt') as f:
    for line in f:
        label = line.strip().split('+')[0]
        #if label == 'normal':
        #    print('fire done')
        data = line.strip().split('+')[-1]
        #call(['rm','{}/*'.format(label)])
        #os.system('rm {}/*'.format(label))
        os.system('cp {} {}/'.format(data, label))
        #call(['cp',data,'{}/'.format(label)])
