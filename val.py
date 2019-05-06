import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time
import sys

from utils import AverageMeter, calculate_accuracy
import numpy as np

def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))
    # In PyTorch 0.4, "volatile=True" is deprecated.
    torch.set_grad_enabled(False)
    model.eval()

    batch_time = AverageMeter()
    forward_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    res = []
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs)
        b, ncrops, c, h, w = inputs.size()
        inputs = inputs.view(-1, c, h, w)
        targets = Variable(targets)
        tmp=time.time()
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=-1)
        outputs = outputs.view(b, ncrops, -1)
        outputs = outputs[:, :, 0].topk(3, dim=1)[0].mean(dim=1)
        #print(outputs)
        res.append(outputs.cpu().numpy())

        forward_time.update(time.time()-tmp)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time))

    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
    print("forward_time: ",forward_time.avg)

    res = np.concatenate(res,axis=0)
    np.save("tmp.npy",res)
    
    return losses.avg
