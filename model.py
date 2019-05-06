import torch
from torch import nn

from models import resnet, se_resnet


def generate_model(opt):
    assert opt.model in [
        'resnet', 'densenet', 'se_resnet'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        from models.resnet import get_fine_tuning_parameters

        if opt.model_depth == 10:
            model = resnet.resnet10(pretrained=True, num_classes=opt.n_classes)
        elif opt.model_depth == 18:
            model = resnet.resnet18(pretrained=True, num_classes=opt.n_classes)
        elif opt.model_depth == 34:
            model = resnet.resnet34(pretrained=True, num_classes=opt.n_classes)
        elif opt.model_depth == 50:
            model = resnet.resnet50(pretrained=True, num_classes=opt.n_classes)
        elif opt.model_depth == 101:
            model = resnet.resnet101(pretrained=True, num_classes=opt.n_classes)
        elif opt.model_depth == 152:
            model = resnet.resnet152(pretrained=True, num_classes=opt.n_classes)
        elif opt.model_depth == 200:
            model = resnet.resnet200(pretrained=True, num_classes=opt.n_classes)
    elif opt.model == 'se_resnet':
        assert opt.model_depth in [18, 34, 50, 101, 152]

        from models.se_resnet import get_fine_tuning_parameters

        if opt.model_depth == 18:
            model = se_resnet.resnet18(pretrained=True, num_classes=opt.n_classes)
        elif opt.model_depth == 34:
            model = se_resnet.resnet34(pretrained=True, num_classes=opt.n_classes)
        elif opt.model_depth == 50:
            model = se_resnet.resnet50(pretrained=True, num_classes=opt.n_classes)
        elif opt.model_depth == 101:
            model = se_resnet.resnet101(pretrained=True, num_classes=opt.n_classes)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
        return model, parameters
    else:
        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
        return model, parameters

    return model, model.parameters()

