from datasets.custom import custom
def get_training_set(opt, spatial_transform,
                     target_transform):
    assert opt.dataset in ['custom', 'all']

    if opt.dataset == 'custom':
        training_data = custom(
            'train',
            spatial_transform=spatial_transform,
            target_transform=target_transform)
    elif opt.dataset == 'all':
        training_data = custom(
            'list',
            spatial_transform=spatial_transform,
            target_transform=target_transform)

    return training_data


def get_validation_set(opt, spatial_transform,
                       target_transform):
    assert opt.dataset in ['custom', 'all', 'test']

    if opt.dataset == 'custom':
        validation_data = custom(
            'val',
            spatial_transform,
            target_transform)
    if opt.dataset == 'all':
        validation_data = custom(
            'list',
            spatial_transform,
            target_transform)
    if opt.dataset == 'test':
        validation_data = custom(
            'test_list',
            spatial_transform,
            target_transform)

    return validation_data

def get_test_set(opt, spatial_transform, target_transform):
    assert opt.dataset in ['custom']
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'val'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'custom':
        test_data = custom(
            subset,
            spatial_transform,
            target_transform)
    return test_data

