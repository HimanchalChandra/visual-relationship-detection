from datasets.vrd import VrdDataset


def get_dataset(opt, type, transform):
    assert opt.dataset in ['vrd', 'visual_genome']

    if opt.dataset == 'vrd':
        dataset = VrdDataset(opt.dataset_path, opt.num_classes, type, transform)
    # elif opt.dataset == 'activitynet':
    #     training_data = ActivityNet(
    #         opt.video_path,
    #         opt.annotation_path,
    #         'training',
    #         False,
    #         spatial_transform=spatial_transform,
    #         temporal_transform=temporal_transform,
    #         target_transform=target_transform)
    return dataset

