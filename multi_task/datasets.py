import torch
from torchvision import transforms
from loaders.multi_mnist_loader import MNIST
from loaders.cityscapes_loader import CITYSCAPES
from loaders.segmentation_augmentations import *
from loaders.celeba_loader import CELEBA

# Setup Augmentations
cityscapes_augmentations= Compose([RandomRotate(10),
                                   RandomHorizontallyFlip()])

def global_transformer():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])


def get_dataset(params, configs):
    if 'dataset' not in params:
        print('ERROR: No dataset is specified')

    if 'mnist' in params['dataset']:
        train_dst = MNIST(root=configs['mnist']['path'], train=True, download=True, transform=global_transformer(), multi=True)
        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=4)

        val_dst = MNIST(root=configs['mnist']['path'], train=False, download=True, transform=global_transformer(), multi=True)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=100, shuffle=True, num_workers=4)
        return train_loader, train_dst, val_loader, val_dst

    if 'cityscapes' in params['dataset']:
        train_dst = CITYSCAPES(root=configs['cityscapes']['path'], is_transform=True, split=['train'], img_size=(configs['cityscapes']['img_rows'], configs['cityscapes']['img_cols']), augmentations=cityscapes_augmentations)
        val_dst = CITYSCAPES(root=configs['cityscapes']['path'], is_transform=True, split=['val'], img_size=(configs['cityscapes']['img_rows'], configs['cityscapes']['img_cols']))

        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'], num_workers=4)
        return train_loader, train_dst, val_loader, val_dst

    if 'celeba' in params['dataset']:
        train_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='train', img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']), augmentations=None)
        val_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='val', img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']), augmentations=None)

        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'], num_workers=4)
        return train_loader, train_dst, val_loader, val_dst

    # if 'celeba' in params['dataset']:
    #     train_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='train',
    #                        img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']),
    #                        augmentations=None)
    #     # train2_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='train2',img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']), augmentations=None)
    #     val1_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='val',
    #                       img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']),
    #                       augmentations=None)
    #
    #     train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True,
    #                                                num_workers=4)
    #     # train2_loader = torch.utils.data.DataLoader(train2_dst, batch_size=params['batch_size'], shuffle=True,num_workers=4)
    #     val_loader = torch.utils.data.DataLoader(val1_dst, batch_size=params['batch_size'], num_workers=4,
    #                                              shuffle=False)
    #     # val2_loader = torch.utils.data.DataLoader(val2_dst, batch_size=params['batch_size'], num_workers=4,
    #     #                                           shuffle=True)
    #     return train_loader, val_loader, None  # train2_loader

# DANGER! TEST SET SHOULD NOT BE USED UNLESS IN EMERGENCY
def get_test_dataset(params, configs):
    if 'dataset' not in params:
        print('ERROR: No dataset is specified')

    if 'celeba' in params['dataset']:
        test_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='val',
                          img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']),
                          augmentations=None)

        test_loader = torch.utils.data.DataLoader(test_dst, batch_size=params['batch_size'], num_workers=4,
                                                  shuffle=False)

        return test_loader