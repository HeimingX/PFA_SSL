from models.resnet import resnet18, resnet34, resnet50, resnet152, resnet101, mocov2_ep200, mocov2_ep800

from data.tranforms import TransformTrain, TransformTrainWeakStrongStrong
from data.tranforms import TransformTest
import data
from data.cifar100 import get_cifar100
from torch.utils.data import DataLoader, RandomSampler
import torch
import os

imagenet_mean=(0.485, 0.456, 0.406)
imagenet_std=(0.229, 0.224, 0.225)

def load_data(args):
    batch_size_dict = {"train": args.batch_size, "unlabeled_train": args.batch_size, "test": args.batch_size}

    if 'cifar100' in args.root:
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, args.root)
        labeled_trainloader = DataLoader(
            labeled_dataset,
            sampler=RandomSampler(labeled_dataset),
            batch_size=batch_size_dict["train"],
            num_workers=args.workers,
            drop_last=True)

        unlabeled_trainloader = DataLoader(
            unlabeled_dataset,
            sampler=RandomSampler(unlabeled_dataset),
            batch_size=batch_size_dict["unlabeled_train"],
            num_workers=args.workers,
            drop_last=True)

        ## We didn't apply tencrop test since other SSL baselines neither
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size_dict["test"],
            shuffle=False,
            num_workers=args.workers)

        dataset_loaders = {"train": labeled_trainloader,
                           "unlabeled_train": unlabeled_trainloader,
                           "test": test_loader}

    else:
        dataset = data.__dict__[os.path.basename(args.root)]
        transform_test = TransformTest(mean=imagenet_mean, std=imagenet_std)
        transform_train = TransformTrain()
        train_aug_ul = TransformTrainWeakStrongStrong()

        if args.label_ratio == 100:
            datasets = {"train": dataset(root=args.root, split='train', label_ratio=args.label_ratio, download=True, transform=transform_train),
                }
        else:
            if hasattr(args, 'output_index') and args.output_index:
                output_index = True
            else:
                output_index = False
            datasets = {"train": dataset(root=args.root, split='train', label_ratio=args.label_ratio, download=True, transform=transform_train),
                    "unlabeled_train": dataset(root=args.root, split='unlabeled_train', label_ratio=args.label_ratio, download=True, transform=train_aug_ul, output_index=output_index)}
        test_dataset = {
            'test' + str(i): dataset(root=args.root, split='test', label_ratio=100, download=True, transform=transform_test["test" + str(i)]) for i in range(10)
        }
        datasets.update(test_dataset)

        if args.label_ratio == 100:
            dataset_loaders = {x: DataLoader(datasets[x], batch_size=batch_size_dict[x], shuffle=True, num_workers=args.workers)
                           for x in ['train']}
        else:
            dataset_loaders = {x: DataLoader(datasets[x], batch_size=batch_size_dict[x], shuffle=True, num_workers=args.workers)
                            for x in ['train', 'unlabeled_train']}
        dataset_loaders.update({'test' + str(i): DataLoader(datasets["test" + str(i)], batch_size=4, shuffle=False, num_workers=args.workers)
                                for i in range(10)})

    return dataset_loaders


def load_network(backbone):
    if 'resnet' in backbone:
        if backbone == 'resnet18':
            network = resnet18
            feature_dim = 512
        elif backbone == 'resnet34':
            network = resnet34
            feature_dim = 512
        elif backbone == 'resnet50':
            network = resnet50
            feature_dim = 2048
        elif backbone == 'resnet101':
            network = resnet101
            feature_dim = 2048
        elif backbone == 'resnet152':
            network = resnet152
            feature_dim = 2048
    elif 'efficientnet' in backbone:
        from models.efficientnet import EfficientNetFc
        network = EfficientNetFc
        print(backbone)
        if backbone == 'efficientnet-b0':
            feature_dim = 1280
        elif backbone == 'efficientnet-b1':
            feature_dim = 1280
        elif backbone == 'efficientnet-b2':
            feature_dim = 1408
        elif backbone == 'efficientnet-b3':
            feature_dim = 1536
        elif backbone == 'efficientnet-b4':
            feature_dim = 1792
        elif backbone == 'efficientnet-b5':
            feature_dim = 2048
        elif backbone == 'efficientnet-b6':
            feature_dim = 2304
    else:
        network = resnet50
        feature_dim = 2048

    return network, feature_dim


def trackpl(satisfied_mask, ul_idx, predict_unlabeled, label_ul, track_pl_bank):
    for _ul_idx, _pl, _gt in zip(ul_idx[satisfied_mask], predict_unlabeled[satisfied_mask], label_ul[satisfied_mask]):
        if track_pl_bank[_ul_idx].sum() == 0:
            # 1st shot
            if _pl == _gt:
                track_pl_bank[_ul_idx][0] += 1
            else:
                track_pl_bank[_ul_idx][3] += 1
        else:
            # after 1st shot
            filled_loc = torch.nonzero(track_pl_bank[_ul_idx])[:, 0][0]
            if filled_loc < 2:
                if _pl == _gt:
                    track_pl_bank[_ul_idx][0] += 1
                else:
                    track_pl_bank[_ul_idx][1] += 1
            else:
                if _pl == _gt:
                    track_pl_bank[_ul_idx][2] += 1
                else:
                    track_pl_bank[_ul_idx][3] += 1
    tracked_pl_1st_true = track_pl_bank[:, :2].sum(1) > 0
    tracked_pl_1st_false = track_pl_bank[:, 2:].sum(1) > 0
    if tracked_pl_1st_true.sum() > 0:
        acc_pl_1st_true = (track_pl_bank[tracked_pl_1st_true, 0] / track_pl_bank[tracked_pl_1st_true, :2].sum(1)).mean()
        err_pl_1st_true = (track_pl_bank[tracked_pl_1st_true, 1] / track_pl_bank[tracked_pl_1st_true, :2].sum(1)).mean()
    else:
        acc_pl_1st_true = 0
        err_pl_1st_true = 0
    if tracked_pl_1st_false.sum() > 0:
        acc_pl_1st_false = (track_pl_bank[tracked_pl_1st_false, 2] / track_pl_bank[tracked_pl_1st_false, 2:].sum(1)).mean()
        err_pl_1st_false = (track_pl_bank[tracked_pl_1st_false, 3] / track_pl_bank[tracked_pl_1st_false, 2:].sum(1)).mean()
    else:
        acc_pl_1st_false = 0
        err_pl_1st_false = 0
    return track_pl_bank, (acc_pl_1st_true, err_pl_1st_true, acc_pl_1st_false, err_pl_1st_false)