import sys
import os
proj_dir = os.path.abspath(os.getcwd())
print("proj_dir: ", proj_dir)
sys.path.append(proj_dir)

import argparse
import random
import matplotlib; matplotlib.use('agg')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.classifier import Classifier
from models.method import ProgressiveFeatureAdjustment
from tensorboardX import SummaryWriter
from src.utils import load_network, load_data


def test_cifar(loader, model, classifier, device):
    with torch.no_grad():
        start_test = True
        iter_val = iter(loader['test'])
        for i in range(len(loader['test'])):
            data = iter_val.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            labels = labels.to(device)
            feat = model.inference(inputs)
            outputs = classifier(feat)

            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)

        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy


def test(loader, model, classifier, device):
    with torch.no_grad():
        model.eval()
        classifier.eval()
        start_test = True
        val_len = len(loader['test0'])
        iter_val = [iter(loader['test' + str(i)]) for i in range(10)]
        for _ in range(val_len):
            data = [iter_val[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            for j in range(10):
                inputs[j] = inputs[j].to(device)
            labels = labels.to(device)
            outputs = []
            for j in range(10):
                feat = model.inference(inputs[j])
                output = classifier(feat)
                outputs.append(output)
            outputs = sum(outputs)
            if start_test:
                all_outputs = outputs.data.float()
                all_labels = labels.data.float()
                start_test = False
            else:
                all_outputs = torch.cat((all_outputs, outputs.data.float()), 0)
                all_labels = torch.cat((all_labels, labels.data.float()), 0)
        _, predict = torch.max(all_outputs, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_labels).item() / float(all_labels.size()[0])
    return accuracy


def train(args, model, classifier, dataset_loaders, optimizer, scheduler, device=None, writer=None, model_path = None):

    len_labeled = len(dataset_loaders["train"])
    iter_labeled = iter(dataset_loaders["train"])
    if args.label_ratio != 100:
        len_unlabeled = len(dataset_loaders["unlabeled_train"])
        iter_unlabeled = iter(dataset_loaders["unlabeled_train"])

    criterions = {"CrossEntropy": nn.CrossEntropyLoss(), "KLDiv": nn.KLDivLoss(reduction='batchmean')}

    best_acc = 0.0
    best_model = None

    for iter_num in range(1, args.max_iter + 1):
        model.train(True)
        classifier.train(True)
        optimizer.zero_grad()
        if iter_num % len_labeled == 0:
            iter_labeled = iter(dataset_loaders["train"])
        if args.label_ratio != 100 and iter_num % len_unlabeled == 0:
            iter_unlabeled = iter(dataset_loaders["unlabeled_train"])

        data_labeled = iter_labeled.next()
        if args.label_ratio != 100:
            data_unlabeled = iter_unlabeled.next()

        img_labeled_q = data_labeled[0][0].to(device)
        img_labeled_k = data_labeled[0][1].to(device)
        label = data_labeled[1].to(device)
        

        ## For Labeled Data
        PFA_logit_labeled, PFA_label_labeled, feat_labeled = model(img_labeled_q, img_labeled_k, label, l_flag=True)
        out = classifier(feat_labeled)
        classifier_loss = criterions['CrossEntropy'](out, label)
        if not model.skip_flag:
            # warmup the memory bank
            PFA_loss_labeled = criterions['CrossEntropy'](PFA_logit_labeled, PFA_label_labeled) * args.w_pfa_l
        else:
            PFA_loss_labeled = torch.tensor(0.).cuda()
        total_loss = classifier_loss + PFA_loss_labeled
        
        if args.label_ratio != 100:
            ## For Unlabeled Data
            img_unlabeled_pl = data_unlabeled[0][0].to(device)  # weak aug for generating pl
            img_unlabeled_q = data_unlabeled[0][1].to(device)
            img_unlabeled_k = data_unlabeled[0][2].to(device)
            label_ul = data_unlabeled[1].to(device)
            with torch.no_grad():
                q_f_unlabeled = model.inference(img_unlabeled_pl)
                logit_unlabeled = classifier(q_f_unlabeled)
                prob_unlabeled = torch.softmax(logit_unlabeled.detach(), dim=-1)
                confidence_unlabeled, predict_unlabeled = torch.max(prob_unlabeled, dim=-1)
                satisfied_mask = (confidence_unlabeled >= args.threshold)
                satisfied_mask = satisfied_mask.float()
            PFA_logit_unlabeled, PFA_label_unlabeled, _ = model(img_unlabeled_q, img_unlabeled_k, predict_unlabeled, l_flag=False, mem_use_confid=args.mem_use_confid, confid_q=confidence_unlabeled)
            ce_loss = nn.CrossEntropyLoss(reduction="none")
            if not model.skip_flag:
                PFA_loss_unlabeled = (ce_loss(PFA_logit_unlabeled, PFA_label_unlabeled) * satisfied_mask).mean() * args.w_pfa_ul
            else:
                PFA_loss_unlabeled = torch.tensor(0.).cuda()
            total_loss += PFA_loss_unlabeled

        total_loss.backward()
        optimizer.step()
        scheduler.step()


        ## Calculate the training accuracy of current iteration
        if iter_num % 100 == 0:
            _, predict = torch.max(out, 1)
            hit_num = (predict == label).sum().item()
            sample_num = predict.size(0)
            print("iter_num: {}; current acc: {}".format(iter_num, hit_num / float(sample_num)))

        ## Show Loss in TensorBoard
        writer.add_scalar('loss/classifier_loss', classifier_loss, iter_num)
        writer.add_scalar('loss/PFA_loss_labeled', PFA_loss_labeled, iter_num)
        writer.add_scalar('loss/total_loss', total_loss, iter_num)
        if args.label_ratio != 100:
            writer.add_scalar('loss/PFA_loss_unlabeled', PFA_loss_unlabeled, iter_num)
            writer.add_scalar('monitor/mask', satisfied_mask.mean(), iter_num)
            pl_acc = (predict_unlabeled == label_ul).float()
            writer.add_scalar('monitor/pl_acc', pl_acc.mean(), iter_num)
            if satisfied_mask.sum() > 0:
                mask_acc = (pl_acc * satisfied_mask).sum() / satisfied_mask.sum()
            else:
                mask_acc = 0
            writer.add_scalar('monitor/mask_acc', mask_acc, iter_num)

        if iter_num % args.test_interval == 1 or iter_num == 500:
            model.eval()
            classifier.eval()
            if 'cifar100' in args.root:
                test_acc = test_cifar(dataset_loaders, model, classifier, device=device)
            else:
                test_acc = test(dataset_loaders, model, classifier, device=device)
            print("iter_num: {}; test_acc: {}".format(iter_num, test_acc))
            writer.add_scalar('acc/test_acc', test_acc, iter_num)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = {'model': model.state_dict(),
                              'classifier': classifier.state_dict(),
                              'step': iter_num,
                              }
    print("best acc: %.4f" % (best_acc))
    torch.save(best_model, model_path)
    print("The best model has been saved in ", model_path)

def read_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='root path of dataset')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--label_ratio', type=int, default=15)
    parser.add_argument('--logdir', type=str, default='../vis/')
    parser.add_argument('--lr', type=float, default='0.001')
    parser.add_argument('--seed', type=int, default='666666')
    parser.add_argument('--workers', type=int, default='4')
    parser.add_argument('--lr_ratio', type=float, default='10')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--queue_size', type=int, default=32, help='queue size for each class')
    parser.add_argument('--momentum', type=float, default=0.999, help='the momentum hyperparameter for moving average')
    parser.add_argument('--projector_dim', type=int, default=1024)
    parser.add_argument('--class_num', type=int, default=200)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--max_iter', type=float, default=27005)
    parser.add_argument('--test_interval', type=float, default=3000)
    parser.add_argument("--pretrained", action="store_true", help="use the pre-trained model")
    parser.add_argument("--pretrained_path", type=str, default='~/.torch/models/moco_v2_800ep_pretrain.pth.tar')
    ## Only for Cifar100
    parser.add_argument("--expand_label", action="store_true", help="expand label to fit eval steps")
    parser.add_argument('--num_labeled', type=int, default=0, help='number of labeled data') 
    
    parser.add_argument('--threshold', type=float, default=0.95)
    # loss weight
    parser.add_argument('--w_pfa_l', type=float, default=1.0, help='loss weight of labeled metric loss')
    parser.add_argument('--w_pfa_ul', type=float, default=1.0, help='loss weight of unlabeled metric loss')
    # memory param
    parser.add_argument('--mem_use_confid', action='store_true', default=False, help='use of confid im memory accumulation')
    parser.add_argument('--momentum_proto', type=float, default=0.7, help='proto momentum')
    # temperature in metric loss
    parser.add_argument('--temp', type=float, default=0.1, help='the default temp in selftunging is 0.07')
    parser.add_argument('--resume', type=str, default=None, help='resume path')
    configs = parser.parse_args()
    return configs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    args = read_config()
    set_seed(args.seed)

    # Prepare data
    if 'cifar100' in args.root:
        args.class_num = 100
    elif 'CUB200' in args.root:
        args.class_num = 200
    elif 'StanfordCars' in args.root:
        args.class_num = 196
    elif 'Aircraft' in args.root:
        args.class_num = 100

    dataset_loaders = load_data(args)
    print("class_num: ", args.class_num)

    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    if 'cifar100' in args.root:
        model_name = "%s_%s_%s" % (args.backbone, os.path.basename(args.root), str(args.num_labeled))
    else:
        model_name = "%s_%s_%s" % (args.backbone, os.path.basename(args.root), str(args.label_ratio))
    logdir = os.path.join(args.logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(logdir)

    model_path = os.path.join(logdir, "%s_best.pkl" % (model_name))

    # Initialize model
    network, feature_dim = load_network(args.backbone)
    model = ProgressiveFeatureAdjustment(network=network, backbone=args.backbone, queue_size=args.queue_size, projector_dim=args.projector_dim, feature_dim=feature_dim,
                       class_num=args.class_num, momentum=args.momentum, temp=args.temp, pretrained=args.pretrained, pretrained_path=args.pretrained_path, 
                       confid=args.threshold, momentum_proto=args.momentum_proto).to(device)
    classifier = Classifier(feature_dim, args.class_num).to(device)
    print("backbone params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6 / 2))
    print("classifier params: {:.2f}M".format(sum(p.numel() for p in classifier.parameters()) / 1e6))

    if args.resume:
        model.load_state_dict(args.resume)
        classifier.load_state_dict(args.resume)
    
    ## Define Optimizer
    optimizer = optim.SGD([
        {'params': model.parameters()},
        {'params': classifier.parameters(), 'lr': args.lr * args.lr_ratio},
    ], lr= args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    milestones = [6000, 12000, 18000, 24000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    # Train model
    train(args, model, classifier, dataset_loaders, optimizer, scheduler, device=device, writer=writer, model_path=model_path)
    

if __name__ == '__main__':
    main()
