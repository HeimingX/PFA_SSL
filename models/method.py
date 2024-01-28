import torch
import torch.nn as nn


class ProgressiveFeatureAdjustment(nn.Module):
    """
    Build a metric learning based fixmatch model with: a query encoder, a key encoder, and a queue list
    V2: accumulate class prototype with ema
    """
    def __init__(self, network, backbone, queue_size=32, projector_dim=256, feature_dim=256,
                       class_num=200, momentum=0.999, temp=0.07, pretrained=True, pretrained_path=None, 
                       confid=0.95, momentum_proto=0.99, norm_proto=False):
        """
        network: the network of the backbone
        backbone: the name of the backbone
        queue_size: the queue size for each class
        projector_dim: the dimension of the projector (default: 1024)
        feature_dim: the dimension of the output from the backbone
        class_num: the class number of the dataset
        pretrained: loading from pre-trained model or not (default: True)
        momentum: the momentum hyperparameter for moving average to update key encoder (default: 0.999)
        temp: softmax temperature (default: 0.07)
        pretrained_path: the path of the pre-trained model
        """
        super(ProgressiveFeatureAdjustment, self).__init__()
        self.queue_size = queue_size
        self.momentum = momentum
        self.class_num = class_num
        self.backbone = backbone
        self.pretrained = pretrained
        self.temp = temp
        self.pretrained_path = pretrained_path
        self.confid = confid
        self.momentum_proto = momentum_proto
        self.norm_proto = norm_proto

        # create the encoders
        if 'efficientnet' in self.backbone:
            self.encoder_q = network(backbone=self.backbone, feature_dim=feature_dim, projector_dim=projector_dim)
            self.encoder_k = network(backbone=self.backbone, feature_dim=feature_dim, projector_dim=projector_dim)
        else:
            self.encoder_q = network(projector_dim=projector_dim)
            self.encoder_k = network(projector_dim=projector_dim)

        if backbone == 'MOCOv2':  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        self.load_pretrained(network)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # don't be updated by gradient
        
        # create the queue
        self.register_buffer("queue_list", torch.randn(projector_dim, self.class_num))
        self.queue_list = nn.functional.normalize(self.queue_list, dim=0)
        self.register_buffer("queue_pivot", torch.zeros(self.class_num, dtype=torch.long))  # pivot for warmup
        self.skip_flag = True

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, key_c, c):
        # momentum update
        if self.queue_pivot[c] == 0:
            self.queue_list[:, c:c+1] = key_c.T
            self.queue_pivot[c] += 1
        else:
            self.queue_list[:, c:c+1] = self.queue_list[:, c:c+1] * self.momentum_proto + key_c.T * (1. - self.momentum_proto)
            if self.norm_proto:
                self.queue_list[:, c:c+1] = nn.functional.normalize(self.queue_list[:, c:c+1], dim=0)

    def forward(self, im_q, im_k, labels, l_flag=False, mem_use_confid=False, confid_q=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        batch_size = im_q.size(0)

        # compute query features
        q_c, q_f = self.encoder_q(im_q)  # queries: q_c (N x projector_dim)
        q_c = nn.functional.normalize(q_c, dim=1)

        # compute key features
        self._momentum_update_key_encoder()  # update the key encoder
        k_c, k_f = self.encoder_k(im_k)  # keys: k_c (N x projector_dim)
        k_c = nn.functional.normalize(k_c, dim=1)

        # compute logits
        # cur_proto_list: feat_dim * class_num
        cur_proto_list = self.queue_list.clone().detach()
        if self.queue_pivot.sum() == self.class_num:
            self.skip_flag = False
        else:
            self.skip_flag = True

        for i in range(batch_size):
            if not l_flag:
                if mem_use_confid:
                    if confid_q[i] >= self.confid:
                        self._dequeue_and_enqueue(k_c[i:i+1], labels[i])
                else:
                    self._dequeue_and_enqueue(k_c[i:i+1], labels[i])
            else:
                self._dequeue_and_enqueue(k_c[i:i+1], labels[i])

        PFA_labels = labels
        PFA_logits = torch.einsum('nl,lk->nk', q_c, cur_proto_list)
        PFA_logits = PFA_logits / self.temp

        return PFA_logits, PFA_labels, q_f

    def load_pretrained(self, network):
        if self.backbone == 'MOCOv1' and self.pretrained:
            if self.pretrained_path is None:
                self.pretrained_path = "~/.torch/models/moco_v1_200ep_pretrain.pth.tar"
            ckpt = torch.load(self.pretrained_path)['state_dict']
            state_dict_cut = {}
            for k, v in ckpt.items():
                if not k.startswith("module.encoder_q."):
                    continue
                k = k.replace("module.encoder_q.", "")
                state_dict_cut[k] = v
            self.encoder_q.load_state_dict(state_dict_cut)
            print('Successfully load the pre-trained model of MOCOv1')
        elif self.backbone == 'MOCOv2' and self.pretrained:
            if self.pretrained_path is None:
                # self.pretrained_path = '~/.torch/models/moco_v2_800ep_pretrain.pth.tar'
                self.pretrained_path = 'models/moco_v2_800ep_pretrain.pth.tar'
            ckpt = torch.load(self.pretrained_path)['state_dict']
            state_dict_cut = {}
            for k, v in ckpt.items():
                if not k.startswith("module.encoder_q."):
                    continue
                # if 'fc.2' in k:
                #     continue
                if 'fc' in k:
                    continue
                k = k.replace("module.encoder_q.", "")
                state_dict_cut[k] = v
            self.encoder_q.load_state_dict(state_dict_cut, strict=False)
            print('Successfully load the pre-trained model of MOCOv2')
        elif 'resnet' in self.backbone:
            q = network(projector_dim=1000, pretrained=self.pretrained)
            q.fc = self.encoder_q.fc
            self.encoder_q = q
        elif 'densenet' in self.backbone:
            q = network(projector_dim=1000, pretrained=self.pretrained)
            q.classifier = self.encoder_q.classifier
            self.encoder_q = q

    def inference(self, img):
        _, feat = self.encoder_q(img)
        return feat

    def load_state_dict(self, ckpt_path, strict=True):
        ckpt = torch.load(ckpt_path)
        return super().load_state_dict(ckpt['model'], strict=strict)