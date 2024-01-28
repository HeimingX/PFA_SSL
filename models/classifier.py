import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, inputs, class_num):
        super(Classifier, self).__init__()
        self.classifier_layer = nn.Linear(inputs, class_num)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, x):
        return self.classifier_layer(x)
    
    def load_state_dict(self, ckpt_path, strict=True):
        ckpt = torch.load(ckpt_path)
        return super().load_state_dict(ckpt['classifier'], strict=strict)