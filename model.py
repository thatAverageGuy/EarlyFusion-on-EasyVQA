import torch
from torch import nn
import numpy as np
class EasyVQAEarlyFusion(nn.Module):

    def __init__(self, hyperparms=None):

        super(EasyVQAEarlyFusion, self).__init__()        
        self.dropout = nn.Dropout(0.3)
        # self.vision_projection = nn.Linear(2048, 768) 
        # self.text_projection = nn.Linear(512, 768)
        self.fc1 = nn.Linear(768, 256) 
        self.bn1 = nn.BatchNorm1d(256)
        self.classifier = nn.Linear(256, 13) 
        W = torch.Tensor(768, 768)
        self.W = nn.Parameter(W)
        self.relu_f = nn.ReLU()
        # initialize weight matrices
        nn.init.kaiming_uniform_(self.W, a=np.sqrt(5))
        
    def forward(self, image_emb, text_emb):

        x1 = image_emb   
        x1 = torch.nn.functional.normalize(x1, p=2, dim=1)
        Xv = self.relu_f(x1)
        
        x2 = text_emb
        x2 = torch.nn.functional.normalize(x2, p=2, dim=1)
        Xt = self.relu_f(x2)
        
        # print(Xv.shape, Xt.shape)
        
        Xvt = Xv * Xt
        Xvt = self.relu_f(torch.mm(Xvt, self.W.t()))

        Xvt = self.fc1(Xvt)
        Xvt = self.bn1(Xvt)
        Xvt = self.dropout(Xvt)
        Xvt = self.classifier(Xvt)

        return Xvt