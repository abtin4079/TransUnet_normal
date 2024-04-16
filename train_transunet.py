import torch
from torch.optim import SGD

# Additional Scripts
from utils.transunet import TransUNet
from utils.utils import dice_loss
from config import cfg
from utils.metrics import *

class TransUNetSeg:
    def __init__(self, device):
        self.device = device
        self.model = TransUNet(img_dim=cfg.transunet.img_dim,
                               in_channels=cfg.transunet.in_channels,
                               out_channels=cfg.transunet.out_channels,
                               head_num=cfg.transunet.head_num,
                               mlp_dim=cfg.transunet.mlp_dim,
                               block_num=cfg.transunet.block_num,
                               patch_dim=cfg.transunet.patch_dim,
                               class_num=cfg.transunet.class_num).to(self.device)

        self.criterion = dice_loss
        self.optimizer = SGD(self.model.parameters(), lr=cfg.learning_rate,
                             momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    def load_model(self, path):
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        self.model.eval()

    def train_step(self, **params):
        self.model.train()

        self.optimizer.zero_grad()
        pred_mask = self.model(params['img'])
        loss = self.criterion(pred_mask, params['mask'])
        loss.backward()
        self.optimizer.step()

        # Compute accuracy
        accuracy = self.pixelwise_accuracy(pred_mask, params['mask'])


        return loss.item(), pred_mask, accuracy

    def pixelwise_accuracy(self, pred_mask, true_mask):
        # Flatten predicted and true masks
        pred_flat = pred_mask.view(-1)
        true_flat = true_mask.view(-1)

        # Compute accuracy
        correct = (pred_flat == true_flat).float().sum()
        total_pixels = true_flat.numel()
        accuracy = correct / total_pixels

        return accuracy.item()   






    def test_step(self, **params):
        self.model.eval()

        pred_mask = self.model(params['img'])
        loss = self.criterion(pred_mask, params['mask'])
        


        return loss.item(), pred_mask
