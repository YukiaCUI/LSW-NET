'''
@file: dr_spaam.py
@breif: the DR-SPAAM module
@author: Winter
@update: 2023.10.8
'''
import numpy as np
from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import DrspaamNetCriterion

def _conv1d(in_channel, out_channel, kernel_size, padding):
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm1d(out_channel),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )

def _conv1d_3(in_channel, out_channel):
    return _conv1d(in_channel, out_channel, kernel_size=3, padding=1)

class DrSpaam(nn.Module):
    '''
    DR-SPAAM Network

    Parameters
    ----------
    loss_kwargs: dict
        loss function configure parameters
    model_kwargs: dict
        model configure parameters
    '''
    def __init__(self, loss_kwargs: dict, model_kwargs: dict) -> None:
        super(DrSpaam, self).__init__()

        self.dropout = model_kwargs["DROPOUT"]
        self.max_num_pts = model_kwargs["MAX_NUM_PTS"]
        self.num_pts = model_kwargs["NUM_PTS"]
        self.alpha = model_kwargs["ALPHA"]
        self.d_model = model_kwargs["D_MODEL"]
        self.win_size = model_kwargs["WIN_SIZE"]
        self.panoramic_scan = model_kwargs["PANORAMIC_SCAN"]

        self.criterion = DrspaamNetCriterion(loss_cfg=loss_kwargs)

        # backbone
        self.conv_block_1 = nn.Sequential(
            _conv1d_3(1, 64), _conv1d_3(64, 64), _conv1d_3(64, 128)
        )
        self.conv_block_2 = nn.Sequential(
            _conv1d_3(128, 128), _conv1d_3(128, 128), _conv1d_3(128, 256)
        )
        self.conv_block_3 = nn.Sequential(
            _conv1d_3(256, 256), _conv1d_3(256, 256), _conv1d_3(256, 512)
        )
        self.conv_block_4 = nn.Sequential(_conv1d_3(512, 256), _conv1d_3(256, 128))

        # detection layer
        self.conv_cls = nn.Conv1d(128, 1, kernel_size=1)
        self.conv_reg = nn.Conv1d(128, 2, kernel_size=1)

        # spatial attention
        self.gate = _SpatialAttentionMemory(
            n_pts=int(ceil(self.num_pts / 4)),
            n_channel=256,
            embedding_length=self.d_model,
            alpha=self.alpha,
            window_size=self.win_size,
            panoramic_scan=self.panoramic_scan,
        )

        # initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __str__(self) -> str:
        return "DR-SPAAM"

    def forward(self, x, inference=False):
        """
        Args:
            x (tensor[B, CT, N, L]): (batch, cutout, scan, points per cutout)
            inference (bool, optional): Set to true for sequencial inference
                (i.e. in deployment). Defaults to False.

        Returns:
            pred_cls (tensor[B, CT, C]): C = number of class
            pred_reg (tensor[B, CT, 2])
        """
        B, CT, N, L = x.shape

        if not inference:
            self.gate.reset()

        # NOTE: Ablation study, DR-SPA, no auto-regression, only two consecutive scans
        # x = x[:, :, -2:, :]

        # process scan sequentially
        n_scan = x.shape[2]
        for i in range(n_scan):
            x_i = x[:, :, i, :]  # (B, CT, L)

            # extract feature from current scan
            out = x_i.view(B * CT, 1, L)
            out = self._conv_and_pool(out, self.conv_block_1)  # /2
            out = self._conv_and_pool(out, self.conv_block_2)  # /4
            out = out.view(B, CT, out.shape[-2], out.shape[-1])  # (B, CT, C, L)

            # combine current feature with memory
            out, sim = self.gate(out)  # (B, CT, C, L)

        # detection using combined feature memory
        out = out.view(B * CT, out.shape[-2], out.shape[-1])
        out = self._conv_and_pool(out, self.conv_block_3)  # /8
        out = self.conv_block_4(out)
        out = F.avg_pool1d(out, kernel_size=int(out.shape[-1]))  # (B * CT, C, 1)

        pred_cls = self.conv_cls(out).view(B, CT, -1)  # (B, CT, cls)
        pred_reg = self.conv_reg(out).view(B, CT, 2)  # (B, CT, 2)

        return pred_cls, pred_reg

    def _conv_and_pool(self, x, conv_block):
        out = conv_block(x)
        out = F.max_pool1d(out, kernel_size=2)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)

        return out

    def run(self, batch: dict):
        '''
        Run the inference.

        Parameters
        ----------
        batch: dict
            batch information with annotations
        
        Return
        ----------
        losses: Tensor
            weighted loss value
        tb_dict: dict
            information for tensorboard
        rtn_dict: dict
            information to return
        '''
        tb_dict, rtn_dict = {}, {}

        x = batch["input"]
        target_cls, target_reg = batch["target_cls"], batch["target_reg"]
        B, N = target_cls.shape

        # train only on part of scan, if the GPU cannot fit the whole scan
        if self.training and N > self.max_num_pts:
            idx0 = np.random.randint(0, N - self.max_num_pts)
            idx1 = idx0 + self.max_num_pts
            target_cls = target_cls[:, idx0:idx1]
            target_reg = target_reg[:, idx0:idx1, :]
            net_input = net_input[:, idx0:idx1, :, :]
            N = self.max_num_pts

        # to gpu
        x = torch.from_numpy(x).cuda(non_blocking=True).float()
        target_cls = torch.from_numpy(target_cls).cuda(non_blocking=True).float()
        target_reg = torch.from_numpy(target_reg).cuda(non_blocking=True).float()

        # forward pass
        pred_cls, pred_reg, _ = self.forward(x)

        outputs = {"pred_cls": pred_cls, "pred_reg": pred_reg}
        targets = {"target_cls": target_cls, "target_reg": target_reg}

        # loss calculation
        losses = self.criterion(outputs, targets)
        for loss_type, loss_val in losses.items():
            tb_dict[loss_type] = loss_val if isinstance(loss_val, float) else loss_val.item()

        # losses sum-up
        losses = sum([losses[i] * self.criterion.loss_weight[i] for i in losses.keys()
            if i in self.criterion.loss_weight])
        
        rtn_dict["pred_reg"] = pred_reg.view(B, N, 2)
        rtn_dict["pred_cls"] = pred_cls.view(B, N)

        return losses, tb_dict, rtn_dict


class _SpatialAttentionMemory(nn.Module):
    def __init__(
        self, n_pts, n_channel, embedding_length, alpha, window_size, panoramic_scan
    ):
        """A memory network that updates with similarity-based spatial attention and
        auto-regressive model.

        Args:
            n_pts (int): Length of the input sequence (cutout)
            n_channel (int): Channel of the input sequence
            embedding_length (int): Each cutout is converted to an embedding vector
            alpha (float): Auto-regressive update rate, in range [0, 1]
            window_size (int): Full neighborhood window size to compute attention
            panoramic_scan (bool): True if the scan span 360 degree, used to warp
                window indices accordingly
        """
        super(_SpatialAttentionMemory, self).__init__()
        self._alpha = alpha
        self._window_size = window_size
        self._embedding_length = embedding_length
        self._panoramic_scan = panoramic_scan

        self.conv = nn.Sequential(
            nn.Conv1d(n_channel, self._embedding_length, kernel_size=n_pts, padding=0),
            nn.BatchNorm1d(self._embedding_length),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self._memory = None

        # place holder, created at runtime
        self.neighbor_masks, self.neighbor_inds = None, None

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def reset(self):
        self._memory = None

    def forward(self, x_new):
        if self._memory is None:
            self._memory = x_new
            return self._memory, None

        # NOTE: Ablation study, DR-AM, no spatial attention
        # self._memory = self._alpha * x_new + (1.0 - self._alpha) * self._memory
        # return self._memory, None

        n_batch, n_cutout, n_channel, n_pts = x_new.shape

        # only need to generate neighbor mask once
        if (
            self.neighbor_masks is None
            or self.neighbor_masks.shape[0] != x_new.shape[1]
        ):
            self.neighbor_masks, self.neighbor_inds = self._generate_neighbor_mask(
                x_new
            )

        # embedding for cutout
        emb_x = self.conv(x_new.view(n_batch * n_cutout, n_channel, n_pts))
        emb_x = emb_x.view(n_batch, n_cutout, self._embedding_length)

        # embedding for template
        emb_temp = self.conv(self._memory.view(n_batch * n_cutout, n_channel, n_pts))
        emb_temp = emb_temp.view(n_batch, n_cutout, self._embedding_length)

        # pair-wise similarity (batch, cutout, cutout)
        sim = torch.matmul(emb_x, emb_temp.permute(0, 2, 1))

        # masked softmax
        sim = sim - 1e10 * (
            1.0 - self.neighbor_masks
        )  # make sure the out-of-window elements have small values
        maxes = sim.max(dim=-1, keepdim=True)[0]
        exps = torch.exp(sim - maxes) * self.neighbor_masks
        exps_sum = exps.sum(dim=-1, keepdim=True)
        sim = exps / exps_sum

        # weighted average on the template
        atten_memory = self._memory.view(n_batch, n_cutout, n_channel * n_pts)
        atten_memory = torch.matmul(sim, atten_memory)
        atten_memory = atten_memory.view(n_batch, n_cutout, n_channel, n_pts)

        # update memory using auto-regressive
        self._memory = self._alpha * x_new + (1.0 - self._alpha) * atten_memory

        return self._memory, sim

    def _generate_neighbor_mask(self, x):
        # indices of neighboring cutout
        n_cutout = x.shape[1]
        hw = int(self._window_size / 2)
        inds_col = torch.arange(n_cutout).unsqueeze(dim=-1).long()
        window_inds = torch.arange(-hw, hw + 1).long()
        inds_col = inds_col + window_inds.unsqueeze(dim=0)  # (cutout, neighbors)
        # NOTE On JRDB, DR-SPAAM takes part of the panoramic scan and at test time
        # takes the whole panoramic scan
        inds_col = (
            inds_col % n_cutout
            if self._panoramic_scan and not self.training
            else inds_col.clamp(min=0, max=n_cutout - 1)
        )
        inds_row = torch.arange(n_cutout).unsqueeze(dim=-1).expand_as(inds_col).long()
        inds_full = torch.stack((inds_row, inds_col), dim=2).view(-1, 2)
        # self.register_buffer('neighbor_inds', inds_full)

        masks = torch.zeros(n_cutout, n_cutout).float()
        masks[inds_full[:, 0], inds_full[:, 1]] = 1.0
        return masks.cuda(x.get_device()) if x.is_cuda else masks, inds_full
