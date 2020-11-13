import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import _Loss

EPS = 1e-8

class RegularizationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, E):
        """
        E: Size of (Speaker x Dim) Embedding Table
        """
        Dist = (E[:,None] - E[None,:]).abs().sum(-1)
        Identity = torch.eye(E.shape[0])
        Dist = Dist.masked_fill( (Identity!=0).cuda() , 1000000 ) 

        return -torch.log( (Dist.min(dim=-1)[0])).mean()

class SpeakerVectorLoss(nn.Module):
    def __init__(self, alpha=10.0, beta=5.0, distance='l2'):
        super(SpeakerVectorLoss, self).__init__()
        # self.dist = nn.MSELoss(reduction='none')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.alpha = nn.Parameter(torch.ones(1)*alpha)
        self.beta = nn.Parameter(torch.ones(1)*beta)

    def forward(self, H, S, E, N=2):
        """Computes mel-spectrograms from a batch of waves
        Input
        ------
            H: Size of (Batch x N x Dim x Time)  
            S: Size of (Batch x N x Time) Speaker activate 
            E: Size of (Speaker x Dim) Embedding Table

        Output
        -------
            loss: Size of 1
        """

        Batch, N, Dim, Time = H.shape
        Speaker, Dim_ = E.shape
        assert Dim == Dim_

        # Get target with size of (Batch x N x Time) from speaker labels S
        E_S = F.embedding(S,E).transpose(2,3)   # size of (Batch x N x Dim x Time)
        H_ = H[:,None,:]                        # size of (Batch x 1 x N x Dim x Time)
        E_S_ = E_S[:,:,None]                    # size of (Batch x N x 1 x Dim x Time)
        
        Dist_ = ( H_ - E_S_).pow(2).sum(dim=3)  # size of (Batch x Nh x Ne x Time)

        Dist__ = []
        if N == 2:
            paths = [[[0,0],[1,1]],[[0,1],[1,0]]]
        for path in paths:
            tmp = torch.cat(
                    [
                        Dist_[:,p[0],p[1],:].unsqueeze(1)    
                        for p in path
                    ],
                    dim=1
                ).sum(dim=1,keepdim=True)       # size of (Batch x 1 x Time)
            Dist__.append(tmp)

        Dist_ = torch.cat(Dist__,dim=1)         # size of (Batch x Npath x Time)
        Dist_, idx_ = Dist_.min(dim=1)          # size of (Batch x Time)
        Dist_ = -1 * self.alpha * Dist_ + self.beta
            
        # Calculate global distance
        H_ = H[:,None]                          # size of (Batch x 1 x N x Dim x Time)
        E_ = E[None,:,None,:,None]              # size of (1 x Speaker x 1 x Dim x 1)

        Dist = ( H_ - E_).pow(2).sum(dim=3)    # size of (Batch x Speaker x N x Time)
        Dist = -1 * self.alpha * Dist + self.beta

        # Method 1
        return E_S[:,:,:,0] , - Dist_.mean() + torch.logsumexp(Dist,dim=1).mean()

        # Method 2
        #return self.cross_entropy( Dist, target)

class SingleSrcNegSDR(_Loss):
    """ Base class for single-source negative SI-SDR, SD-SDR and SNR.
        Args:
            sdr_type (string): choose between "snr" for plain SNR, "sisdr" for
                SI-SDR and "sdsdr" for SD-SDR [1].
            zero_mean (bool, optional): by default it zero mean the target and
                estimate before computing the loss.
            take_log (bool, optional): by default the log10 of sdr is returned.
            reduction (string, optional): Specifies the reduction to apply to
                the output:
            ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output.
        Shape:
            est_targets (:class:`torch.Tensor`): Expected shape [batch, time].
                Batch of target estimates.
            targets (:class:`torch.Tensor`): Expected shape [batch, time].
                Batch of training targets.
        Returns:
            :class:`torch.Tensor`: with shape [batch] if reduction='none' else
                [] scalar if reduction='mean'.
        Examples:
            >>> import torch
            >>> from asteroid.losses import PITLossWrapper
            >>> targets = torch.randn(10, 2, 32000)
            >>> est_targets = torch.randn(10, 2, 32000)
            >>> loss_func = PITLossWrapper(SingleSrcNegSDR("sisdr"),
            >>>                            pit_from='pw_pt')
            >>> loss = loss_func(est_targets, targets)
        References:
            [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
            International Conference on Acoustics, Speech and Signal
            Processing (ICASSP) 2019.
        """

    def __init__(self, sdr_type, zero_mean=True, take_log=True, reduction="none"):
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)

        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log

    def forward(self, est_target, target, dim=-1, keepdim=False):
        assert target.size() == est_target.size()
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(target, dim=dim, keepdim=True)
            mean_estimate = torch.mean(est_target, dim=dim, keepdim=True)
            target = target - mean_source
            est_target = est_target - mean_estimate
        # Step 2. Pair-wise SI-SDR.
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, 1]
            dot = torch.sum(est_target * target, dim=dim, keepdim=True)
            # [batch, 1]
            s_target_energy = torch.sum(target ** 2, dim=dim, keepdim=True) + EPS
            # [batch, time]
            scaled_target = dot * target / s_target_energy
        else:
            # [batch, time]
            scaled_target = target
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = est_target - target
        else:
            e_noise = est_target - scaled_target
        # [batch]
        losses = torch.sum(scaled_target ** 2, dim=dim, keepdim=keepdim) \
                / (torch.sum(e_noise ** 2, dim=dim, keepdim=keepdim) + EPS)
        if self.take_log:
            losses = 10 * torch.log10(losses + EPS)
        losses = losses.mean() if self.reduction == "mean" else losses
        return -losses

class PairwiseNegSDR(_Loss):
    """ Base class for pairwise negative SI-SDR, SD-SDR and SNR on a batch.
        Args:
            sdr_type (str): choose between "snr" for plain SNR, "sisdr" for
                SI-SDR and "sdsdr" for SD-SDR [1].
            zero_mean (bool, optional): by default it zero mean the target
                and estimate before computing the loss.
            take_log (bool, optional): by default the log10 of sdr is returned.
        Shape:
            est_targets (:class:`torch.Tensor`): Expected shape
                [batch, n_src, time]. Batch of target estimates.
            targets (:class:`torch.Tensor`): Expected shape
                [batch, n_src, time]. Batch of training targets.
        Returns:
            :class:`torch.Tensor`: with shape [batch, n_src, n_src].
            Pairwise losses.
        Examples:
            >>> import torch
            >>> from asteroid.losses import PITLossWrapper
            >>> targets = torch.randn(10, 2, 32000)
            >>> est_targets = torch.randn(10, 2, 32000)
            >>> loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"),
            >>>                            pit_from='pairwise')
            >>> loss = loss_func(est_targets, targets)
        References:
            [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
            International Conference on Acoustics, Speech and Signal
            Processing (ICASSP) 2019.
        """

    def __init__(self, sdr_type, zero_mean=True, take_log=True):
        super(PairwiseNegSDR, self).__init__()
        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log

    def forward(self, est_targets, targets):
        assert targets.size() == est_targets.size()
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source
            est_targets = est_targets - mean_estimate
        # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
        s_target = torch.unsqueeze(targets, dim=1)
        s_estimate = torch.unsqueeze(est_targets, dim=2)

        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, n_src, n_src, 1]
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
            # [batch, 1, n_src, 1]
            s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS
            # [batch, n_src, n_src, time]
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy
        else:
            # [batch, n_src, n_src, time]
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj
        # [batch, n_src, n_src]
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (
            torch.sum(e_noise ** 2, dim=3) + EPS
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + EPS)
        return -pair_wise_sdr


def PITminLoss(self,conf):
    source = conf['source']
    estimate = conf['estimate']
    loss = conf['Loss']

    recLoss = loss(estimate_wave,clean_wave).sum(dim=-1).sum(dim=-1,keepdim=True)
    source_swap = source
    source_swap[:,[0, 1],:] = source_swap[:,[1, 0],:]

    recLoss_swap = loss(estimate_wave,clean_wave).sum(dim=-1).sum(dim=-1,keepdim=True)
    recLoss = torch.cat([recLoss,recLoss_swap],dim=1)
    return torch.min(recLoss,dim=1)[0].sum() / 2


pairwise_neg_sisdr = PairwiseNegSDR("sisdr")