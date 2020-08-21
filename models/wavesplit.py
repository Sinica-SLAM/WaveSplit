import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import k_means
from ..utils import torch_utils
from ..filterbanks import make_enc_dec
from ..masknn import TDConvNet
from ..losses import SpeakerVectorLoss
#from .base_models import BaseTasNet
import ipdb
class WaveSplit(nn.Module):
    def __init__(self, n_src, out_chan=None, 
                 n_blocks_spk=14, n_repeats_spk=1, bn_chan_spk=128, hid_chan_spk=512, skip_chan_spk=128, 
                 n_blocks_sep=11, n_repeats_sep=4, bn_chan_sep=128, hid_chan_sep=512, skip_chan_sep=128,
                 conv_kernel_size=3, norm_type="gLN", mask_act_spk='relu', mask_act_sep='relu', 
                 in_chan=None, fb_name='free', kernel_size=16, n_filters=512, stride=8, **fb_kwargs):

        super(WaveSplit, self).__init__()
        encoder, decoder = make_enc_dec(
            fb_name, kernel_size=kernel_size, n_filters=n_filters,
            stride=stride, **fb_kwargs
        )
        n_feats = encoder.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, ('Number of filterbank output channels'
                                        ' and number of input channels should '
                                        'be the same. Received '
                                        f'{n_feats} and {in_chan}')
        ## Speaker Stack: Output Speaker Vector (Batch x N x Dim x Time)
        speakerstack = TDConvNet(
            n_feats, n_src, out_chan=out_chan, n_blocks=n_blocks_spk,
            n_repeats=n_repeats_spk, bn_chan=bn_chan_spk, hid_chan=hid_chan_spk,
            skip_chan=skip_chan_spk, conv_kernel_size=conv_kernel_size,
            norm_type=norm_type, mask_act=mask_act_spk
        )
        separationstack = TDConvNet(
            n_feats, 1, out_chan=out_chan, n_blocks=n_blocks_sep,
            n_repeats=n_repeats_sep, bn_chan=bn_chan_sep, hid_chan=hid_chan_sep,
            skip_chan=skip_chan_sep, conv_kernel_size=conv_kernel_size,
            norm_type=norm_type, mask_act=mask_act_sep
        )
        speakervectorloss = SpeakerVectorLoss()
        
        self.Encoder = encoder
        self.Decoder = decoder
        self.SpeakerStack = speakerstack
        self.SeparationStack = separationstack
        self.SpeakerVectorLoss = speakervectorloss
        self.ReconstructionLoss = nn.MSELoss(reduction='sum') 
        self.Emb_table = nn.Parameter(torch.randn(129,hid_chan_spk)) ##Total Speaker,hid dim

    def forward(self, inputs):
        if self.training: #training
            mix_wave, clean_wave, spk_label = inputs
            enc_wave = self.Encoder(mix_wave)
            speaker_vector = self.SpeakerStack(enc_wave,None)
            spk_label_resample = fix_target_size(spk_label, speaker_vector.size(-1), dim=-1)
            Speaker_centroids , PITLoss = self.SpeakerVectorLoss(speaker_vector,spk_label_resample,self.Emb_table)
            RecLoss = 0
            masked_wave = list()
            for n in range(2):
                speaker_centroid = Speaker_centroids[:,n,:] #B*N*D
                mask = self.SeparationStack(enc_wave, speaker_centroid)
                masked_wave.append(mask * enc_wave.unsqueeze(1))
            masked_wave = torch.cat(masked_wave,dim=1)
            estimate_wave = torch_utils.pad_x_to_y(self.Decoder(masked_wave), clean_wave)
            RecLoss = self.ReconstructionLoss(estimate_wave,clean_wave)
            loss = PITLoss + RecLoss
            loss_dict = dict()
            loss_dict["PITLoss"] = PITLoss
            loss_dict["RecLoss"] = RecLoss / 2
            return loss, loss_dict

        else: #val
            if isinstance(inputs, torch.Tensor):
                mix_wave = inputs
            elif len(inputs) is 3:
                mix_wave, clean_wave, spk_label = inputs
            elif len(inputs) is 2:
                mix_wave, clean_wave = inputs
            elif len(inputs) is 1:
                mix_wave = inputs
            else:
                print("Wrong inputs!")
                exit()

            Batch = mix_wave.shape[0]
            enc_wave = self.Encoder(mix_wave)
            speaker_vector = self.SpeakerStack(enc_wave,None)
            
            speaker_vector_trans = speaker_vector.transpose(2,3).contiguous() # B*N*D*T to B*N*T*D
            speaker_vector_trans = speaker_vector_trans.view(Batch,-1,speaker_vector.shape[3]).cpu().detach().numpy() #B*NT*D
            Speaker_centroids = np.array([
                    k_means( samples, speaker_vector.shape[1])[0] 
                    for samples in speaker_vector_trans
                ])
            Speaker_centroids = torch.from_numpy(Speaker_centroids).float().to(speaker_vector.device)
            RecLoss = 0
            masked_wave = list()
            for n in range(2):
                speaker_centroid = Speaker_centroids[:,n,:] #B*N*D
                mask = self.SeparationStack(enc_wave, speaker_centroid)
                masked_wave.append(mask * enc_wave.unsqueeze(1))
            masked_wave = torch.cat(masked_wave,dim=1)              
            if isinstance(inputs, torch.Tensor) or len(inputs) is 1:
                return decoder(masked_wave)
            else:
                estimate_wave = torch_utils.pad_x_to_y(self.Decoder(masked_wave), clean_wave)
                RecLoss = self.ReconstructionLoss(estimate_wave,clean_wave)
                return estimate_wave, RecLoss / 2
            

def fix_target_size(target, target_length, dim=-1):
    assert dim == -1
    target_resample = target[...,:1].repeat(1,1,target_length)
    target_resample = target[...,::80]
    if target_resample.size(-1) >= target_length:
        target_resample = target_resample[...,:target_length]
    else:
        target_resample = torch.cat([
            target_resample,
            target_resample[...,:1].repeat(1,1,target_length-target_resample.size(-1))
        ],dim=-1)
    return target_resample


    """ ConvTasNet separation model, as described in [1].

    Args:
        n_src (int): Number of sources in the input mixtures.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
            If 0 or None, TDConvNet won't have any skip connections and the
            masks will be computed from the residual output.
            Corresponds to the ConvTasnet architecture in v1 or the paper.
        conv_kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.
        in_chan (int, optional): Number of input channels, should be equal to n_filters.
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``].
        n_filters (int): Number of filters / Input dimension of the masker net.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.
    """
    