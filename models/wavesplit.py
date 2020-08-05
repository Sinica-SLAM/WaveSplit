import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .. import torch_utils
from ..filterbanks import make_enc_dec
from ..masknn import TDConvNet
from .base_models import BaseTasNet

class BaseWaveSplit(nn.Module)
    def __init__(self, encoder, decoder, speakerstack, separationstack, speakervectorloss):
        super().__init__()
        self.Encoder = encoder
        self.Decoder = decoder
        self.SpeakerStack = speakerstack
        self.SeparationStack = separationstack
        self.SpeakerVectorLoss = speakervectorloss 
    def foward(self, wave, spk_label, emb_table):
        enc_wave = self.Encoder(wave)
        speaker_vector = self.SpeakerStack(enc_wave,None)
        ###
        Speaker_centroid , _ = self.SpeakerVectorLoss(speaker_vector,spk_label,emb_table)
        ###
        mask = self.SeparationStack(enc_wave,speaker_centroid)
        masked_wave = mask * enc_wave.unsqueeze(1)
        estimate_wave = torch_utils.pad_x_to_y(self.decoder(masked_wave), wave)
        return speaker_vector,estimate_wave

class WaveSplit(BaseWaveSplit):
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
    
    def __init__(self, n_src, out_chan=None, n_blocks_speaker=14, n_repeats_speaker=1,
                 n_blocks_separation=11, n_repeats_speaker_separation=4,
                 bn_chan=128, hid_chan=512, skip_chan=128, conv_kernel_size=3,
                 norm_type="gLN", mask_act='relu', in_chan=None, fb_name='free',
                 kernel_size=16, n_filters=512, stride=8, **fb_kwargs):

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
            n_feats, n_src, out_chan=out_chan, n_blocks=n_blocks_speaker,
            n_repeats=n_repeats_speaker, bn_chan=bn_chan, hid_chan=hid_chan,
            skip_chan=skip_chan, conv_kernel_size=conv_kernel_size,
            norm_type=norm_type, mask_act=mask_act
        )
        separationstack = TDConvNet(
            n_feats, n_src, out_chan=out_chan, n_blocks=n_blocks_separation,
            n_repeats=n_repeats_separation, bn_chan=bn_chan, hid_chan=hid_chan,
            skip_chan=skip_chan, conv_kernel_size=conv_kernel_size,
            norm_type=norm_type, mask_act=mask_act
        )
        speakervectorloss = SpeakerVectorLoss()
        super().__init__(encoder, decoder, speakerstack, separationstack, speakervectorloss)