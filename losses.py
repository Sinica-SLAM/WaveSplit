import torch
import torch.nn as nn
import torch.nn.functional as F

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
        Dist_ = Dist_.min(dim=1)[0]             # size of (Batch x Time)
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
