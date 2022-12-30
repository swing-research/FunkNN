import torch
import normflow as nf
import numpy as np

def real_nvp(latent_dim, K=64):
    

    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_dim)])
    flows = []
    for i in range(K):
        s = nf.nets.MLP([latent_dim, 2 * latent_dim, latent_dim], init_zeros=True)
        t = nf.nets.MLP([latent_dim, 2 * latent_dim, latent_dim], init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(latent_dim)]
    
    q0 = nf.distributions.DiagGaussian(latent_dim)
    
    # Construct flow model
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    
    return nfm
