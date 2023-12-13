from typing import Optional, Tuple
import torch.nn as nn
import torch as tc
import math
from torch.linalg import pinv
import numpy as np


class PLRNN(nn.Module):
    """
    Piece-wise Linear Recurrent Neural Network (Durstewitz 2017)

    Args:
        dim_x: Dimension of the observations
        dim_z: Dimension of the latent states (number of hidden neurons)
        n_bases: Number of bases to use in the BE-PLRNN
        clip_range: latent state clipping value
        latent_model: Name of the latent model to use. Has to be in LATENT_MODELS
        layer_norm: Use LayerNormalization (no learnable parameters currently)
    """

    LATENT_MODELS = ['PLRNN', 'clipped-PLRNN', 'dendr-PLRNN', 'clipped-shPLRNN']

    def __init__(self, dim_x: int, dim_z: int, dim_s: int, dim_h: int, dim_Enc: int, n_trials: int, W_trial: bool, n_bases: int, clip_range: float,
                 latent_model: str, mean_center: bool, process_noise: float, h_trial: int, pm_noise: float):
        super(PLRNN, self).__init__()
        self.d_x = dim_x
        self.d_z = dim_z
        self.d_s = dim_s
        self.dh = dim_h
        self.d_Enc = dim_Enc
        self.ntrials = n_trials
        self.W_trial = W_trial
        self.h_trial = h_trial
        self.n_bases = n_bases
        self.use_bases = False
        self.process_noise = process_noise
        self.pm_noise = pm_noise

        if latent_model == 'shPLRNN':
            if n_bases > 0:
                print("Chosen model is vanilla PLRNN, the bases Parameter has no effect here!")
            self.latent_step = PLRNN_Step(dz=self.d_z, ds=self.d_s, dim_hidden=self.dh, trials=self.ntrials,
                                          W_trial=self.W_trial, clip_range=clip_range, mean_center=mean_center,
                                          noise=self.process_noise)
        elif latent_model == 'clipped-shPLRNN':
            if n_bases > 0:
                print("Chosen model is clipped shPLRNN, the bases Parameter has no effect here!")
            self.latent_step = Clipped_shPLRNN_Step(dz=self.d_z, ds=self.d_s, dim_hidden=self.dh, trials=self.ntrials,
                                                    W_trial=self.W_trial, clip_range=clip_range, mean_center=mean_center,
                                                    noise=self.process_noise, h_trial=self.h_trial)
        else:
            if latent_model == 'clipped-PLRNN':
                assert n_bases > 0, "n_bases has to be a positive integer for clipped-PLRNN!"
                self.latent_step = PLRNN_Clipping_Step(self.n_bases, dz=self.d_z, clip_range=clip_range, mean_center=mean_center)
                self.use_bases = True
            elif latent_model == 'dendr-PLRNN':
                assert n_bases > 0, "n_bases has to be a positive integer for dendr-PLRNN!"
                self.latent_step = PLRNN_Basis_Step(self.n_bases, dz=self.d_z, ds=self.d_s, clip_range=clip_range, mean_center=mean_center)
                self.use_bases = True
            else:
                raise NotImplementedError(f"{latent_model} is not yet implemented. Use one of: {self.LATENT_MODELS}.")

    def get_latent_parameters(self, indices=None):
        A = self.latent_step.A
        if (indices is None) or (not self.W_trial):
            W1 = self.latent_step.W1
            W2 = self.latent_step.W2
        else:
            W1 = self.latent_step.W1[indices]
            W2 = self.latent_step.W2[indices]
        if (indices is None) or (not self.h_trial):
            h1 = self.latent_step.h1
        else:
            h1 = self.latent_step.h1[indices]
        h2 = self.latent_step.h2
        C = self.latent_step.C
        return A, W1, W2, h1, h2, C

    def get_basis_expansion_parameters(self):
        alphas = self.latent_step.alphas
        thetas = self.latent_step.thetas
        return alphas, thetas

    def get_parameters(self, indices=None):
        params = self.get_latent_parameters(indices)
        if self.use_bases:
            params += self.get_basis_expansion_parameters()
        return params

    @tc.no_grad()
    def add_pm_noise(self, pms):
        pms_ = []
        for pm in pms:
            pm_noise = tc.normal(mean=0., std=1.0*self.pm_noise, size=pm.shape).to(pm.device)
            pm += pm_noise
            pms_.append(pm)
        return pms_

    def forward(self, Z_, s, alpha, z0=None):
        '''
        Forward pass with observations interleaved every n-th step.
        Credit @Florian Hess
        '''

        indices, z = Z_  ## indices of trials and Z_enc(batch_size,seq_len,d_enc)
        # switch dimensions for performance reasons
        z_p = z.permute(1, 0, 2).detach()
        s_ = s.permute(1, 0, 2) #external_inputs
        T, S, dz = z_p.size()

        # no interleaving obs. if alpha is not specified
        if alpha is None:
            alpha = 0

        # initial state
        if z0 is None:
            z0 = tc.zeros(size=(S, self.d_z), device=z_p.device)
            z0[:, :self.d_Enc] = z_p[0]

        # stores whole latent state trajectory
        Z = tc.zeros(size=(T, S, self.d_z), device=z_p.device)

        # gather parameters
        params = self.get_parameters(indices)
        if self.pm_noise:
            params = self.add_pm_noise(params)
        z = z0
        for t in range(T):
            # interleave observation every n time steps
            z = self.teacher_force(z, z_p[t], alpha)
            z = self.latent_step(z, s_[t], *params)
            Z[t] = z

        return Z.permute(1, 0, 2)
        

    @tc.no_grad()
    def generate(self, T, inputs, z0, indices=None, noise=0):
        '''
        Generate a trajectory of T time steps given
        an initial condition z0. If no initial condition
        is specified, z0 is teacher forced.
        '''
        # holds the whole generated trajectory
        Z = tc.zeros((T, 1, self.d_z), device=inputs.device)
        #S = tc.empty((T, 1, self.d_s), device=data.device)

        S = inputs   #######

        if T > S.shape[0]:
            tmp = tc.zeros((T,S.shape[1]))
            tmp[:S.shape[0]] = S

        Z[0] = z0
        params = self.get_parameters(indices)
        self.latent_step.noise = noise
        for t in range(1, T):
            Z[t] = self.latent_step(Z[t-1], S[t], *params)

        return Z.squeeze_(1)

    ### CRISTIAN CODE
    @tc.no_grad()
    def generate_test(self,T,inputs,z0, W2, W1 ,indices=None):
        '''
        Generate a trajectory of T time steps given
        an initial condition z0. If no initial condition
        is specified, z0 is teacher forced.
        '''
        # holds the whole generated trajectory
        Z = tc.empty((T, 1, self.d_z), device=inputs.device)
        #S = tc.empty((T, 1, self.d_s), device=data.device)

        S = inputs   #######

        Z[0] = z0
        params = self.get_parameters(indices)
        params = list(params)
        params[1] = W1
        params[2] = W2
        params = tuple(params)
        noise=0
        self.latent_step.noise = noise
        for t in range(1, T):
            Z[t] = self.latent_step(Z[t-1], S[t], *params)

        return Z.squeeze_(1)
    ###

    def teacher_force(self, z: tc.Tensor, z_: tc.Tensor, alpha) -> tc.Tensor:
        '''
        Applies weak teacher forcing.
        '''
        z[:, :self.d_Enc] = alpha * z_ + (1 - alpha) * z[:, :self.d_Enc]
        return z


class Latent_Step(nn.Module):
    def __init__(self, dz, ds, dim_hidden, trials, W_trial, clip_range=None, mean_center=False, noise=0., h_trial=0):
        super(Latent_Step, self).__init__()
        self.clip_range = clip_range
        #self.nonlinearity = nn.ReLU()
        self.dz = dz
        self.ds = ds
        self.dim_hidden = dim_hidden
        self.trials = trials
        self.W_trial = W_trial
        self.h_trial = h_trial
        self.noise = noise

        if mean_center:
            self.norm = lambda z: z - z.mean(dim=1, keepdim=True)
        else:
            self.norm = nn.Identity()

    def init_uniform(self, shape: Tuple[int]) -> nn.Parameter:
        # empty tensor
        tensor = tc.zeros(*shape)
        # value range
        r = 1 / math.sqrt(shape[-1])
        nn.init.uniform_(tensor, -r, r)
        return nn.Parameter(tensor, requires_grad=True)
    
    def init_uniform_same(self, shape: Tuple[int]) -> nn.Parameter:
        # empty tensor
        tr, a, b = shape
        tensor = tc.zeros((a, b))
        # value range
        r = 1 / math.sqrt(shape[-1])
        nn.init.uniform_(tensor, -r, r)
        tensor = tensor.repeat(tr, 1, 1)
        return nn.Parameter(tensor, requires_grad=True)
    
    def init_uniform_same_h(self, shape: Tuple[int]) -> nn.Parameter:
        # empty tensor
        tr, a = shape
        tensor = tc.zeros(a)
        # value range
        r = 1 / math.sqrt(shape[-1])
        nn.init.uniform_(tensor, -r, r)
        tensor = tensor.repeat(tr, 1)
        return nn.Parameter(tensor, requires_grad=True)

    def init_AW(self):
        # from: Talathi & Vartak 2016: Improving Performance of Recurrent Neural Network with ReLU Nonlinearity
        matrix_random = tc.randn(self.dz, self.dz)
        matrix_positive_normal = 1 / (self.dz * self.dz) * matrix_random @ matrix_random.T
        matrix = tc.eye(self.dz) + matrix_positive_normal
        max_ev = tc.max(tc.abs(tc.linalg.eigvals(matrix)))
        matrix_spectral_norm_one = matrix / max_ev
        return nn.Parameter(matrix_spectral_norm_one, requires_grad=True)

    def clip_z_to_range(self, z):
        if self.clip_range is not None:
            tc.clip_(z, -self.clip_range, self.clip_range)
        return z


class PLRNN_Step(Latent_Step):
    def __init__(self, *args, **kwargs):
        super(PLRNN_Step, self).__init__(*args, **kwargs)
        #self.AW = self.init_uniform((self.dz, self.dz))
        if not self.W_trial:
            self.A = nn.Parameter(tc.ones(self.dz)*0.95)
            self.W1 = self.init_uniform((self.dim_hidden, self.dz))
            self.W2 = self.init_uniform((self.dz, self.dim_hidden))
        else:
            self.A = nn.Parameter(tc.ones(self.dz)*0.95)
            self.W1 = self.init_uniform_same((self.trials, self.dim_hidden, self.dz))
            self.W2 = self.init_uniform_same((self.trials, self.dz, self.dim_hidden))

        self.h1 = nn.Parameter(tc.randn(self.dim_hidden))
        self.h2 = nn.Parameter(tc.randn(self.dz))
        self.C = self.init_uniform((self.dz, self.ds))
   
    def forward(self, z, s, A, W1, W2, h1, h2, C):
        if self.noise:
            noise = tc.normal(mean=tc.zeros(z.shape), std=tc.ones(z.shape)*self.noise).to(z.device) #!!!
        else:
            noise = tc.zeros(size=z.shape, device=z.device)
        if W1.dim() == 2:
            z_activated = tc.relu(self.norm(z @ W1.t() + h1))
            z = A * z + z_activated @ W2.t() + h2 + s @ C.t() + noise
        else:
            z_activated = tc.relu(self.norm(tc.einsum('ij,ijk->ik', z, tc.transpose(W1, 1, 2)) + h1))
            z = A * z + tc.einsum('ij,ijk->ik', z_activated, tc.transpose(W2, 1, 2)) + h2 + s @ C.t() + noise
        return z
    
class Clipped_shPLRNN_Step(Latent_Step):
    def __init__(self, *args, **kwargs):
        super(Clipped_shPLRNN_Step, self).__init__(*args, **kwargs)
        #self.AW = self.init_uniform((self.dz, self.dz))
        if not self.W_trial:
            self.A = nn.Parameter(tc.ones(self.dz)*0.95)
            self.W1 = self.init_uniform((self.dim_hidden, self.dz))
            self.W2 = self.init_uniform((self.dz, self.dim_hidden))
            #print('Using stationary version of clipped shPLRNN!')
        else:
            self.A = nn.Parameter(tc.ones(self.dz)*0.95)
            self.W1 = self.init_uniform_same((self.trials, self.dim_hidden, self.dz))
            self.W2 = self.init_uniform_same((self.trials, self.dz, self.dim_hidden))
            #print('Using non-stationary version of clipped shPLRNN!')
        
        
        if self.h_trial:
            self.h1 = self.init_uniform_same_h((self.trials, self.dim_hidden))
        else:
            self.h1 = nn.Parameter(tc.randn(self.dim_hidden))
        self.h2 = nn.Parameter(tc.randn(self.dz))
        self.C = self.init_uniform((self.dz, self.ds))

    @tc.no_grad()
    def jacobian(self, idx, z):
        W1 = self.W1[idx]
        W2 = self.W2[idx]
        W1z = z @ W1.t()
        return tc.diag(self.A) + W2 @ tc.diag((W1z > - self.h1).float() - (W1z > 0).float()) @ W1 #might throw an error

    def forward(self, z, s, A, W1, W2, h1, h2, C):
        if self.noise:
            noise = tc.normal(mean=tc.zeros(z.shape), std=tc.ones(z.shape)*self.noise).to(z.device) #!!!
        else:
            noise = tc.zeros(size=z.shape, device=z.device)
        if W1.dim() == 2:
            Wz = z @ W1.t()
            z_activated = tc.relu(self.norm(Wz + h1)) - tc.relu(self.norm(Wz))
            z = A * z + z_activated @ W2.t() + h2 + s @ C.t() + noise
        else:
            Wz = tc.einsum('ij,ijk->ik', z, tc.transpose(W1, 1, 2))
            z_activated = tc.relu(self.norm(Wz + h1)) - tc.relu(self.norm(Wz))
            z = A * z + tc.einsum('ij,ijk->ik', z_activated, tc.transpose(W2, 1, 2)) + h2 + s @ C.t() + noise
        return z
    




class PLRNN_Basis_Step(Latent_Step):
    def __init__(self, db, *args, **kwargs):
        super(PLRNN_Basis_Step, self).__init__(*args, **kwargs)
        #self.AW = self.init_uniform((self.dz, self.dz))
        #self.AW = self.init_AW()
        if self.W_trials == False:
            self.AW = self.init_AW()
        else:
            self.A = self.init_uniform((self.dz, ))
        #self.W =[self.init_W() for i in range(args.trials)]
            self.W = self.init_uniform((self.trials, self.dz, self.dz))

        self.h = self.init_uniform((self.dz, ))
        self.C = self.init_uniform((self.dz, self.ds))
        self.db = db
        self.thetas = self.init_uniform((self.dz, self.db))
        self.alphas = self.init_uniform((self.db, ))

    def forward(self, z, s, A, W, h, C, alphas, thetas):
        z_norm = self.norm(z).unsqueeze(-1)
        # thresholds are broadcasted into the added dimension of z
        be = tc.sum(alphas * tc.relu(z_norm + thetas), dim=-1)

        if W.dim() == 2:
           z = A * z + be @ W.t() + h + s @ C.t()
        else:
           z = A * z + tc.einsum('ij,ijk->ik', be, tc.transpose(W, 1, 2)) + h + s @ C.t()
        #z = A * z + be @ W.t() + h + s @ C.t()
        return self.clip_z_to_range(z)

class PLRNN_Clipping_Step(Latent_Step):
    def __init__(self, db, *args, **kwargs):
        super(PLRNN_Clipping_Step, self).__init__(*args, **kwargs)
        #self.AW = self.init_uniform((self.dz, self.dz))
        #self.AW = self.init_AW()
        if self.W_trials == False:
            self.AW = self.init_AW()
        else:
            self.A = self.init_uniform((self.dz, ))
        #self.W =[self.init_W() for i in range(args.trials)]
            self.W = self.init_uniform((self.trials, self.dz, self.dz))
        self.h = self.init_uniform((self.dz, ))
        self.C = self.init_uniform((self.dz, self.ds))
        self.db = db
        self.thetas = self.init_uniform((self.dz, self.db))
        self.alphas = self.init_uniform((self.db, ))

    def forward(self, z, s, A, W, h, C, alphas, thetas):
        z_norm = self.norm(z).unsqueeze(-1)
        be_clip = tc.sum(alphas * (tc.relu(z_norm + thetas) - tc.relu(z_norm)), dim=-1)
        if W.dim() == 2:
           z = A * z + be_clip @ W.t() + h + s @ C.t()
        else:
           z = A * z + tc.einsum('ij,ijk->ik', be_clip, tc.transpose(W, 1, 2)) + h + s @ C.t()
        #z = A * z + be_clip @ W.t() + h + s @ C.t()
        return z
