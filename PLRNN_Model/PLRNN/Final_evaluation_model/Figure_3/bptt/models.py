import os
import torch as tc
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import utils
from bptt import saving
from bptt import PLRNN_model
import random
from typing import Optional
from sae.encoder import *
from sae.decoder import *




def load_args(model_path):
    args_path = os.path.join(model_path, 'hypers.pkl').replace('\\','/')
    args = np.load(args_path, allow_pickle=True)
    return args

class Model(nn.Module):
    def __init__(self, args=None, data_set=None):
        super().__init__()
        self.latent_model = None
        self.device = None
        self.args = args
        self.data_set = data_set
        self.fix_observation_model = 0
        #self.inputs_set = inputs_set

        if args is not None:
            # cast args to dictionary
            self.args = vars(args)
            self.init_from_args()

    def forward(self, x, s, alpha=None):
        '''
        Whole SAE forward pass.
        '''
        indices, X = x
        S, T, dx = X.size()

        if self.fix_observation_model:
            # Encoder
            with tc.no_grad():
                Z_enc = self.E(X.view(-1, dx))
                _, dEnc = Z_enc.size()

                # reconstructed x
                X_rec = self.D(Z_enc).view(S, T, dx)

            
            # z0 for plrnn initial cond.
            z0 = None
            if self.z0_model:
                # this is only needed if the latent model has more dynamical variables
                # than the encoder provides
                z0 = self.z0_model(Z_enc.view(S, T, -1)[:, 0, :])

            # Z is of shape S x T x dz
            Z = self.latent_model([indices, Z_enc.view(S, T, -1)], s, alpha, z0)
            Z_enc_pred = Z[:, :, :dEnc] ##mapping to observation space ?

            # map forward stepped latent dynamics to obs. space
            with tc.no_grad():
                X_pred = self.D(Z_enc_pred.reshape(-1, dEnc)).view(S, T, dx)

        else:
            # Encoder
            Z_enc = self.E(X.view(-1, dx))
            _, dEnc = Z_enc.size()

            # reconstructed x
            X_rec = self.D(Z_enc).view(S, T, dx)

            
            # z0 for plrnn initial cond.
            z0 = None
            if self.z0_model:
                # this is only needed if the latent model has more dynamical variables
                # than the encoder provides
                z0 = self.z0_model(Z_enc.view(S, T, -1)[:, 0, :])

            # Z is of shape S x T x dz
            Z = self.latent_model([indices, Z_enc.view(S, T, -1)], s, alpha, z0)
            Z_enc_pred = Z[:, :, :dEnc] ##mapping to observation space ?

            # map forward stepped latent dynamics to obs. space
            X_pred = self.D(Z_enc_pred.reshape(-1, dEnc)).view(S, T, dx)

        
        return Z_enc.view(S, T, -1), Z_enc_pred, X_rec, X_pred
    

    def to(self, device: tc.device):
        self.device = device
        return super().to(device)

    def get_latent_parameters(self, indices=None):
        '''
        Return a list of all latent model parameters:
        A: (dz, )
        W: (dz, dz)
        h: (dz, )

        For BE-models additionally:
        alpha (db, )
        thetas (dz, db)
        '''
        return self.latent_model.get_parameters(indices)

    def get_num_trainable(self):
        '''
        Return the number of total trainable parameters
        '''
        return sum([p.numel() if p.requires_grad else 0 
                    for p in self.parameters()])

    def init_from_args(self):
        # resume from checkpoint?
        model_path = self.args['load_model_path']
        if model_path is not None:
            epoch = None
            if self.args['resume_epoch'] is None:
                epoch = utils.infer_latest_epoch(model_path)
            self.init_from_model_path(model_path, epoch)
        else:
            self.init_submodules()

    def init_from_model_path(self, model_path, epoch=None):
        # load arguments
        self.args = load_args(model_path)
        if 'process_noise_level' not in self.args.keys():
            self.args['process_noise_level'] = 0
        
        # infer input dim from args
        S = np.load(self.args["inputs_path"], allow_pickle=True)
        self.args["dim_s"] = S[0].shape[-1]

        # init using arguments
        self.init_submodules()

        # restore model parameters
        self.load_state_dict(self.load_statedict(model_path, 'model', epoch=epoch))

    def init_submodules(self):
        '''
        Initialize latent model, output layer and z0 model.
        '''
        # TODO: Add RNN/LSTM as separate models w/o explicit latent model..
        dx, dz, dEnc = self.args['dim_x'], self.args['dim_z'], self.args['enc_dim']
        dh = self.args['dim_hidden']
        if 'h_trial' not in self.args.keys():
            self.args['h_trial'] = 0
        self.latent_model = PLRNN_model.PLRNN(dx, dz, self.args['dim_s'], dh, dEnc, self.args['n_trials'], self.args['W_trial'],
                                              self.args['n_bases'],
                                              latent_model=self.args['latent_model'],
                                              clip_range=self.args['clip_range'],
                                              mean_center=self.args['mean_center'],
                                              process_noise=self.args['process_noise_level'],
                                              h_trial=self.args['h_trial'],
                                              pm_noise=self.args['pm_noise'])

        #self.output_layer = self.init_obs_model(self.args['fix_obs_model'])
        self.z0_model = self.init_z0_model(self.args['learn_z0'])

        # SAE
        self.initialize_autoencoder()

    def initialize_autoencoder(self):
        dx, dEnc = self.args['dim_x'], self.args['enc_dim']
        n_layers = self.args['n_layers']
        activation_str = self.args['activation_fn']

        activation_functions = {
                                "ReLU": nn.ReLU,
                                "LeakyReLU": nn.LeakyReLU,
                                "PReLU": nn.PReLU,
                                "RReLU": nn.RReLU,
                                "ELU": nn.ELU,
                                "SELU": nn.SELU,
                                "CELU": nn.CELU,
                                "GELU": nn.GELU,
                                "Sigmoid": nn.Sigmoid,
                                "Tanh": nn.Tanh,
                                "Softmax": nn.Softmax,
                                "LogSoftmax": nn.LogSoftmax,
                                "Softplus": nn.Softplus,
                                "Softmin": nn.Softmin,
                                "Softshrink": nn.Softshrink,
                                "Softsign": nn.Softsign,
                                "Tanhshrink": nn.Tanhshrink,
                                "Threshold": nn.Threshold,
                            }
        if self.args["autoencoder_type"] == "Custom":
            activation_function = activation_functions[activation_str]()
        else:
            activation_function = activation_functions[activation_str]
            #print(activation_function)

        # cases
        if self.args["autoencoder_type"] == "Identity":
            assert dx == dEnc, "Observation dimension has to \
                has to equal encoder dimension when using Identity Encoder/Decoder!"
            self.E = self.D = nn.Identity()
        elif self.args["autoencoder_type"] == "Linear":
            self.E = nn.Linear(dx, dEnc)
            self.D = nn.Linear(dEnc, dx)
        elif self.args["autoencoder_type"] == "MLP":
            self.E = EncoderMLP(dx, dEnc, nn.LayerNorm, activation_function)
            self.D = DecoderMLP(dEnc, dx, nn.LayerNorm, activation_function)
            #print('self.E')
        elif self.args["autoencoder_type"] == "Custom":
            self.E = Encoder_Custom(dx, dEnc, activation_function, n_layers)
            self.D = Decoder_Custom(dEnc, dx, activation_function, n_layers)

    def return_Encoder_Decoder(self):
        return (self.E, self.D)

    def init_z0_model(self, learn_z0: bool):
        z0model = None
        if learn_z0:
            z0model = Z0Model(self.args['enc_dim'], self.args['dim_z'])
        return z0model

    def load_statedict(self, model_path, model_name, epoch=None):
        if epoch is None:
            epoch = self.args['n_epochs']
        path = os.path.join(model_path, '{}_{}.pt'.format(model_name, str(epoch)))
        state_dict = tc.load(path,map_location='cuda:0')
        return state_dict

    @tc.no_grad()
    def generate_free_trajectory(self, data: tc.Tensor, inputs: tc.Tensor, T: int,
                                 trial_index: int, z0: tc.Tensor = None, noise=0):
        # encode first time point as initial condition
        z0_enc = self.E(data[[0]])
        _, dEnc = z0_enc.size()

        # optionally predict an initial z0 of shape (1, dz)
        if self.z0_model:
            z0 = self.z0_model(z0_enc)
        else:
            z0 = z0_enc

        # latent traj is T x dz
        latent_traj = self.latent_model.generate(T, inputs, z0, trial_index, noise)
        obs_traj = self.D(latent_traj[:, :dEnc])
        # T x dx, T x dz
        return obs_traj, latent_traj
    ### CRISTIAN CODE
    @tc.no_grad()
    def generate_test_trajectory(self, data: tc.Tensor, W2_par: tc.Tensor,W1_par: tc.Tensor,
                                 inputs: tc.Tensor, T: int, trial_index: int, z0: tc.Tensor = None):
        # encode first time point as initial condition
        z0_enc = self.E(data[[0]])
        _, dEnc = z0_enc.size()

        # optionally predict an initial z0 of shape (1, dz)
        if self.z0_model:
            z0 = self.z0_model(z0_enc)
        else:
            z0 = z0_enc

        # latent traj is T x dz
        latent_traj = self.latent_model.generate_test(T, inputs, z0, W2_par,W1_par,trial_index)
        obs_traj = self.D(latent_traj[:, :dEnc])
        # T x dx, T x dz
        return obs_traj, latent_traj

    ###
    @tc.no_grad()
    def plot_simulated(self, data: tc.Tensor, inputs: tc.Tensor, T: int, indices):
        X, Z = self.generate_free_trajectory(data, inputs, T, indices)
        fig = plt.figure()
        plt.title('simulated')
        plt.axis('off')
        plot_list = [X, Z]
        names = ['x', 'z']
        for i, x in enumerate(plot_list):
            fig.add_subplot(len(plot_list), 1, i + 1)
            plt.plot(x.cpu())
            plt.title(names[i])
        plt.xlabel('time steps')
    
    @tc.no_grad()
    def plot_obs_simulated(self, data: tc.Tensor, inputs: tc.Tensor, indices):
        time_steps = len(data)
        X, Z = self.generate_free_trajectory(data, inputs, time_steps, indices)
        fig = plt.figure()
        #fig = plt.figure(figsize=(18, 3))
        #plt.figure(figsize=(18, 3))
        plt.title('observations')
        plt.axis('off')
        n_units = data.shape[1]
        max_units = min([n_units, 10])
        max_time_steps = 1000
        for i in range(max_units):
            fig.add_subplot(max_units, 1, i + 1)
            #plt.figure(figsize=(18, 3))
            plt.plot(data[:max_time_steps, i].cpu())
            ax = plt.gca()
            lim = ax.get_ylim()
            plt.plot(X[:max_time_steps, i].cpu())
            ax.set_ylim(lim)
            #ax.set_ylim(-15,15)
        plt.legend(['data', 'x simulated'])
        plt.xlabel('time steps')

    @tc.no_grad()
    def plot_latent_vs_encoded(self, data: tc.Tensor, inputs: tc.Tensor,
                               rand_seq: Optional[bool] = False):
        '''
        Plot the overlap of latent, teacher forced trajectory
        with the latent trajectory inferred by the encoder.
        '''
        #T = self.args['seq_len']
        alpha = self.args['TF_alpha']
        dz = self.args['dim_z']
        dEnc = self.args['enc_dim']
        T_full, _ = data.size()
        T = T_full - 1
        max_units = min([dEnc, 10])

        # input and model prediction
        if rand_seq:
            t = random.randint(0, T_full - T)
        else:
            t = 0
        input_ = data[t: t + T]
        ext_inputs = inputs[t+1: t + T+1]
        enc_z, z, _, _ = self([[0], input_.unsqueeze(0)], ext_inputs.unsqueeze(0), alpha)
        enc_z.squeeze_(0)
        z.squeeze_(0)

        # z_ = tc.cat([enc_z[[0]], z], 0)
        # print(enc_z.size(), z.size())

        # x axis
        x = np.arange(T_full - 1)

        # plot
        fig = plt.figure()
        plt.title('Latent trajectory')
        plt.axis('off')
        for i in range(max_units):
            fig.add_subplot(max_units, 1, i + 1)
            plt.plot(x[:-1], enc_z[1:, i].cpu(), label='Encoded z', color="tab:blue")
            ax = plt.gca()
            lim = ax.get_ylim()
            plt.plot(x[:-1], z[:-1, i].cpu(), label='Generated z', color="tab:orange")
            ax.set_ylim(lim)
            #plt.scatter(x[::N], enc_z[::N, i].cpu(), marker='2',
                        #label='TF', color='r')
            ax.set_ylim(lim)
            plt.legend(prop={"size": 5})
            plt.ylabel(f'$x_{i}$')
        plt.xlabel('t')

    @tc.no_grad()
    def plot_reconstruction(self, data: tc.Tensor, inputs: tc.Tensor,
                            rand_seq: Optional[bool] = False):
        '''
        Plot reconstruction of the input sequence
        passed through the Autoencoder.
        '''
        #T = self.args['seq_len']
        T_full, dx = data.size()
        T = T_full-1
        max_units = min([dx, 10])

        # input and model prediction
        if rand_seq:
            t = random.randint(0, T_full - T)
        else:
            t = 0
        input_ = data[t: t + T]
        ext_inputs = inputs[t+1: t + T+1]
        _, _, rec, _ = self([[0], input_.unsqueeze(0)], ext_inputs.unsqueeze(0))
        rec.squeeze_(0)

        # x axis
        x = np.arange(T)

        # plot
        fig = plt.figure()
        plt.title('Reconstruction')
        plt.axis('off')
        for i in range(max_units):
            fig.add_subplot(max_units, 1, i + 1)
            plt.plot(x, input_[:, i].cpu(), label='GT')
            ax = plt.gca()
            lim = ax.get_ylim()
            plt.plot(x, rec[:, i].cpu(), label='Pred')
            ax.set_ylim(lim)
            plt.legend(prop={"size": 5})
            plt.ylabel(f'$x_{i}$')
        plt.xlabel('t')

    @tc.no_grad()
    def plot_generated_latent_space_3d(self, data: tc.Tensor, inputs: tc.Tensor):
        '''
        Plot reconstruction of the input sequence
        passed through the Autoencoder.
        '''
        #T = self.args['seq_len']
        T_full, dx = data.size()
        #T = T_full
        max_units = min([dx, 10])

        X, Z = self.generate_free_trajectory(data, inputs, T_full, [-1])

        Z = Z.cpu().numpy()
        Z_enc = self.E(data).cpu().numpy()

        to_T = 10000
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(Z_enc[:to_T, 0], Z_enc[:to_T, 1], Z_enc[:to_T, 2], label="GT")
        ax.plot3D(Z[:to_T, 0], Z[:to_T, 1], Z[:to_T, 2], label="Generated")
        plt.legend()
        # plt.tight_layout()



    @tc.no_grad()
    def plot_prediction(self, data: tc.Tensor, inputs: tc.Tensor, indices,
                        rand_seq: Optional[bool] = False):
        '''
        Plot prediction of the model for a given
        input sequence with teacher forcing (interleaved
        observations)
        '''
        if data.shape[0] < self.args['seq_len']:
            T = data.shape[0]-1
        else:
            T = self.args['seq_len']
        #T = 81
        alpha = self.args['TF_alpha']
        T_full, dx = data.size()
        max_units = min([dx, 10])

        # input and model prediction
        if rand_seq:
            t = random.randint(0, T_full - T) ##anpassen für s falls wieder in Benutzung
        else:
            t = 0
        input_ = data[t: t + T]
        s = inputs[t+1: t+1 + T]
        #pred = self([indices, input_.unsqueeze(0)], s.unsqueeze(0), N).squeeze(0)
        _, _, _, pred = self([indices, input_.unsqueeze(0).to(self.device)], s.unsqueeze(0), alpha)
        pred.squeeze_(0).cpu()

        # x axis
        x = np.arange(T-1)

        # plot
        fig = plt.figure()
        plt.title('Prediction')
        plt.axis('off')
        for i in range(max_units):
            fig.add_subplot(max_units, 1, i + 1)
            plt.plot(x, input_[1:, i].cpu(), label='GT')
            ax = plt.gca()
            lim = ax.get_ylim()
            plt.plot(x, pred[:-1, i].cpu(), label='Pred')
            ax.set_ylim(lim)
            #plt.scatter(x[::N], input_[1::N, i].cpu(), marker='2',
                        #label='TF-obs', color='r')
            ax.set_ylim(lim)
            plt.legend(prop={"size": 5})
            plt.ylabel(f'$x_{i}$')
        plt.xlabel('t')


class Z0Model(nn.Module):
    '''
    MLP that predicts an optimal initial latent state z0 given
    an inital observation x0.

    Takes x0 of dimension dx and returns z0 of dimension dz, by
    predicting dz-dx states and then concatenating x0 and the prediction:
    z0 = [x0, MLP(x0)]
    '''
    def __init__(self, dx: int, dz: int):
        super(Z0Model, self).__init__()
        # TODO: MLP currently only affine transformation
        # maybe try non-linear, deep variants?
        self.MLP = nn.Linear(dx, dz - dx, bias=False)
        #self.MLP = nn.Linear(dx, dz, bias=False)

    def forward(self, x0):
        return tc.cat([x0, self.MLP(x0)], dim=1) #self.MLP(x0)
