import torch as tc
from torch import optim
from torch import nn
from bptt import models
from bptt import regularization
from bptt import saving
from bptt.dataset import GeneralDataset
from bptt.dataset import TrialDataset
from bptt.dataset import TestDataset
from evaluation.correlations import autoencoder_reconstruction_correlation
import utils
import random
from tensorboardX import SummaryWriter
from argparse import Namespace
from timeit import default_timer as timer
import datetime
from math import ceil

class BPTT:
    """
    Train a model with (truncated) BPTT.
    """

    def __init__(self, args: Namespace, data_set: GeneralDataset, test_set: TestDataset,
                 writer: SummaryWriter, save_path: str, device: tc.device):
        # dataset, model, device, regularizer
        self.device = device
        self.data_set = data_set
        self.test_set = test_set
        #self.inputs_set = inputs_set
        
        self.model = models.Model(args, data_set)#, inputs_set)
        self.regularizer = regularization.Regularizer(args)
        self.to_device()

        # observation noise
        self.noise_level = args.observation_noise_level

        # loss weightings
        self.rec_w = args.reconstruction_loss_weight
        self.pred_w = args.prediction_loss_weight
        self.lat_w = args.latent_loss_weight

        # sae pretraining
        self.pretrain_epochs = args.pretrain_epochs
        self.only_AE = args.only_AE
        self.use_AE = False if isinstance(self.model.E, nn.Identity) else True

        # optimizers
        self.optimizer = optim.RAdam(self.model.latent_model.parameters(), args.learning_rate)
        if self.use_AE:
            AE_params = list(self.model.E.parameters()) + list(self.model.D.parameters())  
            self.optimizer_AE = optim.RAdam(AE_params, args.AE_learning_rate, weight_decay=args.AE_weight_decay)
        
        # others
        self.n_epochs = args.n_epochs
        self.gradient_clipping = args.gradient_clipping
        self.writer = writer
        self.use_reg = args.use_reg
        self.saver = saving.Saver(writer, save_path, args, self.data_set, self.test_set, self.regularizer)
        self.save_step = args.save_step
        self.alpha = args.TF_alpha
        self.loss_fn = nn.MSELoss()
        self.rec_loss_fn = nn.MSELoss()

        # scheduler
        e = args.n_epochs
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, [int(0.05 * e), int(0.1 * e), int(0.75 * e), int(0.9 * e)], 0.1)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, [int(0.8 * e)], 0.1)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=100, verbose=True) #changed
        if self.use_AE: self.scheduler_AE = optim.lr_scheduler.MultiStepLR(self.optimizer_AE, [int(0.8 * e)], 0.1)

    def standardize_tensor_3d(self, tensor):
        mean = tensor.mean(1).unsqueeze(1)
        std = tensor.std(1).unsqueeze(1)+1e-5
        return (tensor-mean)/std


    def to_device(self) -> None:
        '''
        Moves members to computing device.
        '''
        self.model.to(self.device)
        self.data_set.to(self.device)
        if self.test_set is not None:
            self.test_set.to(self.device)
        #self.inputs_set.to(self.device)
        self.regularizer.to(self.device)

    #def adaptive_observation_noise(self, X):
    #    X.std()

    def adaptive_gaussian_noise(self, X):
        std = X.std(1, keepdims=True)
        noise = tc.randn_like(X) * self.noise_level * std
        return X + noise

    def compute_loss(self, enc_z, lat_z, rec: tc.Tensor, pred: tc.Tensor, inp: tc.Tensor, target: tc.Tensor) -> tc.Tensor:
        loss = .0

        # AR convergence
        #loss += tc.sum(tc.relu(self.model.latent_model.get_latent_parameters()[0] - 1)**2)
        
        if self.use_AE:
            # reconstruction loss
            loss += self.rec_w * self.rec_loss_fn(rec, inp)
            # latent loss
            loss += self.lat_w  * self.loss_fn(enc_z[:, 1:], lat_z[:, :-1])
            # prediction loss
            loss += self.pred_w * self.loss_fn(pred, target)


            #mean loss
            # loss += (enc_z.mean())**2
            #var loss
            # loss += tc.mean((1 - enc_z.var(0))**2)
            # loss += tc.mean((1 - lat_z.var(0))**2)
        else:
            loss += self.loss_fn(pred, target)

        if self.use_reg:
            #indices = list(range(0,))
            indices = None
            lat_model_parameters = self.model.latent_model.get_latent_parameters(indices)
            loss += self.regularizer.loss(lat_model_parameters)

        return loss

    def pretrain_AE(self):
        """
        Perform a pretraining of the AE before joint training w/ dynamics.
        """

        E, D = self.model.E, self.model.D
        AE_p = list(E.parameters()) + list(D.parameters())
        print("Pretraining Autoencoder ...")
        for epoch in range(1, self.pretrain_epochs+1):
            # enter training mode
            self.model.train()

            # measure time
            T_start = timer()

            dataloader, _ = self.data_set.get_rand_dataloader()
            for _, (X, _, _) in enumerate(dataloader):
                self.optimizer_AE.zero_grad(set_to_none=True)

                # add gaussian noise to data, denoising AE
                X_flat = X.view(-1, X.size(-1))
                
                X_noisy = self.adaptive_gaussian_noise(X).view(-1, X.size(-1))

                loss = self.rec_loss_fn(X_flat, D(E(X_noisy)))

                # enc_z = E(X_noisy)

                #mean loss
                # loss += (enc_z.mean())**2
                #var loss
                # loss += tc.mean((1 - enc_z.var(0))**2)

                loss.backward()
                nn.utils.clip_grad_norm_(parameters=AE_p, max_norm=self.gradient_clipping)
                self.optimizer_AE.step()

            T_end = timer()
            if (epoch % self.save_step == 0) or (epoch == 1):
                X = self.data_set.data
                corr = autoencoder_reconstruction_correlation(X, E, D).mean()
                print(f"(Pretraining) Epoch {epoch} took {round(T_end-T_start, 2)}s -----> Loss = {loss.item()}")
                print(f"AE CORR: {corr}")

    def train(self):
        alpha = self.alpha

        if self.pretrain_epochs > 0 and self.use_AE:
            self.pretrain_AE()

        if self.only_AE:
            print("Only Autoencoder training mode, random PLRNN will initialized but training stops after one step")
            self.n_epochs = 1
            self.save_step = 1
            

        for epoch in range(1, self.n_epochs + 1):
            # enter training mode
            self.model.train()

            # measure time
            T_start = timer()

            #dataloader = self.data_set.get_rand_dataloader()
            dataloader, indices = self.data_set.get_rand_dataloader()

            #self.optimizer.zero_grad(set_to_none=True)
            #loss = .0
            for idx, (inp, target, ext_inputs) in enumerate(dataloader):

                self.optimizer.zero_grad(set_to_none=True)

                if self.use_AE: self.optimizer_AE.zero_grad(set_to_none=True)

                # add gaussian noise to input
                inp_noise = self.adaptive_gaussian_noise(inp)

                # denoising AE
                enc_z, lat_z, rec, pred = self.model([indices[idx], inp_noise], ext_inputs, alpha)
                loss = self.compute_loss(enc_z, lat_z, rec, pred, inp, target)
                

                loss.backward()
                nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.gradient_clipping)
                self.optimizer.step()
                if (self.use_AE) & (not self.only_AE):  self.optimizer_AE.step()

            self.scheduler.step()
            if (self.use_AE) & (not self.only_AE):  self.scheduler_AE.step()

            T_end = timer()
            print(f"Epoch {epoch} took {round(T_end-T_start, 2)}s -----> Loss = {loss.item()}")

            if (epoch % self.save_step == 0) or (epoch == 1):
                self.saver.epoch_save(self.model, self.regularizer, epoch)

    
    def train_Annealing(self):
        alpha = self.alpha
        print('Anealing loop with four steps')

        #First Annealing step only observation model training
        if self.pretrain_epochs > 0 and self.use_AE:
            self.pretrain_AE()

        X = self.data_set.data
        E, D = self.model.E, self.model.D
        corr = autoencoder_reconstruction_correlation(X, E, D).mean() 
        print(f"AE CORR after pre-training: {corr}")

        #Annealing epochs
        ann_epochs = int(ceil(self.n_epochs / 3))

        #Second Annealing step fix encoder and decoder only train RNN
        self.model.fix_observation_model = 1
        self.rec_w = 0
        self.lat_w = 1
        self.pred_w = 1
        print('Anealing step 2, observation model fixed')
        for epoch in range(1, ann_epochs):
            # enter training mode
            self.model.train()

            # measure time
            T_start = timer()

            #dataloader = self.data_set.get_rand_dataloader()
            dataloader, indices = self.data_set.get_rand_dataloader()

            #self.optimizer.zero_grad(set_to_none=True)
            #loss = .0
            for idx, (inp, target, ext_inputs) in enumerate(dataloader):

                self.optimizer.zero_grad(set_to_none=True)

                if self.use_AE: self.optimizer_AE.zero_grad(set_to_none=True)

                # add gaussian noise to input
                inp_noise = self.adaptive_gaussian_noise(inp)

                # denoising AE
                enc_z, lat_z, rec, pred = self.model([indices[idx], inp_noise], ext_inputs, alpha)
                loss = self.compute_loss(enc_z, lat_z, rec, pred, inp, target)
                

                loss.backward()
                nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.gradient_clipping)
                self.optimizer.step()
                # if (self.use_AE) & (not self.only_AE):  self.optimizer_AE.step()

            self.scheduler.step()
            # if (self.use_AE) & (not self.only_AE):  self.scheduler_AE.step()

            T_end = timer()
            print(f"Epoch {epoch} took {round(T_end-T_start, 2)}s -----> Loss = {loss.item()}")

            if (epoch % self.save_step == 0) or (epoch == 1):
                self.saver.epoch_save(self.model, self.regularizer, epoch)

        #Third Annealing step no fixed observation model but all data loss weights increased by 10
        self.model.fix_observation_model = 0
        self.rec_w = 10
        self.lat_w = 1
        self.pred_w = 1
        print('Anealing step 3, observation model free again but rec-loss, lat-loss and pred_loss = 10')
        for epoch in range(ann_epochs, ann_epochs*2+1):
            # enter training mode
            self.model.train()

            # measure time
            T_start = timer()

            #dataloader = self.data_set.get_rand_dataloader()
            dataloader, indices = self.data_set.get_rand_dataloader()

            #self.optimizer.zero_grad(set_to_none=True)
            #loss = .0
            for idx, (inp, target, ext_inputs) in enumerate(dataloader):

                self.optimizer.zero_grad(set_to_none=True)

                if self.use_AE: self.optimizer_AE.zero_grad(set_to_none=True)

                # add gaussian noise to input
                inp_noise = self.adaptive_gaussian_noise(inp)

                # denoising AE
                enc_z, lat_z, rec, pred = self.model([indices[idx], inp_noise], ext_inputs, alpha)
                loss = 0.1*self.compute_loss(enc_z, lat_z, rec, pred, inp, target)
                

                loss.backward()
                nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.gradient_clipping)
                self.optimizer.step()
                if (self.use_AE) & (not self.only_AE):  self.optimizer_AE.step()

            self.scheduler.step()
            if (self.use_AE) & (not self.only_AE):  self.scheduler_AE.step()

            T_end = timer()
            print(f"Epoch {epoch} took {round(T_end-T_start, 2)}s -----> Loss = {loss.item()}")

            if epoch % self.save_step == 0:
                self.saver.epoch_save(self.model, self.regularizer, epoch)

        #Fourth Annealing step no fixed observation model but all data loss weights increased by 100
        self.model.fix_observation_model = 0
        self.rec_w = 100
        self.lat_w = 10
        self.pred_w = 10
        print('Anealing step 4, observation model free again but rec-loss, lat-loss and pred_loss = 100')
        for epoch in range(ann_epochs*2, ann_epochs*3+1):
            # enter training mode
            self.model.train()

            # measure time
            T_start = timer()

            #dataloader = self.data_set.get_rand_dataloader()
            dataloader, indices = self.data_set.get_rand_dataloader()

            #self.optimizer.zero_grad(set_to_none=True)
            #loss = .0
            for idx, (inp, target, ext_inputs) in enumerate(dataloader):

                self.optimizer.zero_grad(set_to_none=True)

                if self.use_AE: self.optimizer_AE.zero_grad(set_to_none=True)

                # add gaussian noise to input
                inp_noise = self.adaptive_gaussian_noise(inp)

                # denoising AE
                enc_z, lat_z, rec, pred = self.model([indices[idx], inp_noise], ext_inputs, alpha)
                loss = 0.01*self.compute_loss(enc_z, lat_z, rec, pred, inp, target)
                

                loss.backward()
                nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.gradient_clipping)
                self.optimizer.step()
                if (self.use_AE) & (not self.only_AE):  self.optimizer_AE.step()

            self.scheduler.step()
            if (self.use_AE) & (not self.only_AE):  self.scheduler_AE.step()

            T_end = timer()
            print(f"Epoch {epoch} took {round(T_end-T_start, 2)}s -----> Loss = {loss.item()}")

            if epoch % self.save_step == 0:
                self.saver.epoch_save(self.model, self.regularizer, epoch)

