import torch as tc
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
from tensorboardX import utils as tb_utils
import seaborn as sns
from evaluation.correlations import autoencoder_reconstruction_correlation, generative_reconstruction_correlation, comp_W_corr
import main_eval
import utils


class Saver:
    def __init__(self, writer, save_path, args, data_set, test_set, regularizer):
        self.writer = writer
        self.save_path = save_path
        self.args = args
        self.data_set = data_set
        self.test_set = test_set
        #self.inputs_set = inputs_set
        self.model = None
        self.current_epoch = None
        self.current_model = None
        self.regularizer = regularizer
        self.initial_save()

    def initial_save(self):
        if self.args.use_tb:
            self.save_dataset()

    def save_dataset(self):
        #dataset_snippet = self.data_set.data[:1000].cpu().numpy()
        dataset_snippet = [t[:1000].cpu().numpy() for t in self.data_set.data]
        #inputset_snippet = self.data_set.inputs[:1000].cpu().numpy()
        inputset_snippet = [t[:1000].cpu().numpy() for t in self.data_set.inputs]
        #dataset_snippet, inputset_snippet = self.data_set.data
        plt.plot(dataset_snippet[0])
        plt.title('Observations')
        plt.xlabel('time steps')
        figure = plt.gcf()
        save_figure_to_tb(figure, self.writer, text='data set', global_step=None)
        plt.close()
        plt.plot(inputset_snippet[0])
        plt.title('Inputs')
        plt.xlabel('time steps')
        figure = plt.gcf()
        save_figure_to_tb(figure, self.writer, text='input set', global_step=None)
        plt.close()

    @tc.no_grad()
    def epoch_save(self, model, regularizer, epoch, scalars=None):
        # update members
        self.current_epoch = epoch
        self.current_model = model
        alpha, lr_PLRNN, lr_AE = scalars
        
        # switch to evaluation mode
        self.current_model.eval()

        if self.args.use_tb:
            with tc.no_grad():
                self.save_loss_terms()
                self.save_metrics()

                # add alpha decay
                self.writer.add_scalar(tag='aGTF alpha decay', scalar_value=alpha, global_step=self.current_epoch)
                self.writer.add_scalar(tag='lr PLRNN decay', scalar_value=lr_PLRNN, global_step=self.current_epoch)
                self.writer.add_scalar(tag='lr AE decay', scalar_value=lr_AE, global_step=self.current_epoch)


                # save plots and params w/ different step
                s = self.args.save_img_step
                if (self.current_epoch % s == 0):
                    self.save_prediction()

                    self.save_reconstruction()
                    self.save_generated_latent_space_3d()
                    self.save_latent_vs_encoded()

                    self.save_simulated()
                    #self.save_parameters()
                    self.save_grad_flow(self.current_model.named_parameters())
                    self.save_W1_corr(self.current_model.named_parameters())
                    self.save_W2_corr(self.current_model.named_parameters())
                    if self.args.h_trial:
                        self.save_h1_vals(self.current_model.named_parameters())

                    # save state dict (parameters)
                    #tc.save(self.current_model.state_dict(),
                            #os.path.join(self.save_path,
                            #f'model_{epoch}.pt'))
                    # save state dict (parameters)
                    path = os.path.join(self.save_path,
                                        f'model_{epoch}.pt')
                    while (not os.path.exists(path)):
                        tc.save(self.current_model.state_dict(),
                                path)

    def save_loss_terms(self):
        # regularization
        self.save_regularization_loss()
        # training losses
        self.compute_losses_across_all_trials()

        if self.test_set is not None:
            test_input, test_target, test_ext_inp = self.test_set[0]

        # latent model parameters
        latent_model = self.current_model.latent_model
        latent_model_parameters = latent_model.get_latent_parameters()

        if self.args.test:
            last_trial = self.args.n_trials-1
            enc_z_, z_, rec_, test_pred = self.current_model([[last_trial], test_input.unsqueeze_(0)], test_ext_inp.unsqueeze_(0), self.args.n_interleave)
            test_loss = nn.functional.mse_loss(test_pred, test_target.unsqueeze_(0)) #### why functional ?
            self.writer.add_scalar(tag='Test_Loss', scalar_value=test_loss, global_step=self.current_epoch)

        # A norm
        A, W1, W2, h1, h2, C = latent_model_parameters
        max_eig_val = tc.max(A)
        self.writer.add_scalar(tag='max(eig(A))', scalar_value=max_eig_val, global_step=self.current_epoch)

        # Keep in mind: We clip the gradients from the last backward pass of the training loop at
        # current epoch here, which are already clipped during training
        # so this line has the sole purpose of getting the total_norm from the last gradients
        total_norm = nn.utils.clip_grad_norm_(self.current_model.parameters(),
                                              self.args.gradient_clipping)
        self.writer.add_scalar(tag='total_grad_norm', scalar_value=total_norm, global_step=self.current_epoch)

        # autoencoder performance
        X = self.data_set.data
        if self.args.autoencoder_type != "Identity":
            corrs = autoencoder_reconstruction_correlation(X, self.current_model.E, self.current_model.D)
            self.writer.add_scalar(tag='AE_reconstruction_correlation', scalar_value=corrs.mean(), global_step=self.current_epoch)

        # generative model performance
        corrs = generative_reconstruction_correlation(X, self.data_set.inputs,
                                                      self.current_model)
        self.save_gen_rec_corr(corrs)

        # mean corr
        self.writer.add_scalar(tag='Mean correlation across trials', scalar_value=corrs.mean(), global_step=self.current_epoch)

    def compute_losses_across_all_trials(self):
        rec_loss = 0.
        pred_loss = 0.
        lat_loss = 0.

        ntr = len(self.data_set)
        for i in range(ntr):
            data = self.data_set.data[i]
            x = data[:-1, :].unsqueeze(0)
            y = data[1:, :].unsqueeze(0)
            inputs = self.data_set.inputs[i]
            s = inputs[1:, :].unsqueeze(0)
        
            enc_z, z, rec, pred = self.current_model([[i], x], s, self.args.TF_alpha)

            pred_loss += nn.functional.mse_loss(pred, y)
            if self.args.autoencoder_type != "Identity":
                lat_loss += nn.functional.mse_loss(z[:, :-1], enc_z[:, 1:])
                rec_loss += nn.functional.mse_loss(rec, x)
        
        rec_loss /= ntr
        pred_loss /= ntr
        lat_loss /= ntr

        self.writer.add_scalar(tag='Sum losses', scalar_value=pred_loss + lat_loss + rec_loss,
                                global_step=self.current_epoch)
        self.writer.add_scalar(tag='Prediction loss', scalar_value=pred_loss, global_step=self.current_epoch)
        self.writer.add_scalar(tag='Reconstruction Loss', scalar_value=rec_loss, global_step=self.current_epoch)
        self.writer.add_scalar(tag='Latent Loss', scalar_value=lat_loss, global_step=self.current_epoch)

    def save_gen_rec_corr(self, corrs):
        '''
        Save a reconstruction plot to tensorboard.
        '''
        fig = plt.figure()
        plt.title('Generated trial correlations')
        plt.plot(list(range(len(corrs))), tc.relu(corrs), ls="--", marker=".")
        plt.ylim(0, 1)
        plt.xlabel("Trials")
        plt.ylabel("Pearson correlation")
        save_plot_to_tb(self.writer, text='Generated trial correlations',
                        global_step=self.current_epoch)
        plt.close()


    def save_regularization_loss(self):
        reg = self.regularizer
        A, W1, W2, h1, h2, C = self.current_model.get_latent_parameters()
        for i, W in enumerate([W1, W2], 1):

            W_hat = W
            l2_reg = tc.norm(W_hat.flatten(), p=2)
            self.writer.add_scalar(tag=f'W{i}_L2_Norm', scalar_value=l2_reg, global_step=self.current_epoch)

            if W1.dim() == 3:
                n = W.size(0)
                indices1 = [0] + list(range(n-1))
                W1_hat = W[indices1]
                indices2 = [0, 1] + list(range(n-2))
                W2_hat = W[indices2]
                term_1st_order = W_hat - W1_hat
                loss_1st_order = tc.norm(term_1st_order.flatten(), p=2)
                term_2nd_order = W_hat - 2*W1_hat + W2_hat
                term_2nd_order[1, :, :] = 0
                loss_2nd_order = tc.norm(term_2nd_order.flatten(), p=2)

                self.writer.add_scalar(tag=f'W{i}_1st_order', scalar_value=loss_1st_order, global_step=self.current_epoch)
                self.writer.add_scalar(tag=f'W{i}_2nd_order', scalar_value=loss_2nd_order, global_step=self.current_epoch)

        reg_loss_total = reg.loss((A, W1, W2, h1, h2, C))
        self.writer.add_scalar(tag='Total_reg_loss', scalar_value=reg_loss_total, global_step=self.current_epoch)

    def save_grad_flow(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n) and (p.grad is not None):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu())
                max_grads.append(p.grad.abs().max().cpu())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        # plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        figure = plt.gcf()
        save_figure_to_tb(figure, self.writer, text='gradient flow ',
                          global_step=self.current_epoch)
        plt.close()

    def sort_and_rearrange(self, array_to_sort, array_to_rearrange):
        sorted_indices = np.argsort(array_to_sort)
        sorted_array = np.sort(array_to_sort)
        rearranged_array = array_to_rearrange[sorted_indices]
        return rearranged_array

    def sort_trace(self, w):
        ref = np.linspace(-1, 1, w.shape[1])
        corrs = np.zeros((w.shape[0]))
        for i, trace in enumerate(w):
            corrs[i] = np.corrcoef(trace, ref)[0,1]
        sorted_w = self.sort_and_rearrange(corrs, w)
        return sorted_w

    def save_h1_vals(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        par_dict = {**dict(self.current_model.state_dict())}
        h1 = par_dict['latent_model.latent_step.h1'].cpu().detach().numpy().T
        
        h1 = self.sort_trace(h1)
        h1 = (h1 - h1.mean(1).reshape(-1,1))/h1.std(1).reshape(-1,1)
        sns.heatmap(h1, cmap='coolwarm')
        figure = plt.gcf()
        save_figure_to_tb(figure, self.writer, text='h2 vals',
                          global_step=self.current_epoch)
        plt.close()

    def save_W1_corr(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        par_dict = {**dict(self.current_model.state_dict())}
        W1 = par_dict['latent_model.latent_step.W1']
        
        wcorr = comp_W_corr(W1)
        sns.heatmap(wcorr, vmin=-1, vmax=1, cmap='coolwarm')
        figure = plt.gcf()
        save_figure_to_tb(figure, self.writer, text='W1 correlation',
                          global_step=self.current_epoch)
        plt.close()

    def save_W2_corr(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        par_dict = {**dict(self.current_model.state_dict())}
        W2 = par_dict['latent_model.latent_step.W2']
        
        wcorr = comp_W_corr(W2)
        sns.heatmap(wcorr, vmin=-1, vmax=1, cmap='coolwarm')
        figure = plt.gcf()
        save_figure_to_tb(figure, self.writer, text='W2 correlation',
                          global_step=self.current_epoch)
        plt.close()



    def save_metrics(self):
        """Evaluate metrics on a subset of the training data, then save them to tensorboard"""
        import main_eval
        main_eval.DATA_GENERATED = None
        for metric in self.args.metrics:
            data_batch = utils.read_data(self.args.data_path)
            inputs_batch = utils.read_data(self.args.inputs_path)
            inputs_subset = inputs_batch[:-1]#[:1000]
            data_subset = data_batch[:-1]#[:1000]
            #test_data = data_batch[-1]
            #test_inputs = inputs_batch[-1]

            metric_value = main_eval.eval_model_on_data_with_metric(model=self.current_model, data=data_subset,
                                                                    inputs=inputs_subset, metric=metric, indices= [0])
            metric_value = metric_value[0]  # only take first metric value, e.g. mse 1 step ahead, and klz mc
            tag = 'metric_{}'.format(metric)
            self.writer.add_scalar(tag=tag, scalar_value=metric_value, global_step=self.current_epoch)

            if self.args.test:
                metric_value_test = main_eval.eval_model_on_data_with_metric(model=self.current_model, data=data_batch[-1],
                                                                             inputs=inputs_batch[-1], metric=metric, indices=self.args.n_trials-1)
                metric_value_test = metric_value_test[0]
                tag = 'metric_test_{}'.format(metric)
                self.writer.add_scalar(tag=tag, scalar_value=metric_value_test, global_step=self.current_epoch)
            if not self.args.no_printing:
                print("{}: {:.3f}".format(metric, metric_value))
        
    def save_metrics_eval(self):
        """Evaluate metrics on evaluation_trial at last epoch and save them """
        import main_eval
        main_eval.DATA_GENERATED = None
        for metric in self.args.metrics:
            data_batch = utils.read_data(self.args.data_path)
            inputs_batch = utils.read_data(self.args.inputs_path)
            inputs_subset = inputs_batch[:-1]#[:1000]
            data_subset = data_batch[:-1]#[:1000]
            #test_data = data_batch[-1]
            #test_inputs = inputs_batch[-1]

            metric_value = main_eval.eval_model_on_data_with_metric(model=self.current_model, data=data_subset,
                                                                    inputs=inputs_subset, metric=metric, indices= [0])
            metric_value = metric_value[0]  # only take first metric value, e.g. mse 1 step ahead, and klz mc

            main_eval.save_dict(metric_value)

            tag = 'metric_{}'.format(metric)
            self.writer.add_scalar(tag=tag, scalar_value=metric_value, global_step=self.current_epoch)




    def save_parameters(self):
        '''
        Save all parameters to tensorboard.
        '''

        par_dict = {**dict(self.current_model.state_dict())}
        if self.args.W_trial:


            #W = self.current_model.state_dict()['latent_model.latent_step.W']
            #par_dict['latent_model.latent_step.W'] = par_dict['latent_model.latent_step.W'].reshape((W.shape[0]*W.shape[1], W.shape[2]))
            #par_dict['latent_model.latent_step.W'] = par_dict['latent_model.latent_step.W'].reshape((W.shape[2], W.shape[0] * W.shape[1] ))
            W1 = par_dict['latent_model.latent_step.W1'][0]
            W2 = par_dict['latent_model.latent_step.W2'][0]
            #device = W.get_device()
            #tmp = (1 - tc.eye(W.shape[0]))
            par_dict['latent_model.latent_step.W1'] = W1
            par_dict['latent_model.latent_step.W2'] = W2
        par_to_tb(par_dict, epoch=self.current_epoch, writer=self.writer)

    def save_reconstruction(self):
        '''
        Save a reconstruction plot to tensorboard.
        '''
        data = self.data_set.data[0]
        inputs = self.data_set.inputs[0]
        self.current_model.plot_reconstruction(data, inputs)
        save_plot_to_tb(self.writer, text='GT vs Reconstructed',
                        global_step=self.current_epoch)
        plt.close()

    def save_latent_vs_encoded(self):
        '''
        Save a reconstruction plot to tensorboard.
        '''
        data = self.data_set.data[0]
        inputs = self.data_set.inputs[0]
        self.current_model.plot_latent_vs_encoded(data, inputs)
        save_plot_to_tb(self.writer, text='Latent vs Encoded',
                        global_step=self.current_epoch)
        plt.close()

    def save_prediction(self):
        '''
        Save a GT-Prediction plot to tensorboard.
        '''
        #data = self.data_set.data
        #inputs = self.data_set.inputs
        if isinstance(self.data_set.data, list): #check whether working with trial_sliced data or not
            data = self.data_set.data[0]
            inputs = self.data_set.inputs[0]
        else:
            data = self.data_set.data
            inputs = self.data_set.inputs
        self.current_model.plot_prediction(data, inputs, indices= [0])
        save_plot_to_tb(self.writer, text='GT vs Prediction',
                        global_step=self.current_epoch)
        plt.close()

    def indices_max_val(self, k):
        #inds = tc.randn((k))
        #temp = tc.randn((k))
        #temp = tc.topk(C.T.flatten(), k).indices
        pass

    def save_k_max_corr_units(self):
        '''
        Save k units where data is maximally correlated with inputs
        '''
        k = 3
        data = self.data_set.data
        inputs = self.inputs_set.inputs
        units_x = tc.randn((data.shape[0], k))
        latent_states_z = tc.randn((data.shape[0], k))
        C = self.C

        if self.args['use_inv_tf']:
            pass
        else:
            inds = tc.unique(tc.topk(C.T.flatten(), k).indices-C.shape[0])
            for i in inds:
                if i <= data.shape[1]:
                    pass
                else:
                    pass


        self.current_model.plot_simulated(data, inputs, T)
        figure = plt.gcf()
        save_figure_to_tb(figure, self.writer, text='maximally correlated units',
                          global_step=self.current_epoch)
        plt.close()

    def save_generated_latent_space_3d(self):
        data = self.data_set.data[0]
        inputs = self.data_set.inputs[0]
        self.current_model.plot_generated_latent_space_3d(data, inputs)
        save_plot_to_tb(self.writer, text='3D latent state space',
                        global_step=self.current_epoch)
        plt.close()

    def save_simulated(self):
        #T = 82
        #data = self.data_set.data[:T]
        #inputs = self.data_set.inputs[:T]
        if isinstance(self.data_set.data, list):
            #self.data = [t.to(model.device) for t in self.data]
            #self.inputs = [t.to(model.device) for t in self.inputs]
            data = self.data_set.data[0]
            inputs = self.data_set.inputs[0]
            T = data.shape[0]
        else:
            data = self.data_set.data
            inputs = self.data_set.inputs
            T = 1000



        self.current_model.plot_simulated(data[:T], inputs[:T], T, indices = [0])
        figure = plt.gcf()
        save_figure_to_tb(figure, self.writer, text='curve trial simulated', 
                          global_step=self.current_epoch)
        plt.close()
    
        self.current_model.plot_obs_simulated(data[:T], inputs[:T], indices = [0])
        save_plot_to_tb(self.writer, text='curve trial simulated against data'.format(0),
                        global_step=self.current_epoch)
        plt.close()

    def get_min_max(self, values):
        list_ = list(values)
        indices = [i for i in range(len(list_)) if list_[i] == 1]
        return min(indices), max(indices)


    def plot_as_image(self):
        time_steps = 1000
        data_generated = self.current_model.gen_model.get_observed_time_series(time_steps=time_steps + 1000)
        data_generated = data_generated[1000:1000 + time_steps]
        data_ground_truth = self.data_set.data
        #data_ground_truth = d_g_t[0][:time_steps]
        data_generated = data_generated[:(data_ground_truth.shape[0])]  # in case trial data is shorter than time_steps

        plt.subplot(121)
        plt.title('ground truth')
        plt.imshow(data_ground_truth, aspect=0.05, origin='lower', interpolation='none', cmap='Blues_r')
        plt.xlabel('observations')
        plt.ylabel('time steps')
        plt.subplot(122)
        plt.title('simulated')
        plt.imshow(data_generated, aspect=0.05, origin='lower', interpolation='none', cmap='Blues_r')
        plt.ylabel('time steps')
        plt.xlabel('observations')

        save_plot_to_tb(self.writer, text='curve image'.format(), global_step=self.current_epoch)


def initial_condition_trial_to_tb(gen_model, epoch, writer):
    for i in range(len(gen_model.z0)):
        trial_z0 = gen_model.z0[i].unsqueeze(0)
        x = gen_model.get_observed_time_series(800, trial_z0)  # TODO magic length of trial
        plt.figure()
        plt.title('trial {}'.format(i))
        plt.plot(x)
        figure = plt.gcf()
        save_figure_to_tb(figure, writer, text='curve_trial{}'.format(i + 1), global_step=epoch)


def data_plot(x):
    x = x.cpu().detach().numpy()
    plt.ylim(top=4, bottom=-4)
    plt.xlim(right=4, left=-4)
    plt.scatter(x[:, 0], x[:, -1], s=3)
    plt.title('{} time steps'.format(len(x)))
    return plt.gcf()


def save_plot_to_tb(writer, text, global_step=None):
    #figure = plt.figure(figsize=(10, 10))
    figure = plt.gcf()
    save_figure_to_tb(figure, writer, text, global_step)


def save_figure_to_tb(figure, writer, text, global_step=None):
    image = tb_utils.figure_to_image(figure)
    writer.add_image(text, image, global_step=global_step)


def save_data_to_tb(data, writer, text, global_step=None):
    if type(data) is list:
        for i in range(len(data)):
            plt.figure()
            plt.title('trial {}'.format(i))
            plt.plot(data[i])
            figure = plt.gcf()
            save_figure_to_tb(figure=figure, writer=writer, text='curve_trial{}_data'.format(i), global_step=None)
    else:
        plt.figure()
        plt.plot(data)
        figure = plt.gcf()
        # figure = data_plot(data)
        save_figure_to_tb(figure=figure, writer=writer, text=text, global_step=global_step)


def par_to_tb(par_dict, epoch, writer):
    for key in par_dict.keys():
        if 'E' in key or 'D' in key:
            continue

        par = par_dict[key].cpu()
        if len(par.shape) == 1:
            par = np.expand_dims(par, 1)
        # tranpose weight matrix of nn.Linear
        # to get true weight (Wx instead of xW)
        elif '.weight' in key:
            par = par.T
        par_to_image(par, par_name=key)
        save_plot_to_tb(writer, text='par_{}'.format(key), global_step=epoch)
        plt.close()


def par_to_image(par, par_name):
    plt.figure()
    # plt.title(par_name)
    sns.set_context('paper', font_scale=1.)
    sns.set_style('white')
    max_dim = max(par.shape)
    use_annot = not (max_dim > 20)
    sns.heatmap(data=par, annot=use_annot, linewidths=float(use_annot), cmap='Blues_r', square=True, fmt='.2f',
                yticklabels=False, xticklabels=False)
