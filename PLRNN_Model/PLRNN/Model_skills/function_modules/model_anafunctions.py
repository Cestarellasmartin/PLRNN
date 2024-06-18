import numpy as np
import torch as tc
import torch.nn as nn
import os
import pickle
from typing import List
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from bptt.models import Model
from dtaidistance import dtw
from evaluation import pse as ps


def load_data(path_loc,name):
    # Funtion for loading the experimental data used for the training of the model
    # Input:
        # path_loc: root of the path where the data is localised 
    # Output:
        # data_n: neural activity data
        # data_i: input data
    file_name_neuron =os.path.join(path_loc,name+'_data.npy')                             # data of the neuron (path+file_name)
    file_name_input = os.path.join(path_loc,name+'_inputs.npy')                           # data of the input (path+file_name)
    with open(file_name_neuron,'rb') as f:                                                  # loading neuron data
        data_n = pickle.load(f)
    with open(file_name_input,'rb') as g:                                                   # loading input data
        data_i = pickle.load(g)
    return data_n, data_i    

def concatenate_list(lis,ax):
    # Function for concatenating elements of a list (Modification of Max Thurm version)
    # Input:
        # lis: list where each element is one trial (data of each neuron)
        # ax: the axe that you want to concatenate
    # Output:
        # res: concatenated trial
        # Ls: list of the concatenated elements that constitute each new concatenated trial
    res = lis[0]                                                                            # first element of the list
    Ls = [lis[0].shape[ax]]                                                                 #first length of the list
    for i in range(1, len(lis)):
        Ls.append(lis[i].shape[ax])
        res = np.concatenate((res, lis[i]), axis=ax)
    return res, Ls

def deconcatenate_list(arr, Ls):
    # Function for deconcatenating elements of a list from the concatenate_list function (Max Thurm version)
    # Input:
        # arr: concatenated trials from  the concatenate_list function
        # Ls: list of the concatenated elements that constitute each new concatenated trial 
    # Output:
        # lis: original list of individual trials
    lis = []
    tmp = 0
    for i, l in enumerate(Ls):
        lis.append(arr[:,tmp:tmp+l])
        tmp = tmp + l
    return lis

def sort_by_slope(W):
    # Function for sorting the W matrix depending on the slope of the regression of the W element across trials
    # Input: 
        # W: W matrix unrolled. Original W dimensions (3D:n_hidden,n_latents,trials). Unrolled W dimension & used (2D:n_hidden x n_latents,trials)
    # Output:
        # slopes: sorted position of the unrolled W matrix elements.
    slopes=[]
    X=np.arange(W.shape[1]).reshape(-1,1)
    for i in range(W.shape[0]):
        reg = LinearRegression().fit(X,W[i,:])
        slopes.append(reg.coef_.item())
    return np.argsort(slopes)

def pearson_r(X: tc.Tensor, Y: tc.Tensor):
    # Function for the correlation between two tensors (signal of indiviual neurons: real vs simulation)
    corr_mat = tc.corrcoef(tc.stack([X, Y]))
    return corr_mat[0, 1]

def mean_trial_correlation(X: tc.Tensor, Y: tc.Tensor):
    # Function for the mean correlation between simulated and real trials across all the neurons.
    # Input:
        # X: tensor of real data for one trial
        # Y: tensor of simulated data for one trial
    # Output:
        # rs.mean: mean value of the correlation across all neurons
    N = X.size(1)                                                                          # number of neurons
    eps = tc.randn_like(X) * 1e-5                                                          # generation of little noise to avoid problems with silent neurons
    X_eps_noise = X + eps                                                                  # adding noise to the signal
    rs = tc.zeros(N)                                                                       # initialization of the correlation variable
    for n in range(N):
        rs[n] = pearson_r(X_eps_noise[:, n], Y[:, n])                                      # computation of the pearson correlation
    return rs.mean()

def trial_correlation(X: tc.Tensor, S: tc.Tensor, model: nn.Module)-> List[float]:
    # Function for the mean correlation for each trial of the session between real and simulated data
    # Input:
        # X: tensor Neural data - whole session
        # S: tensor Input data - whole session
        # model: model trained
    # Output:
        # rs_trials: tensor of the mean correlation across neurons per each trial
    ntr = len(X)                                                                           # number of trials
    rs_trials = tc.zeros(ntr)                                                              # initialization of the correlation variable
    for i in range(ntr):
        x, s = X[i], S[i]                                                                  # extraction of the data of the individual trial
        xg, _ = model.generate_free_trajectory(x, s, x.size(0), i)                         # generation of the simulated data
        rs_trials[i] = mean_trial_correlation(x, xg)                                       # computation of the mean correlation across neurons
    return rs_trials


def trial_pse(X: tc.Tensor, S: tc.Tensor, model: nn.Module)-> List[float]:
    # Function for the power spectrum error pse for each trial of the session between real and simulated data
    # Input:
        # X: tensor Neural data - whole session
        # S: tensor Input data - whole session
        # model: model trained
    # Output:
        # rs_trials: tensor of the mean correlation across neurons per each trial
    ntr = len(X)                                                                           # number of trials
    pse_trial = []                                                          # initialization of the correlation variable
    pse_triallist=[]
    for i in range(ntr):
        x, s = X[i], S[i]                                                                  # extraction of the data of the individual trial
        xg, _ = model.generate_free_trajectory(x, s, x.size(0), i)                         # generation of the simulated data
        pse_mean,pse_triallist = ps.power_spectrum_error(x, xg)                                      # computation of the mean pse across neurons
        pse_trial.append(pse_mean)
        pse_triallist.append(pse_triallist)
    return pse_trial,pse_triallist


def Pext_simul(idx_trial,data_n,data_i,ext_in,m):
    # Function for the simulation of the individual trials in PARALLEL.
    # The simulation also take into account an extra term to simulate long trials (variable: ext_in)
    # Input data:
        # idx_trial: number of the trial simulated
        # data_n: experimental trial data
        # data_i: input trial data 
        # ext_in: number of extra points used for the simulation of the trial without input
        # m: trained model
    # Output: 
        # X: numpy array 2D- rows:time, columns: latent states(z). Individual trial
    num_inputs=data_i.shape[1]                                                              # number of inputs
    zero_add=tc.zeros(ext_in,num_inputs)                                                    # extra inputs 0 for simulation (0.02 s/bins)
    inputs_t1=tc.cat((data_i,zero_add),axis=0)                                              # adding inputs to long trajetories: value - 0
    T=int(inputs_t1.shape[0])
    Xt, _ = m.generate_free_trajectory(data_n,inputs_t1, T,idx_trial)                       # generation of simulated data
    X=Xt.detach().numpy()                                                                   # transformation from tensor to array
    return X

def ext_simul(num_trials,data_n,data_i,m,zero_add):
    # Serial version of ext_simul_par. (Slower that the paralell version. Not recomended for long trials and long session)
    # Input:
        # num_trials: number of trials that you want to simulate
        # data_n: experimental data (whole session)
        # data_i: input data (whole session)
        # m: trained model
        # zero_add: tensor of zeros inputs that you want to add to the trial. Example: zero_add=tc.zeros(80000,num_inputs) #-- extra inputs 0 for simulation (1600 s-80000 bins)
    # Output: 
        # X: list of numpy array 2D- rows:time, columns: latent states(z). Each element of the list is a new trial.
    X=[]
    # Long Trajectory
    for idx_trial in tqdm(range(num_trials), desc='Simulating long trials of session'):
        #load data of one trial
        data_t1=data_n[idx_trial]
        #load inputs of one trial 
        inputs_1=data_i[idx_trial]
        #adding inputs to long trajetories: value - 0
        inputs_t1=tc.cat((inputs_1,zero_add),axis=0)
        T=int(inputs_t1.shape[0])
        Xt, Zt = m.generate_free_trajectory(data_t1,inputs_t1, T,idx_trial)
        X.append(Xt.detach().numpy())
    return X

def PW_gene_corre(idx_w,idx_trial,data_n,data_i,m):
    # Paralell function to analyse the generalization of the model with the W matrix. 
    # The function generate the simulation of each trial with all the trained W and compute the correlation with the real data
    # Input:
        # idx_w: trial_index of the W matrix
        # idx_trial: index of the trial
        # data_n: neural activity Tensor. Dim:(bin_time,# of neurons)
        # data_i: input Tensor. Dim:(bin_time,# of inputs)
        # m: trained model
    # Output:
        # rs_trials: mean correlation across neurons of one trial
        # idx_w: element index of the W used
        # idx_trial: element index of the trial used
    T=int(data_i.shape[0])
    X, _ = m.generate_free_trajectory(data_n,data_i, T,idx_w)
    rs_trials = mean_trial_correlation(data_n,X)
    return rs_trials, idx_w, idx_trial

def W_matrix_generation(mpath):
    # Generation of the model to obtain the parameters of the model
    # The parameters are numpy array
    # Load model
    m = Model()
    m.init_from_model_path(mpath, epoch=5000)
    m.eval()

    print(repr(m), f"\nNumber of Parameters: {m.get_num_trainable()}") # get trainable parameters
    At, W1t, W2t, h1t, h2t, Ct = m.get_latent_parameters()

    # Transform tensor to numpy format
    A = At.detach().numpy()
    W2 = W2t.detach().numpy().transpose(1,2,0)
    W1 = W1t.detach().numpy().transpose(1,2,0)
    h1 = h1t.detach().numpy()
    h2 = h2t.detach().numpy()
    C = Ct.detach().numpy()

    #Setup constant values
    num_trials=W2.shape[-1] #number of trials
    num_neurons=W2.shape[0] #number of neurons
    num_inputs=C.shape[1]   #number of inputs
    
    print('number of trials :'+ str(num_trials))
    print('number of neurons :'+ str(num_neurons))
    print('number of inputs :'+ str(num_inputs))

    return A,W2,W1,h2,h1,C,m

def DTW_para(sim,exp,trial,neun,alpha):
    # Computing the Dynamic Time Warping distance for each neuron (column) and each trial (row)
    dtw_d = dtw.distance(sim,exp)
    ddtw_d = dtw.distance(np.diff(sim,n=1),np.diff(exp,n=1))
    Dist = alpha*dtw_d+(1-alpha)*ddtw_d
    return trial, neun, Dist