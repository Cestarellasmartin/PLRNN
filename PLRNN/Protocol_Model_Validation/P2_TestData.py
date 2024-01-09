# %% Import Libraries
import os
import pickle
import numpy as np
import pandas as pd
import torch as tc
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
from tqdm import tqdm
from scipy import signal
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from bptt.models import Model
from function_modules import model_anafunctions as func
from evaluation import pse as ps
from evaluation import mse as ms
plt.rcParams['font.size'] = 20

#%%%% Functions
def Test_model_trials(ex_path,num_epochs,NeuronPattern,Metadata):
    print("::Loading Model::")
    m = Model()
    m.init_from_model_path(ex_path, epoch=num_epochs)
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

    # W Testing parameters
    print("::Generating W testing parameters::")
    print('Set of test trials: ',Metadata["TestTrials"])

    t_prev = [i for i in Metadata["Training2Test"]]
    t_post = [i+1 for i in Metadata["Training2Test"]]

    print('trials before test trial: ',t_prev)
    print('trials after test trial: ',t_post)

    # Computing W matrices for test trials
    W2_test = np.empty((W2.shape[0],W2.shape[1],len(Metadata["TestTrials"])))
    W1_test = np.empty((W1.shape[0],W1.shape[1],len(Metadata["TestTrials"])))
    for i in range(len(t_prev)):
        W2_test[:,:,i] = (W2[:,:,t_prev[i]]+W2[:,:,t_post[i]])/2.0
        W1_test[:,:,i] = (W1[:,:,t_prev[i]]+W1[:,:,t_post[i]])/2.0
    #Generate Latent states

    TT = []
    #Generate Latent states
    W1_ind = [tc.from_numpy(W1_test[:,:,i]).float() for i in range(len(t_prev))]
    W2_ind = [tc.from_numpy(W2_test[:,:,i]).float() for i in range(len(t_prev))]
    test_tc = [tc.from_numpy(NeuronPattern["Testing_Neuron"][i]).float() for i in range(len(t_prev))]
    for i in range(len(W1_ind)):
        data_test=tc.from_numpy(NeuronPattern["Testing_Neuron"][i]).float()
        input_test=tc.from_numpy(NeuronPattern["Testing_Input"][i]).float()
        T0=int(len(NeuronPattern["Testing_Neuron"][i]))
        X, _ = m.generate_test_trajectory(data_test[0:11,:],W2_ind[i],W1_ind[i],input_test, T0,i)
        TT.append(X)
    return TT

def openhyper(mpath):
    file=open(os.path.join(mpath,'hypers.pkl').replace("\\","/"),'rb')
    hyper=pickle.load(file)
    file.close()
    return hyper

def Hyper_mod(mpath,data_path):
    file=open(os.path.join(mpath,'hypers.pkl').replace("\\","/"),'rb')
    hyper=pickle.load(file)
    file.close()
    hyper['data_path']=os.path.join(data_path,'Training_data.npy').replace('\\','/')
    hyper['inputs_path']=os.path.join(data_path,'Training_inputs.npy').replace('\\','/')
    full_name = open(os.path.join(mpath,'hypers.pkl').replace("\\","/"),"wb")                      # Name for training data
    pickle.dump(hyper,full_name)            # Save train data
    #close save instance 
    full_name.close()

#%%%% Main

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  LOAD DATA & MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% MAIN SCRIPT

############################################# Set Paths #######################################################
# Select Path for Data (Training and Test Trials)
data_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/JG15_190724/datasets/' 
# Select Path for Models (Folder containing the specific models to test)
model_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/Tuning_OFC_JG15_190724'
# Select Path for saving Data:
save_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/Tuning_OFC_JG15_190724/Evaluation_Sheets'


############################################ Load data ##########################################################

# Load Training & Test Data
train_n,train_i = func.load_data(data_path,'Training')
test_n,test_i = func.load_data(data_path,'Test')
Data_info={"Training_Neuron":train_n,"Training_Input":train_i}
NeuronPattern={"Training_Neuron":train_n,"Training_Input":train_i,
               "Testing_Neuron":test_n,"Testing_Input":test_i}
# Load Metadata
file=open(os.path.join(data_path,'Metadata.pkl'),'rb')
Metadata=pickle.load(file)
file.close()

######################################## Test measurements #######################################################
ex_path=os.path.join(model_path,"JG15_01_HU_256_l1_0.001_l2_64_l3_00_SL_400_encdim_74","001").replace('\\','/')
num_epochs=200000
#%% Generation of test trials
Hyper_mod(ex_path,data_path)
#### Load model
hyper = openhyper(ex_path)
save_files=os.listdir(ex_path)
save_models=[s for s in save_files if "model" in s]
num_epochs = len(save_models)*hyper["save_step"]
print("::Loading Model::")
m = Model()
m.init_from_model_path(ex_path, epoch=num_epochs)
m.eval()

print(repr(m), f"\nNumber of Parameters: {m.get_num_trainable()}") # get trainable parameters
_, W1t, W2t, _, _, Ct = m.get_latent_parameters()

# Transform tensor to numpy format
W2 = W2t.detach().numpy().transpose(1,2,0)
W1 = W1t.detach().numpy().transpose(1,2,0)
C = Ct.detach().numpy()

#Setup constant values
num_trials=len(test_n)#number of trials
num_neurons=W2.shape[0] #number of neurons
num_inputs=C.shape[1]   #number of inputs

print('number of trials :'+ str(num_trials))
print('number of neurons :'+ str(num_neurons))
print('number of inputs :'+ str(num_inputs))

#%% W Testing parameters
print("::Generating W testing parameters::")
print('Set of test trials: ',Metadata["TestTrials"])

t_prev = [i for i in Metadata["Training2Test"]]
t_post = [i+1 for i in Metadata["Training2Test"]]

print('trials before test trial: ',t_prev)
print('trials after test trial: ',t_post)

# Computing W matrices for test trials
W2_test = np.empty((W2.shape[0],W2.shape[1],len(Metadata["TestTrials"])))
W1_test = np.empty((W1.shape[0],W1.shape[1],len(Metadata["TestTrials"])))
for i in range(len(t_prev)):
    W2_test[:,:,i] = (W2[:,:,t_prev[i]]+W2[:,:,t_post[i]])/2.0
    W1_test[:,:,i] = (W1[:,:,t_prev[i]]+W1[:,:,t_post[i]])/2.0
#Generate Latent states

TT = []
#Generate Latent states
W1_ind = [tc.from_numpy(W1_test[:,:,i]).float() for i in range(len(t_prev))]
W2_ind = [tc.from_numpy(W2_test[:,:,i]).float() for i in range(len(t_prev))]
test_tc = [tc.from_numpy(NeuronPattern["Testing_Neuron"][i]).float() for i in range(len(t_prev))]
for i in range(len(W1_ind)):
    data_test=tc.from_numpy(NeuronPattern["Testing_Neuron"][i]).float()
    input_test=tc.from_numpy(NeuronPattern["Testing_Input"][i]).float()
    T0=int(len(NeuronPattern["Testing_Neuron"][i]))
    X, _ = m.generate_test_trajectory(data_test[0:11,:],W2_ind[i],W1_ind[i],input_test, T0,i)
    TT.append(X)

#%% Mean Squere Error
# Mean Square Error between model trial and Date per neuron
n_steps=100
val_mse = np.empty((n_steps,num_trials))
for indices in range(num_trials):
    val_mse[:,indices] = ms.test_trials_mse(m,tc.from_numpy(test_n[indices]), TT[indices], n_steps)
MEAN_mse = np.mean(val_mse)
# %% Correlation

# Correlation between Train model trial and Data per neuron
DT=[tc.from_numpy(NeuronPattern["Testing_Neuron"][i]).float() for i in range(len(NeuronPattern["Testing_Neuron"]))]
N = TT[1].size(1)                                                                          # number of neurons
NT = len(TT)
rs = tc.zeros((N,NT))                                                                       # initialization of the correlation variable

for nt in range(NT):
    eps = tc.randn_like(TT[nt]) * 1e-5                                                          # generation of little noise to avoid problems with silent neurons
    X_eps_noise = TT[nt] + eps                                                                  # adding noise to the signal 
    for n in range(N):
        rs[n,nt] = func.pearson_r(X_eps_noise[:, n], DT[nt][:, n])                                      # computation of the pearson correlation
rs = rs.detach().numpy()

MEAN_Corre=rs.mean()

#%%  Correlation Test trials
corre_list=[]
l_train_t=len(train_n) # number of training trials
norm_neu=[]
Nnorm_neu=[]
limit_validation=[]
for i_neu in range(num_neurons):
    rs_training=[]
    # Train correlation
    for i_trial in range(l_train_t-1):
        L1=train_n[i_trial][:,i_neu].shape[0]
        for i_extra in range(i_trial+1,l_train_t):
            L2=train_n[i_extra][:,i_neu].shape[0]
            if L1==L2:
                T1=tc.from_numpy(train_n[i_trial][:,i_neu])
                T2=tc.from_numpy(train_n[i_extra][:,i_neu])
                eps=tc.randn_like(T2)*1e-5
                X_eps_noise=T2+eps
                correlation=func.pearson_r(X_eps_noise,T1)
                correlation=correlation.detach().numpy()
                rs_training.append(correlation)
            elif L1<L2:
                X_rsample = signal.resample(train_n[i_extra][:,i_neu],
                                    train_n[i_trial].shape[0])
                T1 = tc.from_numpy(train_n[i_trial][:,i_neu])
                T2 = tc.from_numpy(X_rsample)
                eps=tc.randn_like(T2)*1e-5
                X_eps_noise=T2+eps
                correlation=func.pearson_r(X_eps_noise,T1)
                correlation=correlation.detach().numpy()
                rs_training.append(correlation)
            else:
                X_rsample = signal.resample(train_n[i_trial][:,i_neu],
                                    train_n[i_extra].shape[0])
                T1 = tc.from_numpy(X_rsample)
                T2 = tc.from_numpy(train_n[i_extra][:,i_neu])
                eps=tc.randn_like(T2)*1e-5
                X_eps_noise=T2+eps
                correlation=func.pearson_r(X_eps_noise,T1)
                correlation=correlation.detach().numpy()
                rs_training.append(correlation)
    n,bins,patches=plt.hist(np.array(rs_training),bins=20,range=(-1,1))
    corre_list.append(np.array(rs_training))
    plt.close()
    data=np.array(rs_training)
    norm_prob_plot = norm.ppf(np.linspace(0.01, 0.99, len(data)))  # Generate quantiles from a theoretical normal distribution
    sorted_data = np.sort(data)

    # Create a linear regression model instance
    model = LinearRegression()
    # Fit the model to the data
    model.fit(norm_prob_plot.reshape(-1,1), sorted_data.reshape(-1,1))
    # Predicting y values using the fitted model
    y_pred = model.predict(norm_prob_plot.reshape(-1,1))
    # Calculate R-squared
    r_squared = r2_score(sorted_data.reshape(-1,1), y_pred)
    # Testing normality of the correlation
    if r_squared > 0.98:
        norm_neu.append(i_neu)
        # Calculate sample mean and standard deviation
        sample_mean = np.mean(data)
        sample_std = np.std(data,ddof=1)  # ddof=1 for sample standard deviation
        limit_validation.append(sample_mean)#+sample_std)
    else:
        Nnorm_neu.append(i_neu)
        limit_validation.append(999)
limit_validation=np.array(limit_validation)
#%% Testing correlation Test Trials
ratio_tests=[]
for i in range(num_neurons):
    pass_test=rs[i,:]>limit_validation[i]
    ratio_tests.append(sum(pass_test)/len(test_n))
#Example        
plt.figure()
plt.hist(corre_list[20],bins=20,range=(-1,1))
plt.axvline(rs[20,3],color='r')
plt.axvline(limit_validation[20],color='k')

#Plot Neurons Test
plt.figure()
plt.hist(np.array(ratio_tests),bins=20,range=(0,1))

# %%
plt.figure()
plt.plot(TT[3][:,2])
plt.plot(test_n[3][:,2])
# %%
