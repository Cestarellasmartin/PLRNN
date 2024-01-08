# %% Import Libraries
import os
import pickle
import numpy as np
import pandas as pd
import torch as tc
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
from tqdm import tqdm

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
ex_path=os.path.join(model_path,"JG15_01_HU_128_l1_0.001_l2_0.1_l3_00_SL_400_encdim_74","001").replace('\\','/')
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
