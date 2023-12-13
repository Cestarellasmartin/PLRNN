'''
Deeper study about the simulation of test trials
'''

#%%  Import Libraries
import numpy as np
import torch as tc
import matplotlib.pyplot as plt
from bptt.models import Model
from tqdm import tqdm
import os
import pickle
import torch.nn as nn
from function_modules import model_anafunctions as func
from evaluation import pse as ps
from evaluation import mse as ms
import pandas as pd
from dtaidistance import dtw
plt.rcParams['font.size'] = 20
#%% Loading files

# Select Path for Data (Training and Test Trials)
data_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17/L6/Test0/datasets/' 
# Select Path for Models (Folder containing the specific models to test)
model_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/Tuning_OFC_CE17_L6_221008/A_reg/'
# Select Path for saving Data:
save_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/Tuning_OFC_JG15_190724/A_reg/'
# Select model:
model_name = 'CE17_Test0_01_HU_256_l1_0.001_l2_64_l3_00_SL_400_encdim_65/001'
mpath=os.path.join(model_path,model_name).replace('\\','/') # Complete path of the model selected

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

Hyper_mod(mpath,data_path)

#### Load data 
# Load Training & Test Data
train_n,train_i = func.load_data(data_path,'Training')
test_n,test_i = func.load_data(data_path,'Test')
NeuronPattern={"Training_Neuron":train_n,"Training_Input":train_i,
               "Testing_Neuron":test_n,"Testing_Input":test_i}
# Load Metadata
file=open(os.path.join(data_path,'Metadata.pkl'),'rb')
Metadata=pickle.load(file)
file.close()

#### Load Model
num_epochs = 100000
m = Model()
m.init_from_model_path(mpath, epoch=num_epochs)

# %% Generation of Test Trials by the model

At, W1t, W2t, h1t, h2t, Ct = m.get_latent_parameters()
# Transform tensor to numpy format
A = At.detach().numpy()
W2 = W2t.detach().numpy().transpose(1,2,0)
W1 = W1t.detach().numpy().transpose(1,2,0)
h1 = h1t.detach().numpy()
h2 = h2t.detach().numpy()
C = Ct.detach().numpy()
# W Testing parameters
print("::Generating W testing parameters::")
print('Set of test trials: ',Metadata["TestTrials"])
# Organisign test trial location in the Training Trials
t_prev = [i for i in Metadata["Training2Test"]]
t_post = [i+1 for i in Metadata["Training2Test"]]
# Ouput Test Trial position
print('trials before test trial: ',t_prev)
print('trials after test trial: ',t_post)
# Computing W matrices for test trials
W2_test = np.empty((W2.shape[0],W2.shape[1],len(Metadata["TestTrials"])))
W1_test = np.empty((W1.shape[0],W1.shape[1],len(Metadata["TestTrials"])))
for i in range(len(t_prev)):
    W2_test[:,:,i] = (W2[:,:,t_prev[i]]+W2[:,:,t_post[i]])/2.0
    W1_test[:,:,i] = (W1[:,:,t_prev[i]]+W1[:,:,t_post[i]])/2.0

#Generate Latent states for Test Trials (TT)
TT = []
#Generate Latent states for Test Trials
W1_ind = [tc.from_numpy(W1_test[:,:,i]).float() for i in range(len(t_prev))]
W2_ind = [tc.from_numpy(W2_test[:,:,i]).float() for i in range(len(t_prev))]
test_tc = [tc.from_numpy(NeuronPattern["Testing_Neuron"][i]).float() for i in range(len(t_prev))]
for i in range(len(W1_ind)):
    data_test=tc.from_numpy(NeuronPattern["Testing_Neuron"][i]).float()
    input_test=tc.from_numpy(NeuronPattern["Testing_Input"][i]).float()
    T0=int(len(NeuronPattern["Testing_Neuron"][i]))
    X, _ = m.generate_test_trajectory(data_test[0:11,:],W2_ind[i],W1_ind[i],input_test, T0,i)
    TT.append(X)
# %%
plt.figure()
plt.plot(TT[1][:,3],label="Model")
plt.plot(test_n[1][:,3],label="Data")
plt.legend()
plt.xlabel("Time Steps")
plt.ylabel("Zscore")

#%% Correlation between Test model trial and Data per neuron
DT=[tc.from_numpy(test_n[i]).float() for i in range(len(test_n))]
N = TT[1].size(1)                                                                          # number of neurons
NT = len(TT)
rs = tc.zeros((N,NT))                                                                       # initialization of the correlation variable

for nt in range(NT):
    eps = tc.randn_like(TT[nt]) * 1e-5                                                          # generation of little noise to avoid problems with silent neurons
    X_eps_noise = TT[nt] + eps                                                                  # adding noise to the signal 
    for n in range(N):
        rs[n,nt] = func.pearson_r(X_eps_noise[:, n], DT[nt][:, n])                                      # computation of the pearson correlation
rs = rs.detach().numpy()
#%%
plt.figure()
plt.hist(rs[:,0],alpha=0.3,label="T1")
plt.hist(rs[:,1],alpha=0.3,label="T2")
plt.hist(rs[:,2],alpha=0.3,label="T3")
plt.hist(rs[:,3],alpha=0.3,label="T4")
plt.legend()
plt.xlabel("Corr(Model vs Data)")
plt.ylabel("neurons")
plt.title("Distribution Test Trials")

Higher=np.array([np.where(rs[:,i]>0.4)[0].shape[0]/N for i in range(NT)])
Lower=np.array([np.where(rs[:,i]<0.4)[0].shape[0]/N for i in range(NT)])
Test_trials = ("T0","T1","T2","T3",)
Ratio_neurons = {
    ">0.4": Higher,
    "<0.4": Lower,
}
width = 0.5

fig, ax = plt.subplots()
bottom = np.zeros(4)
for boolean, weight_count in Ratio_neurons.items():
    p = ax.bar(Test_trials, weight_count, width, label=boolean, bottom=bottom)
    bottom += weight_count
ax.set_title("Neuron Groups Test Trials")
ax.set_ylabel("Ratio Neurons")
ax.legend(title="Correlation",bbox_to_anchor=(1.1, 1.05))
plt.show()

# %%
