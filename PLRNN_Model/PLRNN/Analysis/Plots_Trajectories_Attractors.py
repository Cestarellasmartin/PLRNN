#%%
import os
import pickle
import torch as tc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import LinearRegression
from bptt.models import Model
from function_modules import model_anafunctions as func
#%% Main
# Select Path for Data (Training and Test Trials)
data_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17/L6/Test0/datasets/' 
# Select Path for Models (Folder containing the specific models to test)
model_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/Tuning_OFC_CE17_L6_221008/A_reg/'
# Select name of the model:
model_name = 'CE17_Test0_01_HU_256_l1_0.001_l2_64_l3_00_SL_400_encdim_65/001'
mpath=os.path.join(model_path,model_name).replace('\\','/')
############################################ Load data ##########################################################

# Load Training & Test Data
train_n,train_i = func.load_data(data_path,'Training')
test_n,test_i = func.load_data(data_path,'Test')
NeuronPattern={"Training_Neuron":train_n,"Training_Input":train_i,
               "Testing_Neuron":test_n,"Testing_Input":test_i}
# Load Metadata
file=open(os.path.join(data_path,'Metadata.pkl'),'rb')
Metadata=pickle.load(file)
file.close()

######################################## Test measurements #######################################################

# Computation of limitingbehaviour measurements for the models in your model_path

# Load Model
num_epochs=100000       
m = Model()
m.init_from_model_path(mpath, epoch=num_epochs)
# %%
# # Selection Session regions without external inputs
print("Obtaining Inter-Trial Intervals")
ITI_Trial=[]
num_traintrials = len(NeuronPattern["Training_Neuron"])
for i in tqdm(range(num_traintrials),"Obtaining Data"):
     pos=np.where(np.sum(NeuronPattern["Training_Input"][i],1)==0)
     ITI_Trial.append(NeuronPattern["Training_Neuron"][i][pos[0],:])
ITI_Signal,_= func.concatenate_list(ITI_Trial,0)
Length_trialITI = ITI_Signal.shape[0]      

#%% SnapShot
Warm_time=50000
Input_channels = NeuronPattern["Training_Input"][0].shape[1]
# Generation of free trajectories for limiting behaviour - SNAPSHOT
print("::snapshot::")
SS_Model=[]
for w_index in tqdm(range(num_traintrials)):
    Length_trial=ITI_Trial[w_index].shape[0]
    Length_data=Length_trial+1+Warm_time
    Inputs=tc.zeros(Length_data,Input_channels,dtype=tc.float32)
    data_trial=tc.from_numpy(NeuronPattern["Training_Neuron"][w_index]).float()          # tensor of neuronal data for initial trial data
    X, _ = m.generate_free_trajectory(data_trial,Inputs,Length_data,w_index)
    SS_Model.append(X[-Length_trial:,:])
SS_Signal,_=func.concatenate_list(SS_Model,0)
#%% PLot Round1
# individual trial
num_neurons=ITI_Trial[0].shape[1]
tr=45
neu=np.random.choice(num_neurons,3,replace=False)
ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
ax.plot(SS_Model[tr][:,neu[0]], SS_Model[tr][:,neu[1]], SS_Model[tr][:,neu[2]], 'red',linestyle='dashed',label="Generated")
ax.plot(ITI_Trial[tr][:,neu[0]], ITI_Trial[tr][:,neu[1]], ITI_Trial[tr][:,neu[2]], 'blue',label="Real")
ax.legend()
ax.set_xlabel('Neu 1',labelpad =15)
ax.set_ylabel('Neu 2',labelpad =15)
ax.set_zlabel('Neu 3',labelpad =15)
ax.set_title('Limit Behaviour')
plt.show()

#full session
num_neurons=ITI_Trial[0].shape[1]
neu=np.random.choice(num_neurons,3,replace=False)
ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
ax.plot(SS_Signal[:,neu[0]], SS_Signal[:,neu[1]], SS_Signal[:,neu[2]], 'red',linestyle='dashed',label="Generated")
ax.plot(ITI_Signal[:,neu[0]], ITI_Signal[:,neu[1]], ITI_Signal[:,neu[2]], 'blue',label="Real")
ax.legend()
ax.set_xlabel('Neu 1',labelpad =15)
ax.set_ylabel('Neu 2',labelpad =15)
ax.set_zlabel('Neu 3',labelpad =15)
ax.set_title('Limit Behaviour')
plt.show()

#%% Big Generation Data
#%% SnapShot
Warm_time=1000000
Input_channels = NeuronPattern["Training_Input"][0].shape[1]
Inputs=tc.zeros(Warm_time,Input_channels,dtype=tc.float32)
# Generation of free trajectories for limiting behaviour - SNAPSHOT
print("::snapshot::")
G_Model=[]
for w_index in tqdm(range(num_traintrials)):
    data_trial=tc.from_numpy(NeuronPattern["Training_Neuron"][w_index]).float()          # tensor of neuronal data for initial trial data
    X, _ = m.generate_free_trajectory(data_trial,Inputs,Warm_time,w_index)
    G_Model.append(X[:,:])
#%% Selection most important neurons
num_neurons=NeuronPattern["Training_Neuron"][0].shape[1]
M_Activity=[NeuronPattern["Training_Neuron"][i].mean(0).reshape(1,num_neurons) for i in range(num_traintrials)]
Mean_SignalT,_=func.concatenate_list(M_Activity,0)

x_t = np.linspace(0,num_traintrials,num_traintrials).reshape(num_traintrials,1)
model_l = LinearRegression()
r_sq = []
slope = []
for n in range(num_neurons):
    model_l.fit(x_t,Mean_SignalT[:,n])
    r_sq.append(model_l.score(x_t,Mean_SignalT[:,n]))
    slope.append(model_l.coef_)

plt.figure()
plt.scatter(np.abs(np.array(slope)),np.array(r_sq))

r_sq_sorted=np.sort(np.array(r_sq))[-3:]
neu_d=[np.where(r_sq_sorted[i]==np.array(r_sq))[0][0] for i in range(len(r_sq_sorted))]

#%% PLots2
import matplotlib as mpl
num_neurons = G_Model[0].shape[1]
num_trials = len(G_Model)
co=mpl.cm.get_cmap('viridis', num_traintrials)
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=0, vmax=num_traintrials)

fig,ax=plt.subplots()
for iw in range(num_trials):
    ax.scatter(G_Model[iw][-2000:,neu_d[0]], G_Model[iw][-2000:,neu_d[2]], color=co(iw),s=4)
cbar=fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),label="Trials")
ax.set_xlabel('Neu 1',labelpad =15)
ax.set_ylabel('Neu 2',labelpad =15)
#%%
ax = plt.figure().add_subplot(projection='3d')
for iw in range(num_trials):
    ax.plot(G_Model[iw][-2000:,neu_d[0]], G_Model[iw][-2000:,neu_d[1]], G_Model[iw][-2000:,neu_d[2]], color=co(iw),linestyle='dashed')
ax.legend()
ax.set_xlabel('Neu 1',labelpad =15)
ax.set_ylabel('Neu 2',labelpad =15)
ax.set_zlabel('Neu 3',labelpad =15)
plt.show()


# %%
