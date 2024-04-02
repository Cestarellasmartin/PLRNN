'''
Figure 2: Unit activity comparison with model generated activity profiles from single-trials and trial averages
Panel A)
Single-unit activity, single-trial examples of recorded activity (blue) and 
model generated activity red (over trial phases: Cue - Reward Gamble - Reward Safe)

Panel B)
Average trial activity compared with generated average activity of steady state trials 
(depected with standard error) from the two rules and the two cue conditions.

With these averaged connectivity matrices averaged unit activities are simulated by
taking the exact same initial condition form the data, the exact same duration of cue light
presentation, the exact same time the animal needed to respond and the exact same duration of
reward if trial was rewarded.

Panel C)
Comparison of correlation between temporal data mean and single-trial trajectories
and correlation between reconstructed activities (with trial-averaged connectivity matrices) and recorded activities. 
Comparison for all rule and cue site combinations. 
'''

#%% Modules

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import torch as tc

from bptt.models import Model
import model_anafunctions as func
from tqdm import tqdm

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

#%% Load Data and Behaviour
# Select Path for multi-unit data
data_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17_reduction/datasets/' 
# Select Path for Models (Folder containing the specific models to test)
model_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/OFC_red'

# Loading models and simulations
model_name = 'CE14_L6_01_HU_40_l1_0.001_l2_08_l3_00_SL_400_encdim_14/001'
mpath=os.path.join(model_path,model_name).replace('\\','/')

train_n,train_i = func.load_data(data_path,'Training')
test_n,test_i = func.load_data(data_path,'Test')
NeuronPattern={"Training_Neuron":train_n,"Training_Input":train_i,
               "Testing_Neuron":test_n,"Testing_Input":test_i}

# Loading Metadata info
file=open(os.path.join(data_path,'Metadata.pkl'),'rb')
Metadata_info=pickle.load(file)
file.close()

# Load Model
num_epochs = 200000
m = Model()
m.init_from_model_path(mpath, epoch=num_epochs)
m.eval()
# Generation Training Data
ModelS=[]
for w_index in tqdm(range(len(NeuronPattern["Training_Neuron"]))):
    data_trial=tc.from_numpy(NeuronPattern["Training_Neuron"][w_index]).float()          # tensor of neuronal data for initial trial data
    input_trial = tc.from_numpy(NeuronPattern["Training_Input"][w_index]).float()
    length_sim = input_trial.shape[0]
    X, _ = m.generate_free_trajectory(data_trial,input_trial,length_sim,w_index)
    ModelS.append(X[:,:])
Model_Signal,_=func.concatenate_list(ModelS,0)
Train_Signal,_=func.concatenate_list(train_n,0)

# %% Panel A

# Selection of neuron
eneu = 13
# Selection of specfic trial
etrial = [2,15,45]

# Create subplots
alpha_value = 0.2
plt.rcParams['font.size'] = 18
fig, axs = plt.subplots(1, 3, figsize=(22, 5))
for ax,it in zip(axs,range(3)):
    vec_length = train_n[etrial[it]].shape[0]
    temp_vec = np.linspace(0,vec_length,vec_length)*0.02

    #Wheel Stop
    #Sections Cue
    ini_stop = np.where(np.diff(train_i[etrial[it]][:,0])==1.)[0]-50
    end_stop = np.where(np.diff(train_i[etrial[it]][:,0])==1.)[0]
    stop_sections = [(ini_stop[i]*0.02,end_stop[i]*0.02) for i in range(len(ini_stop))]
    #Sections Cue
    ini_cue = np.where(np.diff(train_i[etrial[it]][:,0])==1.)[0]+1
    end_cue = np.where(np.diff(train_i[etrial[it]][:,0])==-1.)[0]+1
    cue_sections = [(ini_cue[i]*0.02,end_cue[i]*0.02) for i in range(len(ini_cue))]
    #Sections Reward Gamble
    ini_grew = np.where(np.diff(train_i[etrial[it]][:,1])==4.)[0]+1
    end_grew = np.where(np.diff(train_i[etrial[it]][:,1])==-4.)[0]+1+25
    grew_sections = [(ini_grew[i]*0.02,end_grew[i]*0.02) for i in range(len(ini_grew))]
    #Sections Reward Safe
    ini_srew = np.where(np.diff(train_i[etrial[it]][:,2])==1.)[0]+1
    end_srew = np.where(np.diff(train_i[etrial[it]][:,2])==-1.)[0]+1+25
    srew_sections = [(ini_srew[i]*0.02,end_srew[i]*0.02) for i in range(len(ini_srew))]


    ax.plot(temp_vec,train_n[etrial[it]][:,eneu],color='black', lw = 2,label="Recorded Activity")
    ax.plot(temp_vec,ModelS[etrial[it]][:,eneu],color='red', lw = 2,label="Simulated Activity")
    for section in stop_sections:
        start, end = section
        ax.add_patch(Rectangle((start, ax.get_ylim()[0]), end - start, ax.get_ylim()[1]-ax.get_ylim()[0], color='#C13E71', alpha=alpha_value))
    for section in cue_sections:
        start, end = section
        ax.add_patch(Rectangle((start, ax.get_ylim()[0]), end - start, ax.get_ylim()[1]-ax.get_ylim()[0], color='#71C13E', alpha=alpha_value))
    for section in grew_sections:
        start, end = section
        ax.add_patch(Rectangle((start, ax.get_ylim()[0]), end - start, ax.get_ylim()[1]-ax.get_ylim()[0], color='#3E71C1', alpha=alpha_value))
    for section in srew_sections:
        start, end = section
        ax.add_patch(Rectangle((start, ax.get_ylim()[0]), end - start, ax.get_ylim()[1]-ax.get_ylim()[0], color='#C18E3E', alpha=alpha_value))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Local firing rate (norm.)")
# Add legend outside the subplots
handles, labels = axs[0].get_legend_handles_labels()
rect_handles = [
    Line2D([0], [0], color='#C13E71', alpha=alpha_value, lw=12, label='Stop'),
    Line2D([0], [0], color='#71C13E', alpha=alpha_value, lw=12, label='Cue'),
    Line2D([0], [0], color='#3E71C1', alpha=alpha_value, lw=12, label='Grew'),
    Line2D([0], [0], color='#C18E3E', alpha=alpha_value, lw=12, label='Srew')
]
fig.legend(handles=handles + rect_handles, labels=labels + ['Wheel Stop','Cue','Gamble Reward','Safe Reward'] , 
           loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=6, fontsize = 22)

plt.tight_layout()
plt.show()

# %% Panel B

# Average W parameters acrross trials
trial_0 = 11
trial_l = 20

# Get parameters from the model
_, W1t, W2t, _, _, Ct = m.get_latent_parameters()
# Transform tensor to numpy format
W2 = W2t.detach().numpy().transpose(1,2,0)
W1 = W1t.detach().numpy().transpose(1,2,0)
C = Ct.detach().numpy()
# General Parameters
Ntraining_trials=len(NeuronPattern["Training_Neuron"])
Ntest_trials = len(NeuronPattern["Testing_Neuron"])
num_neurons=NeuronPattern["Training_Neuron"][0].shape[1]

# Generate Latent states for Test Trials
# Identificating Test Trials in the training trial set
t_prev = 11
t_post = 21

# Computing W matrices for test trials
W2_avg = np.empty((W2.shape[0],W2.shape[1],1))
W1_avg = np.empty((W1.shape[0],W1.shape[1],1))
W2_avg[:,:,0] = W2[:,:,t_prev:t_post].mean(2)
W1_avg[:,:,0] = W1[:,:,t_prev:t_post].mean(2)
#Generate Latent states
ModelAvg = []
#Generate Latent states
data_X = tc.from_numpy(NeuronPattern["Training_Neuron"][t_prev]).float() 
for i in range(t_prev,t_post):
    input_trial=tc.from_numpy(NeuronPattern["Training_Input"][i]).float()
    length_sim = input_trial.shape[0] 
    X, _ = m.generate_test_trajectory(data_X,W2_avg,W1_avg,input_trial, length_sim,i)
    data_X = X[-1,:]
    ModelAvg.append(X)
Model_Average,_=func.concatenate_list(ModelAvg,0)