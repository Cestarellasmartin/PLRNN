# %% Import Libraries
import os
import pickle
import random
import numpy as np
import torch as tc
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

from bptt.models import Model
from function_modules import model_anafunctions as func
import evaluation_Function_S as sf

plt.rcParams['font.size'] = 20

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
########################################  LOAD DATA & MODEL #############################################
#########################################################################################################

#data_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17/L6/Test1/datasets/' 
data_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17/L6/Test_WholePop/datasets/' 

#model_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/Model_Selected/'
model_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/Final_Models/OFC/CE17_L6_221008'

#model_name = 'Test_01_HU_256_l1_0.001_l2_64_l3_00_SL_400_encdim_64/001'
model_name = 'Population_HU_512_l1_0.001_l2_08_l3_00_SL_400_encdim_155/001'

mpath=os.path.join(model_path,model_name).replace('\\','/')
sf.Hyper_mod(mpath,data_path)

# Load Training & Test Data
train_n,train_i = func.load_data(data_path,'Training')
test_n,test_i = func.load_data(data_path,'Test')
NeuronPattern={"Training_Neuron":train_n,"Training_Input":train_i,
               "Testing_Neuron":test_n,"Testing_Input":test_i}

# Load Metadata
file=open(os.path.join(data_path,'Metadata.pkl'),'rb')
Metadata=pickle.load(file)
file.close()

# Load Model
num_epochs = 199000
m = Model()
m.init_from_model_path(mpath, epoch=num_epochs)

#Setup constant values
num_trials=len(NeuronPattern["Training_Neuron"])                                                                                              #number of trials
num_neurons=NeuronPattern["Training_Neuron"][0].shape[1]                                                                                                 #number of neurons
num_inputs=NeuronPattern["Training_Input"][0].shape[1]

# GENERATION SNAPSHOT
Length_trial = 5000
Warm_time=50000
Length_data=Length_trial+1+Warm_time
Input_channels= NeuronPattern["Training_Input"][0].shape[1]
Inputs=tc.zeros(Length_data,Input_channels,dtype=tc.float32)

# Generation of free trajectories for limiting behaviour - SNAPSHOT
print("::snapshot::")
SnapS=[]
for w_index in tqdm(range(len(NeuronPattern["Training_Neuron"]))):
    data_trial=tc.from_numpy(NeuronPattern["Training_Neuron"][w_index]).float()                                                     # tensor of neuronal data for initial trial data
    X, _ = m.generate_free_trajectory(data_trial,Inputs,Length_data,w_index)
    SnapS.append(X[-Length_trial:,:])
SS_Signal,_ = sf.concatenate_list(SnapS,0)

# Selection Session regions without external inputs
Zero_ITrial=[]
change_trials = []
num_traintrials = len(NeuronPattern["Training_Neuron"])
for i in tqdm(range(num_traintrials)):
     pos=np.where(np.sum(NeuronPattern["Training_Input"][i],1)==0)
     pos_change=np.where(np.diff(pos[0])>1)[0]
     pos_change = np.append(0,pos_change)
     pos_change = np.append(pos_change,NeuronPattern["Training_Neuron"][i].shape[0])
     change_trials.append(pos_change)
     Zero_ITrial.append(NeuronPattern["Training_Neuron"][i][pos[0],:])
ZT_Signal,_= sf.concatenate_list(Zero_ITrial,0)

#%% Generation Random Spike Activity

path_rawdata='D:/_work_cestarellas/Analysis/PLRNN/session_selection_v0/OFC/CE17/L6'                                   # Pathway of the data (behaviour & Spike activity)
RandomConvolution = sf.Generation_RandomActivity(path_rawdata)

#%% Computing Mean Activity (Zscore data)

# Inter-Trial Interval of Neural Data
ITI_Avg = sf.Mean_ITI(Zero_ITrial,change_trials)
# Limiting Behaviour
itera_rand=1000
LB_Avg = sf.Mean_LB(SnapS,change_trials,itera_rand)
# Random Data
itera_rand=1000
RA_Avg = sf.Mean_RA(RandomConvolution,change_trials,itera_rand,num_traintrials,num_neurons)

# Example
neuron_example = random.randint(0,num_neurons)
NeuronChange=np.array([ITI_Avg[i][neuron_example] for i in range(num_traintrials)])
ModelChange =np.array([LB_Avg[i][neuron_example] for i in range(num_traintrials)])
RandomChange =np.array([RA_Avg[i][neuron_example] for i in range(num_traintrials)])

plt.figure()
plt.plot(NeuronChange,label='Real Data')
plt.plot(ModelChange,label='Model Data')
plt.plot(RandomChange,label='Random Data')
plt.xlabel('Trials')
plt.ylabel('Mean Zscore Rate')
plt.legend()

#%% Correlation acrross trials

Corr_Data = sf.Correlation_Matrix(ITI_Avg)
Corr_Model = sf.Correlation_Matrix(LB_Avg)
Corr_Random = sf.Correlation_Matrix(RA_Avg)

C_Data_Model = sf.Plot_MixCorrelation(Corr_Data,Corr_Model)
C_Data_Random = sf.Plot_MixCorrelation(Corr_Data,Corr_Random)
C_Model_Random = sf.Plot_MixCorrelation(Corr_Model,Corr_Random)

plt.figure(figsize=(10,8))
sns.heatmap(data=C_Data_Model)
plt.xlabel("Neurons")
plt.ylabel("Neurons")
plt.title("Correlation Data vs Limiting Behaviour")

plt.figure(figsize=(10,8))
sns.heatmap(data=C_Data_Random)
plt.xlabel("Neurons")
plt.ylabel("Neurons")
plt.title("Correlation Data vs Random")

plt.figure(figsize=(10,8))
sns.heatmap(data=C_Model_Random)
plt.xlabel("Neurons")
plt.ylabel("Neurons")
plt.title("Correlation Limiting Behaviour vs Random")

#%%
plot_info={
    "title":"Correlation Diff",
    "ylabel":"Difference",
    "xtick1":"D-M",
    "xtick2":"R-M"
}

sf.Correlation_Comparison(Corr_Model,Corr_Data,Corr_Random,plot_info)

#%%
Corre_MD = []
Corre_MR = []
Corre_DR = []
for i_neuron in range(num_neurons):
    NeuronChange=np.array([ITI_Avg[i][i_neuron] for i in range(num_traintrials)])
    ModelChange =np.array([LB_Avg[i][i_neuron] for i in range(num_traintrials)])
    RandomChange =np.array([RA_Avg[i][i_neuron] for i in range(num_traintrials)])
    Corre_MD.append(np.corrcoef(NeuronChange,ModelChange)[0][1])
    Corre_MR.append(np.corrcoef(RandomChange,ModelChange)[0][1])
    Corre_DR.append(np.corrcoef(RandomChange,NeuronChange)[0][1])
Corre_MD = np.array(Corre_MD)
Corre_MR = np.array(Corre_MR)
Corre_DR = np.array(Corre_DR)

plot_info={
    "title":"Correlation Between Same Neurons",
    "ylabel":"Correlation",
    "xtick1":"D-M",
    "xtick2":"R-M",
    "xtick3":"R-D",
}
sf.Stat_Comparison(Corre_MD,Corre_MR,Corre_DR,plot_info)
# %%
