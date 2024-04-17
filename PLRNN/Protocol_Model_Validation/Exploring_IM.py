# %% Import Libraries
#This is a test
import os
import pickle
import scipy.io
import numpy as np
import pandas as pd
import torch as tc
from scipy.stats import zscore
from sklearn.decomposition import PCA

import ruptures as rpt

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.patches as mpatches
from tqdm import tqdm
from bptt.models import Model
from function_modules import model_anafunctions as func

plt.rcParams['font.size'] = 20
#%% Load Recordings

data_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17/L6/Test0/datasets/' 
# Select Path for Models (Folder containing the specific models to test)
model_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/Tuning_OFC_CE17_221008'

#%% Load Training & Test Data
Act,Inp = func.load_data(data_path,'Training')

#%% Loading models and simulations
model_name = 'CE1701_HU_256_l1_0.001_l2_08_l3_00_SL_400_encdim_65/001'
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

#%%
etrial=29
times_trial=np.linspace(0,ModelS[etrial][:,0].shape[0]*0.02,ModelS[etrial][:,0].shape[0])
plt.figure()
plt.plot(times_trial,ModelS[etrial][:,0])
plt.plot(times_trial,train_n[etrial][:,0])
plt.title("Trial "+str(etrial))
plt.xlabel("Time (s)")
plt.ylabel("FR")

times=np.linspace(0,Model_Signal[0:4000,0].shape[0]*0.02,Model_Signal[0:4000,0].shape[0])

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(times,Model_Signal[0:4000,10],c='blue')
axs[0].plot(times,Train_Signal[0:4000,10],'--',c='k')
axs[0].set_xlabel("Time (s)")
axs[1].plot(times,Model_Signal[0:4000,12],c='red')
axs[1].plot(times,Train_Signal[0:4000,12],'--',c='k')
axs[1].set_xlabel("Time (s)")
axs[2].plot(times,Model_Signal[0:4000,4],c='green')
axs[2].plot(times,Train_Signal[0:4000,4],'--',c='k')
axs[2].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BEHAVIOUR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
path_beh='D:/_work_cestarellas/Analysis/PLRNN/Session_Selected/OFC/JG15_190724_clustered'

os.chdir(path_beh)
list_files = os.listdir(path_beh)

for i in list_files:
    if i.find('Behaviour')>0:
        Behaviour_name = i
    if i.find('Metadata')>0:
        Metadata_name = i
    if i.find('SpikeActivity')>0:
        Spike_name = i

# Load data
# Open the Behaviour file
Bdata = scipy.io.loadmat(Behaviour_name)
BehData = Bdata[list(Bdata.keys())[-1]]
# Classification of trials following the behabiour
GambleRewardTrials = np.where((BehData[:,12]==1) & (BehData[:,13]==1))[0]
GambleNoRewardTrials =  np.where((BehData[:,12]==1) & (BehData[:,13]==0))[0]
SafeRewardTrials = np.where((BehData[:,12]==0) & (BehData[:,13]==1))[0]
SafeNoRewardTrials = np.where((BehData[:,12]==0) & (BehData[:,13]==0))[0]
NoRespondingTrials = np.where(BehData[:,10]==1)[0]

# Blocks
Block_Prob = np.unique(BehData[:,5])
BlockTrials = [np.where(Block_Prob[i]==BehData[:,5])[0][0] for i in range(len(Block_Prob))]
# Smoothing the data for plotting
ScaleDecision=BehData[:,12]+(BehData[:,12]-1)
SigmaDecision=1
Binx =0.5
KernelWindow = np.arange(-3*SigmaDecision, 3*SigmaDecision, Binx)
KernelDecision = np.exp(-(KernelWindow/SigmaDecision)**2/2)
DecisionConvolution=np.convolve(ScaleDecision,KernelDecision,mode='same')
DecisionNormalized=(DecisionConvolution/np.nanmax(np.abs(DecisionConvolution))+1)/2

if NoRespondingTrials[0]<20:
    first_trial = NoRespondingTrials[0]+1
    last_trial = NoRespondingTrials[1]-1
else:
    first_trial = 0
    last_trial = NoRespondingTrials[0]-1

# Concatenated Trials
SR_m=1/0.02
V_mend=[ModelS[i].shape[0]/SR_m for i in range(len(ModelS))]
T_mend=np.zeros((len(V_mend),1))
T_mstart=np.zeros((len(V_mend),1))
T_mend[0]=V_mend[0]
for i in range(1,len(V_mend)):
    T_mend[i]=V_mend[i]+T_mend[i-1]
    T_mstart[i]=T_mend[i-1]

SR_r=20000
StartTrial = np.array(BehData[:,15]/SR_r)                                                  # times of trial initiation
T_rstart=StartTrial-StartTrial[0]
Model_trial=[]
for i in range(len(T_mstart)):
    Model_trial.append(np.sum(T_rstart<T_mstart[i]))

label_xticks = [ i for i in range(len(Model_trial))]
# FIGURE: Plot behaviour performance
plt.figure(figsize=(25,5))
plt.plot(DecisionNormalized)
for i in GambleRewardTrials:
    plt.axvline(i,ymin=0.9,ymax=1.0,color='g') 
for i in GambleNoRewardTrials:
    plt.axvline(i,ymin=0.9,ymax=1.0,color='gray')
for i in SafeRewardTrials:
    plt.axvline(i,ymin=0.0,ymax=0.1,color='orange')
for i in SafeNoRewardTrials:
    plt.axvline(i,ymin=0.0,ymax=0.1,color='gray')
for i in NoRespondingTrials:
    plt.axvline(i,ymin=0.45,ymax=0.55,color='blue')
for i in BlockTrials:
    plt.axvline(i,linestyle='dashed',color='r',)    
for i in Model_trial:
    plt.axvline(i,linestyle='dashed',color='k',)   
plt.ylim([-0.2,1.2])
plt.yticks(ticks=[1.0,0.5,0.0])
plt.xlim([first_trial,last_trial])
plt.xticks(Model_trial[1:-1:2],label_xticks[1:-1:2])
plt.xlabel('Trials')
plt.ylabel("Animal's choice")
plt.title('Session for modelling')

plt.figure()
plt.plot(DecisionNormalized)
plt.xlim([first_trial,last_trial])
plt.xticks(Model_trial[1:-1:10],label_xticks[1:-1:10])
plt.xlabel("Trials")
plt.ylabel("Gamble choice prob")
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BEHAVIOUR  CE17 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
path='D:/_work_cestarellas/Analysis/PLRNN/Session_Selected/OFC/CE17_L6'
filename = "session.csv"
os.chdir(path)
# Load data
# Open the Behaviour file
BehData = pd.read_csv(filename)
# Classification of trials following the behabiour
GambleRewardTrials = np.where((BehData["gamble"]==1) & (BehData["REWARD"]==1))[0]
GambleNoRewardTrials =  np.where((BehData["gamble"]==1) & (BehData["REWARD"]==0))[0]
SafeRewardTrials = np.where((BehData["safe"]==1) & (BehData["REWARD"]==1))[0]
SafeNoRewardTrials = np.where((BehData["safe"]==1) & (BehData["REWARD"]==0))[0]
NoRespondingTrials = np.where(BehData["F"]==1)[0]

# Blocks
Block_Prob = np.unique(BehData["probability G"])
BlockTrials = [np.where(Block_Prob[i]==BehData["probability G"])[0][0] for i in range(len(Block_Prob))]
# Smoothing the data for plotting
ScaleDecision=BehData["gamble"]+(BehData["safe"]*(-1))
SigmaDecision=1
Binx =0.5
KernelWindow = np.arange(-3*SigmaDecision, 3*SigmaDecision, Binx)
KernelDecision = np.exp(-(KernelWindow/SigmaDecision)**2/2)
DecisionConvolution=np.convolve(ScaleDecision,KernelDecision,mode='same')
DecisionNormalized=(DecisionConvolution/np.nanmax(np.abs(DecisionConvolution))+1)/2

if NoRespondingTrials[0]<20:
    first_trial = NoRespondingTrials[0]+1
    last_trial = NoRespondingTrials[1]-1
else:
    first_trial = 0
    last_trial = NoRespondingTrials[0]-1

# Concatenated Trials
SR_m=1/0.02
V_mend=[ModelS[i].shape[0]/SR_m for i in range(len(ModelS))]
T_mend=np.zeros((len(V_mend),1))
T_mstart=np.zeros((len(V_mend),1))
T_mend[0]=V_mend[0]
for i in range(1,len(V_mend)):
    T_mend[i]=V_mend[i]+T_mend[i-1]
    T_mstart[i]=T_mend[i-1]

SR_r=20000
StartTrial = np.array(BehData["wheel_stop"]/SR_r)                                                  # times of trial initiation
T_rstart=StartTrial-StartTrial[0]
Model_trial=[]
for i in range(len(T_mstart)):
    Model_trial.append(np.sum(T_rstart<T_mstart[i]))

label_xticks = [ i for i in range(len(Model_trial))]
# FIGURE: Plot behaviour performance
plt.figure(figsize=(25,5))
plt.plot(DecisionNormalized)
for i in GambleRewardTrials:
    plt.axvline(i,ymin=0.9,ymax=1.0,color='g') 
for i in GambleNoRewardTrials:
    plt.axvline(i,ymin=0.9,ymax=1.0,color='gray')
for i in SafeRewardTrials:
    plt.axvline(i,ymin=0.0,ymax=0.1,color='orange')
for i in SafeNoRewardTrials:
    plt.axvline(i,ymin=0.0,ymax=0.1,color='gray')
for i in NoRespondingTrials:
    plt.axvline(i,ymin=0.45,ymax=0.55,color='blue')
for i in BlockTrials:
    plt.axvline(i,linestyle='dashed',color='r',)    
for i in Model_trial:
    plt.axvline(i,linestyle='dashed',color='k',)   
plt.ylim([-0.2,1.2])
plt.yticks(ticks=[1.0,0.5,0.0])
plt.xlim([first_trial,last_trial])
plt.xticks(Model_trial[1:-1:2],label_xticks[1:-1:2])
plt.xlabel('Trials')
plt.ylabel("Animal's choice")
plt.title('Session for modelling')

plt.figure()
plt.plot(DecisionNormalized)
plt.axvline(Model_trial[23],c="black",linestyle="--")
plt.axvline(Model_trial[28],c="black",linestyle="--")
plt.xlim([first_trial,last_trial])
plt.xticks(Model_trial[1:-1:10],label_xticks[1:-1:10])
plt.xlabel("Trials")
plt.ylabel("Gamble choice prob")
#%% W parameters

At, W1t, W2t, h1t, h2t, Ct = m.get_latent_parameters()
# Transform tensor to numpy format
A = At.detach().numpy()
W2 = W2t.detach().numpy().transpose(1,2,0)
W1 = W1t.detach().numpy().transpose(1,2,0)
h1 = h1t.detach().numpy()
h2 = h2t.detach().numpy()
C = Ct.detach().numpy()

# Parameters
HU=W2.shape[1]                      # Number hidden units
Neurons=W2.shape[0]                 # Number Neurons
Trials=W2.shape[2]                  # Number Trials

# W2 Matrix
W2_np = W2
mat=W2_np
mat=mat.reshape(-1,mat.shape[2])
mat=zscore(mat,axis=1)
sort_pos = func.sort_by_slope(mat)
mat=mat[sort_pos,:]

plt.figure(figsize=(15,10))
plt.rcParams['font.size'] = 20
sns.heatmap(data=mat)
plt.axvline(280,color='k')
plt.title('W2 Map')
plt.xlabel('Trials')
plt.ylabel('Dimz-Neurons')

# W1 Matrix
W1_np = W1
mat1=W1_np
mat1=mat1.reshape(-1,mat1.shape[2])
mat1=zscore(mat1,axis=1)
sort_pos = func.sort_by_slope(mat1)
mat1=mat1[sort_pos,:]

plt.figure(figsize=(15,10))
plt.rcParams['font.size'] = 20
sns.heatmap(data=mat1)
plt.axvline(280,color='k')
plt.title('W1 Map')
plt.xlabel('Trials')
plt.ylabel('Dimz-Neurons')

#Correlation W1 and W2 matrix trialbytrial
print('Computing Correlation W1')
W1_cortbt=np.corrcoef(mat1.T)
print('Computing Correlation W2')
W2_cortbt=np.corrcoef(mat.T)

plt.figure(figsize=(15,10))
plt.rcParams['font.size'] = 20
sns.heatmap(data=W1_cortbt,cmap='viridis',vmin=0,vmax=1)
plt.title('W1 Map Correlation')
plt.xlabel('Trials')
plt.ylabel('Trials')

#Correlation W2 matrix trialbytrial
plt.figure(figsize=(15,10))
plt.rcParams['font.size'] = 20
sns.heatmap(data=W2_cortbt,cmap='viridis')
plt.title('W2 Map Correlation')
plt.xlabel('Trials')
plt.ylabel('Trials')



# %% Compute Change points
Limited_trials=Model_trial[-1]

# Behaviour
BD=DecisionNormalized[0:Limited_trials]
# Perform change point detection
algo = rpt.Pelt(model="l2",min_size=1).fit(BD)
result_beh = algo.predict(pen=1)

# Model parameters
fp_series=[]
for i in range(mat.shape[0]):
    algo = rpt.Pelt(model="l2",min_size=10).fit(mat[i,:])
    result = algo.predict(pen=1)
    fp_series.append(result[0:-1])

fp_model=np.concatenate(fp_series)
fp_model=fp_model.astype(int)
# Transition from concatenated trials to behavioural trials
fp_modelt=np.array([Model_trial[fp_model[i]] for i in range(len(fp_model))])
fp_dis=np.zeros((len(BD),1))
for i in range(len(fp_dis)):
    fp_dis[i]=sum(fp_modelt==i)
fp_dis=fp_dis/np.max(fp_dis)

# Neuronal Activity
full_n,full_i = func.load_data(data_path,'FullSession')
full_signal,_= func.concatenate_list(full_n,0)

FR_trials=[]
for i in range(len(full_n)):
    FR_trials.append(np.mean(full_n[i],axis=0))
FR = np.array(FR_trials)

fp_rec=[]
for i in range(FR.shape[1]):
    algo = rpt.Pelt(model="l2",min_size=10).fit(FR[:,i])
    result = algo.predict(pen=1)
    fp_rec.append(result[0:-1])

fp_rect=np.concatenate(fp_rec)
fp_rect=fp_rect.astype(int)

fp_Ntrial=np.array([Model_trial[fp_rect[i]] for i in range(len(fp_rect))])
fp_Ndis=np.zeros((len(BD),1))
for i in range(len(fp_Ndis)):
    fp_Ndis[i]=sum(fp_Ntrial==i)
fp_Ndis=fp_Ndis/np.max(fp_Ndis)

plt.figure(figsize=(15,5))
plt.plot(BD,label='Behaviour')
plt.axvline(result_beh[0],color='black',label='CP Beh')
for i in range(1,len(result_beh)-1):
    plt.axvline(result_beh[i],color='black')
plt.plot(fp_dis, color='darkred',label='CP model')
plt.plot(fp_Ndis, linestyle='--', color='green', label='CP neurons')
plt.ylabel('Probability')
plt.xlabel('Trials')
plt.title('Change Point Detection')
plt.legend()
plt.show()

# %% Link between Parameters and neurons


W2_z=zscore(W2,axis=2)

trial1=8
plt.figure(figsize=(15,10))
plt.rcParams['font.size'] = 20
sns.heatmap(data=W2_z[:,:,trial1])
plt.title('W2 Map Trial '+str(trial1))
plt.xlabel('Hidden Units')
plt.ylabel('Dimz-Neurons')

trial2=30
plt.figure(figsize=(15,10))
plt.rcParams['font.size'] = 20
sns.heatmap(data=W2_z[:,:,trial2])
plt.title('W2 Map Trial '+str(trial2))
plt.xlabel('Hidden Units')
plt.ylabel('Dimz-Neurons')

plt.figure(figsize=(15,10))
plt.rcParams['font.size'] = 20
sns.heatmap(data=(W2[:,:,trial2]-W2[:,:,trial1]))
plt.title('W2 Map Trial '+str(trial2)+"-"+str(trial1))
plt.xlabel('Hidden Units')
plt.ylabel('Dimz-Neurons')

#%% Classification of Hidden Units

Wh=np.full((HU,Trials),np.nan)
Wh2=np.full((HU,Trials),np.nan)
for i in range(Trials):
    Wh[:,i]=np.abs(W1[:,:,i].T).mean(axis=0)
    Wh2[:,i]=np.abs(W2[:,:,i].T).mean(axis=1)
Whz=zscore(Wh.T).T
Whz2=zscore(Wh2.T).T

sort_pos = func.sort_by_slope(Whz)
sort_pos2 = func.sort_by_slope(Whz2)

plt.figure(figsize=(15,10))
sns.heatmap(Whz[sort_pos,:])
plt.xlabel("Trials")
plt.ylabel("Hidden Units")
plt.title("Normalized Mean Weights accross trials W1")

plt.figure(figsize=(15,10))
sns.heatmap(Whz2[sort_pos2,:])
plt.xlabel("Trials")
plt.ylabel("Hidden Units")
plt.title("Normalized Mean Weights accross trials W2")

def temporal_decomp(X,name):
    # Temporal information from Hidden Units
    # Instantiate PCA object
    pca = PCA(n_components=3)  # Specify the number of components (dimensions) you want to keep
    # Fit PCA model to the data
    pca.fit(X)
    print("Variance explained ratio: ",pca.explained_variance_ratio_)
    # Transform the data to the new feature space
    X_pca = pca.transform(X)
    colors=np.linspace(0,X_pca.shape[0],X_pca.shape[0])
    plt.figure()
    scatter=plt.scatter(X_pca[:,0],X_pca[:,1],marker='o',c=colors,cmap='viridis')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Trials')
    plt.title(name)
    plt.xlabel("Comp. 1")
    plt.ylabel("Comp. 2")
    plt.show()


temporal_decomp(Whz[sort_pos,:].T,"Trial Space W1")
temporal_decomp(Whz2[sort_pos2,:].T,"Trial Space W2")

def spatial_decomp(X,name):
    # Spatial information from Hidden Units
    # Instantiate PCA object
    pca = PCA(n_components=3)  # Specify the number of components (dimensions) you want to keep
    # Fit PCA model to the data
    pca.fit(X)
    print("Variance explained ratio: ",pca.explained_variance_ratio_)
    # Transform the data to the new feature space
    X_pca = pca.transform(X)
    colors=np.linspace(0,X_pca.shape[0],X_pca.shape[0])
    plt.figure()
    scatter=plt.scatter(X_pca[:,0],X_pca[:,1],marker='o',c=colors,cmap='cividis')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Hidden Units')
    plt.title(name)
    plt.xlabel("Comp. 1")
    plt.ylabel("Comp. 2")
    plt.show()
    return X_pca
W1_pca = spatial_decomp(Whz[sort_pos,:],"Hidden Units Space W1")
W2_pca = spatial_decomp(Whz2[sort_pos2,:],"Hidden Units Space W2")

#%% Structure Hidden Units Space


def HU_structure(X_pca,X,pos,limit_C10,limit_C11,Transition_Trial0,Transition_Trial1,param):
    colors=np.linspace(0,X_pca.shape[0],X_pca.shape[0])
    plt.figure()
    scatter=plt.scatter(X_pca[:,0],X_pca[:,1],marker='o',c=colors,cmap='cividis')
    plt.axvline(limit_C10)
    plt.axvline(0,ls='--')
    plt.axhline(0,ls='--')
    plt.axvline(limit_C11)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Hidden Units')
    plt.title("PCA "+param)
    plt.xlabel("Comp. 1")
    plt.ylabel("Comp. 2")
    plt.show()

    # Quadrants
    HU11=np.where((X_pca[:,0]<0) & (X_pca[:,0]>limit_C10) & (X_pca[:,1]>0))[0]
    HU12=np.where((X_pca[:,0]>0) & (X_pca[:,0]<limit_C11) & (X_pca[:,1]>0))[0]
    HU21=np.where((X_pca[:,0]<0) & (X_pca[:,0]>limit_C10) & (X_pca[:,1]<0))[0]
    HU22=np.where((X_pca[:,0]>0) & (X_pca[:,0]<limit_C11) & (X_pca[:,1]<0))[0]

    HU00=np.where((X_pca[:,0]<limit_C10))[0]
    HU33=np.where((X_pca[:,0]>limit_C11))[0]

    plt.figure()
    plt.plot(np.mean(X[pos[HU11],:],0),label="H11")
    plt.plot(np.mean(X[pos[HU12],:],0),label="H12")
    plt.plot(np.mean(X[pos[HU21],:],0),label="H21")
    plt.plot(np.mean(X[pos[HU22],:],0),label="H22")
    plt.axvline(Transition_Trial0,ls='--',c='black')
    plt.axvline(Transition_Trial1,ls='--',c='black')
    plt.xlabel("Trials")
    plt.ylabel("Normalized weights")
    plt.title("Transition Hidden Units "+param)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

    plt.figure()
    plt.plot(np.mean(X[pos[HU00],:],0),label="H00")
    plt.plot(np.mean(X[pos[HU33],:],0),label="H33")
    plt.axvline(Transition_Trial0,ls='--',c='black')
    plt.axvline(Transition_Trial1,ls='--',c='black')
    plt.xlabel("Trials")
    plt.ylabel("Normalized weights")
    plt.title("Transition Hidden Units "+param)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

TT0=23
TT1=28
lC10=-5
lC11=5
HU_structure(W1_pca,Whz,sort_pos,lC10,lC11,TT0,TT1,"W1")

lC10=-5
lC11=5
HU_structure(W2_pca,Whz2,sort_pos2,lC10,lC11,TT0,TT1,"W2")

#%% Limiting behaviour stability

# Input Limit Behaviour
Warm_time=500000
Input_channels= NeuronPattern["Training_Input"][0].shape[1]
Length_data=1000000
# Generation of free trajectories for limiting behaviour - SNAPSHOT
ModelLB=[]
for w_index in range(len(NeuronPattern["Training_Neuron"])):
    Inputs=tc.zeros(Length_data,Input_channels,dtype=tc.float32)
    data_trial=tc.from_numpy(NeuronPattern["Training_Neuron"][w_index]).float()                                             # tensor of neuronal data for initial trial data
    X, _ = m.generate_free_trajectory(data_trial,Inputs,Length_data,w_index)
    ModelLB.append(X)

#%% selection of neurons for visual representation
in1 = 7; in2=8
id_trial=[0,13,19,25,27,29,43,47,49]
fig,axs = plt.subplots(3,3,figsize=(20,25))
axs[0,0].plot(ModelLB[id_trial[0]][-5000:,in1],ModelLB[id_trial[0]][-5000:,in2])
axs[0,0].set_xlabel("Neuron "+str(in1+1))
axs[0,0].set_ylabel("Neuron "+str(in2+1))
axs[0,0].set_title("Trial "+str(id_trial[0]))
axs[0,1].plot(ModelLB[id_trial[1]][-5000:,in1],ModelLB[id_trial[1]][-5000:,in2])
axs[0,1].set_xlabel("Neuron "+str(in1+1))
axs[0,1].set_ylabel("Neuron "+str(in2+1))
axs[0,1].set_title("Trial "+str(id_trial[1]))
axs[0,2].plot(ModelLB[id_trial[2]][-5000:,in1],ModelLB[id_trial[2]][-5000:,in2])
axs[0,2].set_xlabel("Neuron "+str(in1+1))
axs[0,2].set_ylabel("Neuron "+str(in2+1))
axs[0,2].set_title("Trial "+str(id_trial[2]))

axs[1,0].plot(ModelLB[id_trial[3]][-5000:,in1],ModelLB[id_trial[3]][-5000:,in2])
axs[1,0].set_xlabel("Neuron "+str(in1+1))
axs[1,0].set_ylabel("Neuron "+str(in2+1))
axs[1,0].set_title("Trial "+str(id_trial[3]))
axs[1,1].plot(ModelLB[id_trial[4]][-5000:,in1],ModelLB[id_trial[4]][-5000:,in2])
axs[1,1].set_xlabel("Neuron "+str(in1+1))
axs[1,1].set_ylabel("Neuron "+str(in2+1))
axs[1,1].set_title("Trial "+str(id_trial[4]))
axs[1,2].plot(ModelLB[id_trial[5]][-5000:,in1],ModelLB[id_trial[5]][-5000:,in2])
axs[1,2].set_xlabel("Neuron "+str(in1+1))
axs[1,2].set_ylabel("Neuron "+str(in2+1))
axs[1,2].set_title("Trial "+str(id_trial[5]))

axs[2,0].plot(ModelLB[id_trial[6]][-5000:,in1],ModelLB[id_trial[6]][-5000:,in2])
axs[2,0].set_xlabel("Neuron "+str(in1+1))
axs[2,0].set_ylabel("Neuron "+str(in2+1))
axs[2,0].set_title("Trial "+str(id_trial[6]))
axs[2,1].plot(ModelLB[id_trial[7]][-5000:,in1],ModelLB[id_trial[7]][-5000:,in2])
axs[2,1].set_xlabel("Neuron "+str(in1+1))
axs[2,1].set_ylabel("Neuron "+str(in2+1))
axs[2,1].set_title("Trial "+str(id_trial[7]))
axs[2,2].plot(ModelLB[id_trial[8]][-5000:,in1],ModelLB[id_trial[8]][-5000:,in2])
axs[2,2].set_xlabel("Neuron "+str(in1+1))
axs[2,2].set_ylabel("Neuron "+str(in2+1))
axs[2,2].set_title("Trial "+str(id_trial[8]))
plt.show()

# %%


