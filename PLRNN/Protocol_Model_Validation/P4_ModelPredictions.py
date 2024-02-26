# %% Import Libraries
#This is a test
import os
import scipy.io
import numpy as np
import pandas as pd
import torch as tc
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

import matplotlib.patches as mpatches
from tqdm import tqdm
from bptt.models import Model
from function_modules import model_anafunctions as func

plt.rcParams['font.size'] = 20

#%% Load Recordings

data_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/JG15_190724/datasets/' 
# Select Path for Models (Folder containing the specific models to test)
model_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/Tuning_OFC_JG15_190724'

#%% Load Training & Test Data
Act,Inp = func.load_data(data_path,'Training')

#%% Loading models and simulations
model_name = 'JG15_01_HU_256_l1_0.001_l2_08_l3_00_SL_400_encdim_74/001'
mpath=os.path.join(model_path,model_name).replace('\\','/')


train_n,train_i = func.load_data(data_path,'Training')
test_n,test_i = func.load_data(data_path,'Test')
NeuronPattern={"Training_Neuron":train_n,"Training_Input":train_i,
               "Testing_Neuron":test_n,"Testing_Input":test_i}

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

#%%
Nseries,_=func.concatenate_list(Act,0)
Iseries,_=func.concatenate_list(Inp,0)
Mseries,_=func.concatenate_list(ModelS,0)


Cue_ts=np.where(Iseries[:,0]==1)[0]
Limit_cue = np.where(np.diff(Cue_ts)>1)[0]
End_pos=Cue_ts[Limit_cue]+1
Start_pos=End_pos[:-1]+25
Start_pos=np.append(np.array(0),Start_pos)

num_neurons=Act[0].shape[1]
num_trials=Start_pos.shape[0]
# Data Set Mean Activity Zscore Neurons
X_data = np.zeros((num_trials,num_neurons))
X_model = np.zeros((num_trials,num_neurons))
for it in range(num_trials):
    X_data[it,:]=np.mean(Nseries[Start_pos[it]:End_pos[it],:],0)
    X_model[it,:]=np.mean(Mseries[Start_pos[it]:End_pos[it],:],0)
    
#%% Behaviour
path='D:/_work_cestarellas/Analysis/PLRNN/session_selection_v0/OFC/CE17/L6'   # Pathway of the data (behaviour & Spike activity)
filename = "session.csv"
os.chdir(path)
# Load data
# Open the Behaviour file
BehData = pd.read_csv(filename)
# Classification of trials following the behabiour
GambleRewardTrials = ((BehData['gamble']==1) & (BehData['REWARD']==1))*1
GambleNoRewardTrials =  ((BehData['gamble']==1) & (BehData['REWARD']==0))*2
SafeRewardTrials = ((BehData['safe']==1) & (BehData['REWARD']==1))*3
SafeNoRewardTrials = ((BehData['safe']==1) & (BehData['REWARD']==0))*4
# Data To Fit:
# Next Decision (G,S)
ND_se = (BehData['gamble']==1)*1+(BehData['safe']==1)*0
ND=np.array(ND_se[:num_trials])
# Previous Decision (G,S)
PD = ND[:-1]
# Previous Reward (GR,GNR,SR,SNR)
PR=np.array(GambleRewardTrials[:num_trials-1]+GambleNoRewardTrials[:num_trials-1]+SafeRewardTrials[:num_trials-1]+SafeNoRewardTrials[:num_trials-1])

#%% Behaviour No Dimitris
path='D:\\_work_cestarellas\\Analysis\\PLRNN\\Session_Selected\\OFC\\JG15_190724_clustered'   # Pathway of the data (behaviour & Spike activity)
filename = "JG15_190724_Behaviour.mat"
os.chdir(path)
# Load data
# Open the Behaviour file
BehData = scipy.io.loadmat(filename)
BehData=BehData['Trials_Sync']
# Classification of trials following the behabiour
GambleRewardTrials = ((BehData[:,12]==1) & (BehData[:,13]==1))*1
GambleNoRewardTrials =  ((BehData[:,12]==1) & (BehData[:,13]==0))*2
SafeRewardTrials = ((BehData[:,12]==0) & (BehData[:,13]==1))*3
SafeNoRewardTrials = ((BehData[:,12]==0) & (BehData[:,13]==0))*4
# Data To Fit:
# Next Decision (G,S)
ND_se = (BehData[:,12]==1)*1+(BehData[:,12]==0)*0
ND=np.array(ND_se[:num_trials])
# Previous Decision (G,S)
PD = ND[:-1]
# Previous Reward (GR,GNR,SR,SNR)
PR=np.array(GambleRewardTrials[:num_trials-1]+GambleNoRewardTrials[:num_trials-1]+SafeRewardTrials[:num_trials-1]+SafeNoRewardTrials[:num_trials-1])

#%% Next Decisions
score_data=[]
score_model=[]
score_modelSh=[]
score_dataSh=[]

clf = LinearDiscriminantAnalysis()
for i in range(1000):
    list_trials=np.linspace(0,X_data.shape[0]-1,X_data.shape[0]).astype(int)
    np.random.shuffle(list_trials)
    # Classifying data
    X=X_data[list_trials,:]
    y=ND[list_trials]
    random_state=0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
        )
    clf.fit(X_train,y_train)
    score_data.append(clf.score(X_test,y_test))

    # Classifying model
    Xm=X_model[list_trials,:]
    random_state=0
    X_trainM, X_testM, y_trainM, y_testM = train_test_split(
        Xm, y, test_size=0.2, stratify=y, random_state=0
        )
    random_state=0
    clf.fit(X_trainM,y_trainM)
    score_model.append(clf.score(X_testM,y_testM))
    
    # Classifying data Shuffle
    list_trials=np.linspace(0,X_data.shape[0]-1,X_data.shape[0]).astype(int)
    y=ND[list_trials]
    random_state=0
    X_trainS, X_testS, y_trainS, y_testS = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
        )

    clf.fit(X_trainS,y_trainS)
    score_dataSh.append(clf.score(X_testS,y_testS))
    # Classifying model Shuffle
    X_trainMS, X_testMS, y_trainMS, y_testMS = train_test_split(
        Xm, y, test_size=0.2, stratify=y, random_state=0
        )
    clf.fit(X_trainMS,y_trainMS)
    score_modelSh.append(clf.score(X_testMS,y_testMS))


fig, ax = plt.subplots()
Models = ['Data', 'Shuffle','Model','Shuffle']
ax.boxplot([score_data,score_dataSh,score_model,score_modelSh],labels=Models)
ax.set_ylabel('Test Score')
ax.set_title('LDA-ND')
plt.show()
#%% Previouos Decision
score_data=[]
score_model=[]
score_modelSh=[]
score_dataSh=[]
X_dp=X_data[1:,:]
X_mp=X_model[1:,:]

clf = LinearDiscriminantAnalysis()
for i in range(1000):
    list_trials=np.linspace(0,X_dp.shape[0]-1,X_dp.shape[0]).astype(int)
    np.random.shuffle(list_trials)
    # Classifying data
    X=X_dp[list_trials,:]
    y=PD[list_trials]
    random_state=0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
        )
    clf.fit(X_train,y_train)
    score_data.append(clf.score(X_test,y_test))

    # Classifying model
    Xm=X_mp[list_trials,:]
    random_state=0
    X_trainM, X_testM, y_trainM, y_testM = train_test_split(
        Xm, y, test_size=0.2, stratify=y, random_state=0
        )
    random_state=0
    clf.fit(X_trainM,y_trainM)
    score_model.append(clf.score(X_testM,y_testM))
    
    # Classifying data Shuffle
    list_trials=np.linspace(0,X_dp.shape[0]-1,X_dp.shape[0]).astype(int)
    y=PD[list_trials]
    random_state=0
    X_trainS, X_testS, y_trainS, y_testS = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
        )

    clf.fit(X_trainS,y_trainS)
    score_dataSh.append(clf.score(X_testS,y_testS))
    # Classifying model Shuffle
    X_trainMS, X_testMS, y_trainMS, y_testMS = train_test_split(
        Xm, y, test_size=0.2, stratify=y, random_state=0
        )
    clf.fit(X_trainMS,y_trainMS)
    score_modelSh.append(clf.score(X_testMS,y_testMS))


fig, ax = plt.subplots()
Models = ['Data', 'Shuffle','Model','Shuffle']
ax.boxplot([score_data,score_dataSh,score_model,score_modelSh],labels=Models)
ax.set_ylabel('Test Score')
ax.set_title('LDA-PD')
plt.show()


#%% Previouos Decision
score_data=[]
score_model=[]
score_modelSh=[]
score_dataSh=[]
X_dp=X_data[1:,:]
X_mp=X_model[1:,:]

clf = LinearDiscriminantAnalysis()
for i in range(1000):
    list_trials=np.linspace(0,X_dp.shape[0]-1,X_dp.shape[0]).astype(int)
    np.random.shuffle(list_trials)
    # Classifying data
    X=X_dp[list_trials,:]
    y=PR[list_trials]
    random_state=0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
        )
    clf.fit(X_train,y_train)
    score_data.append(clf.score(X_test,y_test))

    # Classifying model
    Xm=X_mp[list_trials,:]
    random_state=0
    X_trainM, X_testM, y_trainM, y_testM = train_test_split(
        Xm, y, test_size=0.2, stratify=y, random_state=0
        )
    random_state=0
    clf.fit(X_trainM,y_trainM)
    score_model.append(clf.score(X_testM,y_testM))
    
    # Classifying data Shuffle
    list_trials=np.linspace(0,X_dp.shape[0]-1,X_dp.shape[0]).astype(int)
    y=PR[list_trials]
    random_state=0
    X_trainS, X_testS, y_trainS, y_testS = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
        )

    clf.fit(X_trainS,y_trainS)
    score_dataSh.append(clf.score(X_testS,y_testS))
    # Classifying model Shuffle
    X_trainMS, X_testMS, y_trainMS, y_testMS = train_test_split(
        Xm, y, test_size=0.2, stratify=y, random_state=0
        )
    clf.fit(X_trainMS,y_trainMS)
    score_modelSh.append(clf.score(X_testMS,y_testMS))


fig, ax = plt.subplots()
Models = ['Data', 'Shuffle','Model','Shuffle']
ax.boxplot([score_data,score_dataSh,score_model,score_modelSh],labels=Models)
ax.set_ylabel('Test Score')
ax.set_title('LDA-PR')
plt.show()


# %%
