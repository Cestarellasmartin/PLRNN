# %% Import Libraries
#This is a test
import os
import pickle
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

data_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17/L6/Test0/datasets/' 
# Select Path for Models (Folder containing the specific models to test)
model_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/Tuning_OFC_CE17_L6_221008/Session_Test'

#%% Load Training & Test Data
Act,Inp = func.load_data(data_path,'Training')

#%% Loading models and simulations
model_name = 'CE17_Test0_01_HU_256_l1_0.001_l2_64_l3_00_SL_400_encdim_65/001'
mpath=os.path.join(model_path,model_name).replace('\\','/')


train_n,train_i = func.load_data(data_path,'Training')
test_n,test_i = func.load_data(data_path,'Test')
NeuronPattern={"Training_Neuron":train_n,"Training_Input":train_i,
               "Testing_Neuron":test_n,"Testing_Input":test_i}

# Load Model
num_epochs = 100000
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

































#%%
RDataT = [tc.from_numpy(NeuronPattern["Training_Neuron"][w_index]).float() for w_index in range(len(NeuronPattern["Training_Neuron"]))]
# Load Metadata
file=open(os.path.join(data_path,'Metadata.pkl'),'rb')
Metadata=pickle.load(file)
file.close()

model_trials = len(NeuronPattern["Training_Input"])
test_trials = len(NeuronPattern["Testing_Input"])
exp_trials=[]
exp_testtrials = []
exp_traintrials = []

def Block_Classification(GambleProbTrial):
# Main: Classification of the blocks with different probabilities reward in one side (default-->Gamble side)
# Input: 
#   GambleProbTrial: DataFrame Column with the probability of gamble side for each trial (pandas Series)
# Output:
#   BlockTrials: List of the trial number where the block probability change
#   BlockLabels: List of the Label for each block depending on the gamble probabilities (Low,Medium,High)

    GambleProbabilities = GambleProbTrial.sort_values().unique()
    NumberBlocks = GambleProbabilities.shape[0]  
    # Block Trial List
    BlockTrials = [] # List of trials where change the reward block probability. Initial trial for BlockLabels
    BlockLabels = [] # List of labels for each block
    Labels = ["Low","Medium","High"] # Default labels for each block:Low,Medium,High 

    # Check if it is Default Gambling Experiment
    assert NumberBlocks<=3, "This is not the defined Gambling Task "

    # Determining the Initial trial per each block of BlockLabels
    for i in range(NumberBlocks):
        BlockTrials.append(GambleProbTrial.index[GambleProbTrial==GambleProbabilities[i]][0])
        BlockLabels.append(Labels[i])

    return BlockTrials,BlockLabels   

BlockTrials,BlockLabels = Block_Classification(BehData["probability G"])
BlockProb = [0.125,0.25,0.75]
Blocksorted=np.argsort(BlockTrials)
BlockTrials=[BlockTrials[Blocksorted[0]],BlockTrials[Blocksorted[1]],BlockTrials[Blocksorted[2]]]
BlockProb=[BlockProb[Blocksorted[0]],BlockProb[Blocksorted[1]],BlockProb[Blocksorted[2]]]

# Smoothing the data for plotting
ScaleDecision=BehData["gamble"]+(BehData["gamble"]-1)
SigmaDecision=1
Binx =0.5
KernelWindow = np.arange(-3*SigmaDecision, 3*SigmaDecision, Binx)
KernelDecision = np.exp(-(KernelWindow/SigmaDecision)**2/2)
DecisionConvolution=np.convolve(ScaleDecision,KernelDecision,mode='same')
DecisionNormalized=(DecisionConvolution/np.nanmax(np.abs(DecisionConvolution))+1)/2


for i in range(model_trials):
      exp_traintrials.append(np.sum(np.diff(NeuronPattern["Training_Input"][i][:,0])==1))
exp_trials=exp_traintrials

for i in range(test_trials):
      exp_testtrials.append(np.sum(np.diff(NeuronPattern["Testing_Input"][i][:,0])==1))

for i in range(len(exp_testtrials)):
      exp_trials.insert(Metadata["Training2Test"][i],exp_testtrials[i])

trial_sum0=0
Block_G_Prob=[]
Block_S_Prob=[0.9]
Block_S_Prob=Block_S_Prob*len(exp_trials)
for i in range(len(exp_trials)):
        trial_sum0=trial_sum0+exp_trials[i]
        prob_pos = np.sum(trial_sum0>np.array(BlockTrials))-1
        Block_G_Prob.append(BlockProb[prob_pos])
Block_Gamble_Prob =[Block_G_Prob[i] for i in range(len(exp_trials)) if i not in Metadata["TestTrials"]] 
Block_Safe_Prob=[Block_S_Prob[i] for i in range(len(exp_trials)) if i not in Metadata["TestTrials"]]

Dec_Prob = []
pos_ini=0
for i in range(len(exp_trials)):
      pos_end=pos_ini+exp_trials[i]
      Dec_Prob.append(np.mean(DecisionNormalized[pos_ini:pos_end]))
      pos_ini=pos_end
Dec_Probability=np.array([Dec_Prob[i] for i in range(len(exp_trials)) if i not in Metadata["TestTrials"]])
Dec_Probability=np.where(Dec_Probability<0.9,Dec_Probability,0.9)

#Action Duration
act_time_m = []
act_time_std =[]
iti_time_m = []
iti_time_std =[]
for i in range(len(NeuronPattern["Training_Input"])):
        initial_act = np.where(np.diff(NeuronPattern["Training_Input"][i][:,0])==1)[0]
        final_act = np.where(np.diff(NeuronPattern["Training_Input"][i][:,0])==-1)[0]
        act_time_m.append(np.mean(final_act-initial_act+2))
        act_time_std.append(np.std(final_act-initial_act+2))
        iti_time_m.append(np.mean(initial_act[1:]-final_act[0:-1]-25))
        iti_time_std.append(np.std(initial_act[1:]-final_act[0:-1]-25))

Input_Training = []
Side_Label = []
Reward_Label=[]
virtual_trials=200
for i in tqdm(range(len(NeuronPattern["Training_Input"]))):
        Input_Gene=np.zeros((0,3))
        Side_D = np.zeros((virtual_trials,1))
        Reward_D = np.zeros((virtual_trials,1))
        for k in range(virtual_trials): #loop of virtual trials
                # generating ITI bins
                if(np.isnan(iti_time_m[i])):
                      iti_time_m[i]=np.nanmean(iti_time_m)
                      iti_time_std[i]=np.nanmean(iti_time_std)
                iti_bins = np.abs(int(np.random.normal(iti_time_m[i],iti_time_std[i])))
                # generating Action bins
                if(np.isnan(act_time_m[i])):
                      act_time_m[i]=np.nanmean(act_time_m)
                      act_time_std[i]=np.nanmean(act_time_std)
                action_bins = np.abs(int(np.random.normal(act_time_m[i],act_time_std[i])))
                # Generating Reward bins (Fixed value - 50 ms)
                reward_bins = 25
                # Defining total length of the virtual trial
                trial_length=iti_bins+action_bins+reward_bins
                # Initialization of External inputs for virtual trial
                Input_TrialGeneration = np.zeros((trial_length,3))
                # Initialization Trial parameters (Side decision, Reward obtained)
                Side_Decision = 0
                Reward_value = 0
                # Creating Cue External Input
                Input_TrialGeneration[iti_bins:(iti_bins+action_bins),0]=1
                # Creating Decision and Reward ouput + External inputs Safe_Reward & Gamble_Reward
                Decision_dice = np.random.random(size=1)
                Reward_Prob = np.random.random(size=1)
                    # Gamble Decision
                if Decision_dice<Dec_Probability[i]:
                        Side_Decision = 1 # Gamble side
                        if Reward_Prob<Block_Gamble_Prob[i]:
                                Input_TrialGeneration[(iti_bins+action_bins):(iti_bins+action_bins+reward_bins),1]=4
                                Reward_value = 1 # Obtains reward
                    # Safe decission
                else:
                        Side_Decision = 0 # Safe side
                        if Reward_Prob<Block_Safe_Prob[i]:
                                Input_TrialGeneration[(iti_bins+action_bins):(iti_bins+action_bins+reward_bins),2]=1
                                Reward_value=1 #Obtains reward
                Input_Gene=np.concatenate((Input_Gene,Input_TrialGeneration),axis=0)
                Side_D[k] = Side_Decision
                Reward_D[k] = Reward_value
        Input_Training.append(Input_Gene)
        Side_Label.append(Side_D)
        Reward_Label.append(Reward_D)
#%% Simulating Neurons with the model

AI_Neurons=[]
X_0, _ = m.generate_free_trajectory(tc.from_numpy(NeuronPattern["Training_Neuron"][0]).float(),tc.from_numpy(Input_Training[0]).float(),Input_Training[0].shape[0],0)
AI_Neurons.append(X_0)
for i in tqdm(range(1,len(Input_Training))):
        X_0, _ = m.generate_free_trajectory(X_0[-1:,:].float(),tc.from_numpy(Input_Training[i]).float(),Input_Training[i].shape[0],i)
        AI_Neurons.append(X_0)

#%%
#%%
Nseries,_=func.concatenate_list(AI_Neurons,0)
Iseries,_=func.concatenate_list(Input_Training,0)

Cue_ts=np.where(Iseries[:,0]==1)[0]
Limit_cue = np.where(np.diff(Cue_ts)>1)[0]
End_pos=Cue_ts[Limit_cue]+1
Start_pos=End_pos[:-1]+25
Start_pos=np.append(np.array(0),Start_pos)

num_neurons=Nseries.shape[1]
num_trials=Start_pos.shape[0]
# Data Set Mean Activity Zscore Neurons
X_data = np.zeros((num_trials,num_neurons))
for it in range(num_trials):
    X_data[it,:]=np.mean(Nseries[Start_pos[it]:End_pos[it],:],0)

# Classification of trials following the behabiour
Side_Signal,_=func.concatenate_list(Side_Label,0)
Reward_Signal,_=func.concatenate_list(Reward_Label,0)
GambleRewardTrials = ((Side_Signal==1) & (Reward_Signal==1))*1
GambleNoRewardTrials =  ((Side_Signal==1) & (Reward_Signal==0))*2
SafeRewardTrials = ((Side_Signal==0) & (Reward_Signal==1))*3
SafeNoRewardTrials = ((Side_Signal==0) & (Reward_Signal==0))*4
# Data To Fit:
# Next Decision (G,S)
ND=Side_Signal[:,0]
# Previous Decision (G,S)
PD = ND[:-1]
# Previous Reward (GR,GNR,SR,SNR)
PR=np.array(GambleRewardTrials[:-1]+GambleNoRewardTrials[:-1]+SafeRewardTrials[:-1]+SafeNoRewardTrials[:-1])
#%% Next Decisions
Sh_score=[]
score=[]
X_pre=X_data[:,:]
clf = LinearDiscriminantAnalysis()
for i in range(100):
    list_trials=np.linspace(0,X_pre.shape[0]-1,X_pre.shape[0]).astype(int)
    np.random.shuffle(list_trials)

    X=X_pre[list_trials,:]
    y=ND[list_trials]
    random_state=0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
        )
    np.random.shuffle(list_trials)
    y=ND[list_trials]
    random_state=50
    X_trainS, X_testS, y_trainS, y_testS = train_test_split(
        X, y, test_size=0.2,stratify=y, random_state=random_state
        )

    
    clf.fit(X_train,y_train)
    score.append(clf.score(X_test,y_test))
    
    clf.fit(X_trainS,y_trainS)
    Sh_score.append(clf.score(X_testS,y_testS))


fig, ax = plt.subplots()
Models = ['Data', 'Shuffle']
counts = [np.array(score).mean(),np.array(Sh_score).mean()]
bar_colors = ['tab:red', 'tab:blue']
ax.bar(Models, counts,  color=bar_colors)
ax.set_ylabel('Test Score')
ax.set_title('LDA-ND')
plt.show()

#%% Preious output
Sh_score=[]
score=[]
clf = LinearDiscriminantAnalysis()
X_pre=X_data[1:,:]

for i in range(100):
    list_trials=np.linspace(0,X_pre.shape[0]-1,X_pre.shape[0]).astype(int)
    np.random.shuffle(list_trials)

    X=X_pre[list_trials,:]
    y=PR[list_trials]
    random_state=0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
        )

    random_state=50
    X_trainS, X_testS, y_trainS, y_testS = train_test_split(
        X, y, test_size=0.2, random_state=random_state
        )

    
    clf.fit(X_train,y_train)
    score.append(clf.score(X_test,y_test))
    
    clf.fit(X_trainS,y_trainS)
    Sh_score.append(clf.score(X_testS,y_testS))


fig, ax = plt.subplots()
Models = ['Data', 'Shuffle']
counts = [np.array(score).mean(),np.array(Sh_score).mean()]
bar_colors = ['tab:red', 'tab:blue']
ax.bar(Models, counts,  color=bar_colors)
ax.set_ylabel('Test Score')
ax.set_title('LDA-PR')
plt.show()


# %%
