# %% Import Libraries
import numpy as np
import torch as tc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import stats
from scipy.signal import decimate
from bptt.models import Model
from tqdm import tqdm
import os
import pickle
import torch.nn as nn
from typing import List
from function_modules import model_anafunctions as func
import pandas as pd
import matplotlib.patches as mpatches
import pandas as pd
from evaluation import klx_gmm as kl
import multiprocessing as mp
from mpl_toolkits import mplot3d
from evaluation import pse as ps
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from sklearn.decomposition import PCA
import ruptures as rpt
import random
from multiprocessing import Pool
import cebra
from cebra import CEBRA
plt.rcParams['font.size'] = 20
#%%
# FUNCTION Concatonate trials
def concatenate_list(lis,ax):
    res = lis[0]    # first element of the list
    Ls = [lis[0].shape[ax]] #first length of the list
    for i in range(1, len(lis)):
        Ls.append(lis[i].shape[ax])
        res = np.concatenate((res, lis[i]), axis=ax)
    return res, Ls

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

# FUNCTION Block Classification:
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
# %%

#data_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17/L6/Test1/datasets/' 
data_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17/L6/Test1/datasets/' 

#model_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/Model_Selected/'
model_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/Final_Models/OFC/CE17_L6_221008'

#model_name = 'Test_01_HU_256_l1_0.001_l2_64_l3_00_SL_400_encdim_64/001'
model_name = 'SubPopulation_HU_256_l1_0.001_l2_64_l3_00_SL_400_encdim_64/001'

mpath=os.path.join(model_path,model_name).replace('\\','/')

Hyper_mod(mpath,data_path)
# Load Training & Test Data
train_n,train_i = func.load_data(data_path,'Training')
test_n,test_i = func.load_data(data_path,'Test')
NeuronPattern={"Training_Neuron":train_n,"Training_Input":train_i,
               "Testing_Neuron":test_n,"Testing_Input":test_i}
RDataT = [tc.from_numpy(NeuronPattern["Training_Neuron"][w_index]).float() for w_index in range(len(NeuronPattern["Training_Neuron"]))]
# Load Metadata
file=open(os.path.join(data_path,'Metadata.pkl'),'rb')
Metadata=pickle.load(file)
file.close()

# Load Model
num_epochs = 199000
m = Model()
m.init_from_model_path(mpath, epoch=num_epochs)
m.eval()
# %% W parameters and its Correlation trial-trial 
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

#behaviour
path='D:/_work_cestarellas/Analysis/PLRNN/Session_Selected/OFC/CE17_L6'   # Pathway of the data (behaviour & Spike activity)
filename = "session.csv"
os.chdir(path)
# Load data
# Open the Behaviour file
BehData = pd.read_csv(filename)
# Classification of trials following the behabiour
GambleRewardTrials = BehData.index[(BehData['gamble']==1) & (BehData['REWARD']==1)]
GambleNoRewardTrials =  BehData.index[(BehData['gamble']==1) & (BehData['REWARD']==0)]
SafeRewardTrials = BehData.index[(BehData['safe']==1) & (BehData['REWARD']==1)]
SafeNoRewardTrials = BehData.index[(BehData['safe']==1) & (BehData['REWARD']==0)]
NoRespondingTrials = BehData.index[BehData["F"]==1]
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

algoB = rpt.Pelt(model="l2",min_size=5)
algoB.fit(DecisionNormalized[0:180])
BehaviourCP = algoB.predict(pen=1)
BehaviourCPModel=[4,25,52]

# FIGURE: Plot behaviour performance
plt.figure(figsize=(30,5))
rpt.show.display(DecisionNormalized,BehaviourCP, figsize=(10, 6))
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
plt.ylim([-0.2,1.2])
plt.yticks(ticks=[1.0,0.5,0.0])
plt.xlim([0,180])
plt.xlabel('Trials')
plt.ylabel("Animal's choice")
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#################### CEBRA Decoding #########################
model_trials = len(NeuronPattern["Training_Input"])
test_trials = len(NeuronPattern["Testing_Input"])
exp_trials = []
exp_testtrials = []
exp_traintrials = []

# Indicating when the cue appears. How many trials are concatenated?
for i in range(model_trials):
      exp_traintrials.append(np.sum(np.diff(NeuronPattern["Training_Input"][i][:,0])==1))
exp_trials=exp_traintrials

for i in range(test_trials):
      exp_testtrials.append(np.sum(np.diff(NeuronPattern["Testing_Input"][i][:,0])==1))
# Adding test trials values
for i in range(len(exp_testtrials)):
      exp_trials.insert(Metadata["Training2Test"][i],exp_testtrials[i])

# Determining the different block reward probabilities for each trial
# Save Reward Probabilities
Block_S_Prob=[0.9]
Block_S_Prob=Block_S_Prob*len(exp_trials)

# Gamble Reward Probabilities
trial_sum0=0
Block_G_Prob=[]
for i in range(len(exp_trials)):
        trial_sum0=trial_sum0+exp_trials[i]
        prob_pos = np.sum(trial_sum0>np.array(BlockTrials))-1
        Block_G_Prob.append(BlockProb[prob_pos])

# Probability of gamble and safe side for concatenated trials        
Block_Gamble_Prob =[Block_G_Prob[i] for i in range(len(exp_trials)) if i not in Metadata["TestTrials"]] 
Block_Safe_Prob=[Block_S_Prob[i] for i in range(len(exp_trials)) if i not in Metadata["TestTrials"]]
# Check length list
assert len(Block_Gamble_Prob)==len(NeuronPattern["Training_Neuron"])
assert len(Block_Gamble_Prob)==len(NeuronPattern["Training_Neuron"])

# Probability of decision: mean probability from the decision of the concatenated trials
Dec_Prob = []
pos_ini=0
for i in range(len(exp_trials)):
      pos_end=pos_ini+exp_trials[i]
      Dec_Prob.append(np.mean(DecisionNormalized[pos_ini:pos_end]))
      pos_ini=pos_end
Dec_Probability=[Dec_Prob[i] for i in range(len(exp_trials)) if i not in Metadata["TestTrials"]]

# Determining the duration of the action and the inter-trial interval
# Distribution of times obtained from the empirical trials
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

# Number of Decisions: 6
# Block 1: Save & Gamble
Gamble_Choice_b1 = np.array(Block_Gamble_Prob)==BlockProb[0]*2
Save_Choice_b1 = np.array(Block_Gamble_Prob)==BlockProb[0]*1
# Block 2: Save & Gamble
Gamble_Choice_b2 = np.array(Block_Gamble_Prob)==BlockProb[1]*4
Save_Choice_b2 = np.array(Block_Gamble_Prob)==BlockProb[1]*3
# Block 3: Save & Gamble
Gamble_Choice_b3 = np.array(Block_Gamble_Prob)==BlockProb[2]*6
Save_Choice_b3 = np.array(Block_Gamble_Prob)==BlockProb[2]*5

Gamble_Choice=Gamble_Choice_b1+Gamble_Choice_b2+Gamble_Choice_b3
Save_Choice=Save_Choice_b1+Save_Choice_b2+Save_Choice_b3

# Generation of Input data to generate new simulations and saving the choice of the side to be decoded
Input_Training = []
Side_Label = []
num_trialgeneration=500
for i in tqdm(range(len(NeuronPattern["Training_Input"]))):
        Input_Gene=np.zeros((0,3))
        Side_D = np.zeros((0,1))
        for k in range(num_trialgeneration):
                #Generation ITI bins
                if(np.isnan(iti_time_m[i])):
                      iti_time_m[i]=np.nanmean(iti_time_m)
                      iti_time_std[i]=np.nanmean(iti_time_std)
                iti_bins = np.abs(int(np.random.normal(iti_time_m[i],iti_time_std[i])))
                #Generation Action bins
                if(np.isnan(act_time_m[i])):
                      act_time_m[i]=np.nanmean(act_time_m)
                      act_time_std[i]=np.nanmean(act_time_std)
                action_bins = np.abs(int(np.random.normal(act_time_m[i],act_time_std[i])))
                # Generation Reward bins
                reward_bins = 25 # 50 ms
                # Total Trial Length
                trial_length=iti_bins+action_bins+reward_bins
                # Building Trial Vectors: Inputs and Side Choice
                Input_TrialGeneration = np.zeros((trial_length,3))
                Side_Decision = np.zeros((trial_length,1))
                Input_TrialGeneration[iti_bins:(iti_bins+action_bins),0]=1
                # Generation Random probabilities
                Decision_dice = np.random.random(size=1)    
                Reward_Prob = np.random.random(size=1)
                # Evaluation of choice and reward
                if Decision_dice<Dec_Probability[i]:
                        Side_Decision[0:(iti_bins+action_bins),0]=Gamble_Choice[i]
                        if Reward_Prob<Block_Gamble_Prob[i]:
                                Input_TrialGeneration[(iti_bins+action_bins):(iti_bins+action_bins+reward_bins),1]=4
                else:
                        Side_Decision[0:(iti_bins+action_bins),0]=Save_Choice[i]
                        if Reward_Prob<Block_Safe_Prob[i]:
                                Input_TrialGeneration[(iti_bins+action_bins):(iti_bins+action_bins+reward_bins),2]=1
                Input_Gene=np.concatenate((Input_Gene,Input_TrialGeneration),axis=0)
                Side_D = np.concatenate((Side_D,Side_Decision),axis=0)
        Input_Training.append(Input_Gene)
        Side_Label.append(Side_D)

#%% Generation of Data
Test_trial = 0
Train_before = Metadata["Training2Test"][Test_trial]
Train_after = Train_before+1

Ractivity = [NeuronPattern["Training_Neuron"][Train_before],NeuronPattern["Testing_Neuron"][Test_trial],NeuronPattern["Training_Neuron"][Train_after]]
Input = [NeuronPattern["Training_Input"][Train_before],NeuronPattern["Testing_Input"][Test_trial],NeuronPattern["Training_Input"][Train_after]]


AI_Neurons=[]
X_0, _ = m.generate_free_trajectory(tc.from_numpy(Ractivity[0]).float(),tc.from_numpy(Input_Training[0]).float(),Input_Training[0].shape[0],0)
AI_Neurons.append(X_0)
for i in tqdm(range(1,len(Input_Training))):
        X_0, _ = m.generate_free_trajectory(X_0[-1:,:].float(),tc.from_numpy(Input_Training[i]).float(),Input_Training[i].shape[0],i)
        AI_Neurons.append(X_0)

#%% Concatenate Neural Data & Labels
Training_Neurons,LS_Training_Neurons=concatenate_list(AI_Neurons,0)
Training_Labels,LS_Training_Labels=concatenate_list(Side_Label,0)
# Gerenetaion of tensors
Train_activity = tc.from_numpy(Training_Neurons).float() 
Train_label = tc.from_numpy(Training_Labels).float()

#%%
max_iterations = 5000 #default is 5000.
output_dimension = 32 #here, we set as a variable for hypothesis testing below.
#%% Creating model
cebra_target_model = CEBRA(model_architecture='offset10-model-mse',
                           batch_size=512,
                           learning_rate=5e-5,
                           temperature=0.01,
                           output_dimension=output_dimension,
                           max_iterations=max_iterations,
                           distance='euclidean',
                           conditional='time_delta',
                           device='cpu',
                           verbose=True,
                           time_offsets=10)


#%% Training Model
cebra_target_model.fit(Train_activity,
                       Train_label.numpy())
cebra_target = cebra_target_model.transform(Train_activity)

#%%
fig = plt.figure(figsize=(12, 5), dpi=100)
plt.suptitle('CEBRA-behavior trained with target label using MSE loss',
              fontsize=20)
ax = plt.subplot(121)
ax.set_title('All trials embedding', fontsize=20, y=-0.1)
x = ax.scatter(cebra_target[:, 0],
               cebra_target[:, 1],
               c=Train_label,
               cmap=plt.cm.hsv,
               s=0.05)
ax.axis('off')

ax = plt.subplot(122)
ax.set_title('Post-averaged by direction', fontsize=20, y=-0.1)
for i in range(3):
    direction_trial = (Train_label == i)
    trial_avg = cebra_target[direction_trial.squeeze(), :].reshape(-1, 600,
                                                         2).mean(axis=0)
    ax.scatter(trial_avg[:, 0],
               trial_avg[:, 1],
               color=plt.cm.hsv(1 / 3 * i),
               s=3)
ax.axis('off')
plt.show()
# #%%

# Side_decoder = cebra.KNNDecoder(n_neighbors=20, metric="cosine")
# Side_decoder.fit()
# %%
