'''
**Script: Test_Trials
**Author: Cristian Estarellas Martin
**Date: 10/2023

**Description:
Computation of test measurements to determinethe generalisation of the model
The Script will generate a DataFrame classifying the models with the following hyperparameters:
- Hidden Dimensions
- Sequence Length
- Lambda 1
- Lambda 2
In case of other classification for the DataFrame, the hyperparameters are stored in the variable hyper_f
You can reconstruct the DataFrame Structure as you wish (in you own version)

*Simulations for limiting behaviour:
There are two kind of simulations:
- Snapshot:
- NonStationary:

*Mesurements:
- PSE for Snapshot
- PSE for Non-Stationary Simulation
- Kulback Leibar divergence

*Inputs:
Path of your data, model and save folder:
- data_path
- model_path
- save_path

*Output:
Dataframe with the hyperparameters selected for the model and the Measurements explained.
- Default name: LimitBehaviour_Model.csv
'''

# %% Import Libraries
import numpy as np
import torch as tc
import matplotlib.pyplot as plt
from bptt.models import Model
from tqdm import tqdm
import os
import pickle
import torch.nn as nn
from typing import List
from function_modules import model_anafunctions as func
from evaluation import pse as ps
import pandas as pd
import matplotlib.patches as mpatches
from evaluation import klx_gmm as kl
from evaluation import klx as kkl
import pandas as pd


#%% SPECIFIC FUNCTIONS
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
    full_name = open(os.path.join(mpath,'hypers.pkl').replace("\\","/"),"wb")                                                   # Name for training data
    pickle.dump(hyper,full_name)            # Save train data
    #close save instance 
    full_name.close()

def Test_limitPSE(mpath,num_epochs,NeuronPattern):
    # Selection Session regions without external inputs
    print("Obtaining Inter-Trial Intervals")
    ITI_Trial=[]
    num_traintrials = len(NeuronPattern["Training_Neuron"])
    for i in tqdm(range(num_traintrials),"Obtaining Data"):
        pos=np.where(np.sum(NeuronPattern["Training_Input"][i],1)==0)
        ITI_Trial.append(NeuronPattern["Training_Neuron"][i][pos[0],:])
    ZI_Signal,_= func.concatenate_list(ITI_Trial,0)
    Length_trialITI = ZI_Signal.shape[0]                                                                                        # Length for each trial simulated 
    length_ITI = [len(ITI_Trial[i]) for i in range(len(ITI_Trial))]

    m = Model()
    m.init_from_model_path(mpath, epoch=num_epochs)
    m.eval()
    # Input Limit Behaviour
    Warm_time=500000
    TimeLength=Length_trialITI
    Length_data=TimeLength+1+Warm_time
    Input_channels= NeuronPattern["Training_Input"][0].shape[1]
    Inputs=tc.zeros(Length_data,Input_channels,dtype=tc.float32)

    # Generation of free trajectories for limiting behaviour - SNAPSHOT
    ModelT=[]
    for w_index in range(len(NeuronPattern["Training_Neuron"])):
        data_trial=tc.from_numpy(NeuronPattern["Training_Neuron"][w_index]).float()                                             # tensor of neuronal data for initial trial data
        X, _ = m.generate_free_trajectory(data_trial,Inputs,Length_data,w_index)
        ModelT.append(X[Warm_time+1:,:])

    # Generation of free trajectories - NON-STATIONARY 
    NS_ModelT=[]
    total_neu = NeuronPattern["Training_Neuron"][0].shape[1]
    data_trial=tc.from_numpy(NeuronPattern["Training_Neuron"][0][-1,:]).float()                                                 # tensor of neuronal data for initial trial data
    Input_channels= NeuronPattern["Training_Input"][0].shape[1]
    Inputs=[tc.zeros(Length_trialITI[i],Input_channels,dtype=tc.float32) for i in range(len(NeuronPattern["Training_Neuron"]))]

    for w_index in range(len(NeuronPattern["Training_Neuron"])):
        data_trial = tc.reshape(data_trial,(1,total_neu))
        X, _ = m.generate_free_trajectory(data_trial,Inputs,Length_trialITI[i],w_index)
        data_trial=X[-1,:]
        NS_ModelT.append(X)
    # Concatenate Trials
    NS_Signal,Ls_NS_Signal=concatenate_list(NS_ModelT,0)

    # PSE Analysis
    print('::Computing SnapShot-PSE::')
    pse_limit = []
    pse_limitlist=[]
    for i in range(len(NeuronPattern["Training_Neuron"])):
        pse,pse_list = ps.power_spectrum_error(ModelT[i][:length_ITI[i],:], ITI_Trial[i])
        pse_limit.append(pse)
        pse_limitlist.append(pse_list)

    print('::Computing KL distance::')
    Dim_kl = int(np.floor(NeuronPattern["Training_Neuron"][0].shape[1]/3))
    val=np.zeros((len(NeuronPattern["Training_Neuron"]),1))
    for i in range(len(NeuronPattern["Training_Neuron"])):
        neu_list = np.array([1,2,3])
        kl_dim = np.ones([Dim_kl,1])*np.nan
        for j in range(Dim_kl):
            kl_dim[j] = kl.calc_kl_from_data(ModelT[i][:length_ITI[i],neu_list],
                                              tc.tensor(ITI_Trial[i][:,neu_list]))
            neu_list += 3
        val[i]=kl_dim.mean()

    print('::Computing NonStationary-PSE::')
    pse_ns = []
    pse_nslist = []
    pse_ns,pse_nslist = ps.power_spectrum_error(NS_Signal,ZI_Signal)

    return(np.array(pse_limit),np.array(pse_ns),val)


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
NeuronPattern={"Training_Neuron":train_n,"Training_Input":train_i,
               "Testing_Neuron":test_n,"Testing_Input":test_i}
# Load Metadata
file=open(os.path.join(data_path,'Metadata.pkl'),'rb')
Metadata=pickle.load(file)
file.close()

######################################## Test measurements #######################################################

# Computation of limitingbehaviour measurements for the models in your model_path
model_list=os.listdir(model_path)
num_epochs=199000                                                       #select the last epoch generated by the model

TimeLength=Metadata["BeforeActivity"].shape[0]
KL_trials=[]
KLdistance=[]
PSE_Nstationary=[]
PSE_Snapshot=[]
RunNumber=[]
Models=[]
hidden=[]
lm1=[]
lm2=[]
sl=[]

for i in tqdm(model_list):
    pathway=os.path.join(model_path,i)
    runs=os.listdir(pathway)
    for j in tqdm(runs):
        mpath=os.path.join(pathway,j).replace('\\','/')
        Hyper_mod(mpath,data_path)
        hyper_f = openhyper(mpath)
        PSE_SS,PSE_NS,KLd=Test_limitPSE(mpath,num_epochs,NeuronPattern)
        PSE_Nstationary = PSE_Nstationary + [PSE_NS]
        PSE_Snapshot=PSE_Snapshot+[PSE_SS.mean()]
        KL_trials.append(KLd)
        KLdistance = KLdistance + [KLd.mean()]
        #DataFrame Parameters
        RunNumber=RunNumber+[j]
        Models=Models+[i]
        hyper_f = openhyper(mpath)
        #Identification Hidden Units
        hidden_val = hyper_f['dim_hidden']
        hidden=hidden+[hidden_val]
        #Identification Parameter Lambda 1
        lm1_val = hyper_f['reg_lambda1'][0]
        lm1=lm1+[lm1_val]
        #Identification Parameter Lambda 2
        lm2_val = hyper_f['reg_lambda2'][0]
        lm2=lm2+[lm2_val]
        #Identification Sequence Length
        sl_val = hyper_f['seq_len']
        sl=sl+[sl_val]

############################################### Saving ############################################################

LimitData={"Models":Models,"Runs":RunNumber,"KLdist":KLdistance,"PSE_NS":PSE_Nstationary,"PSE_SS":PSE_Snapshot,
          "Hidden":hidden,"Lambda1":lm1,"Lambda2":lm2,"SequenceLength":sl}
Limitdf=pd.DataFrame(LimitData)

# Check/Create Path
if os.path.exists(save_path):
    os.chdir(save_path)
else:
    os.makedirs(save_path)
    os.chdir(save_path)

Limitdf.to_csv('LimitBehaviour_Population.csv',index=False)
# %%
