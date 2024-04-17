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

def LongTerm_eval(m_pathway,run,data_path,NeuronPattern):
    # Selection Session regions without external inputs. Obtaining Inter-Trial Intervals
    ITI_Trial=[]
    num_traintrials = len(NeuronPattern["Training_Neuron"])
    for i in tqdm(range(num_traintrials),"Obtaining Data"):
        pos=np.where(np.sum(NeuronPattern["Training_Input"][i],1)==0)
        ITI_Trial.append(NeuronPattern["Training_Neuron"][i][pos[0],:])
    ZI_Signal,_= func.concatenate_list(ITI_Trial,0)
    Length_trialITI = ZI_Signal.shape[0]                                                                                        # Length for each trial simulated 
    length_ITI = [len(ITI_Trial[i]) for i in range(len(ITI_Trial))]

    mpath=os.path.join(m_pathway,run).replace('\\','/')
    Hyper_mod(mpath,data_path)
    # Load model
    hyper = openhyper(mpath)
    save_files=os.listdir(mpath)
    save_models=[s for s in save_files if "model" in s]
    num_epochs = len(save_models)*hyper["save_step"]
    m = Model()
    m.init_from_model_path(mpath, epoch=num_epochs)
    
    # Input Limit Behaviour
    Warm_time=500000
    Input_channels= NeuronPattern["Training_Input"][0].shape[1]
    # Generation of free trajectories for limiting behaviour - SNAPSHOT
    ModelSS=[]
    for w_index in range(len(NeuronPattern["Training_Neuron"])):
        Length_data=ITI_Trial[w_index].shape[0]+Warm_time
        Inputs=tc.zeros(Length_data,Input_channels,dtype=tc.float32)
        data_trial=tc.from_numpy(NeuronPattern["Training_Neuron"][w_index]).float()                                             # tensor of neuronal data for initial trial data
        X, _ = m.generate_free_trajectory(data_trial,Inputs,Length_data,w_index)
        ModelSS.append(X[Warm_time:,:])

    # Generation of free trajectories - NON-STATIONARY 
    ModelNS=[]
    total_neu = NeuronPattern["Training_Neuron"][0].shape[1]
    data_trial=tc.from_numpy(NeuronPattern["Training_Neuron"][0][-1,:]).float()                                                 # tensor of neuronal data for initial trial data
    Input_channels= NeuronPattern["Training_Input"][0].shape[1]
    Inputs=[tc.zeros(ITI_Trial[i].shape[0],Input_channels,dtype=tc.float32) for i in range(len(NeuronPattern["Training_Neuron"]))]

    for w_index in range(len(NeuronPattern["Training_Neuron"])):
        data_trial = tc.reshape(data_trial,(1,total_neu))
        X, _ = m.generate_free_trajectory(data_trial,Inputs[w_index],ITI_Trial[w_index].shape[0],w_index)
        data_trial=X[-1,:]
        ModelNS.append(X)
    # Concatenate Trials
    NS_Signal,_ = func.concatenate_list(ModelNS,0)
    SS_Signal,_ = func.concatenate_list(ModelSS,0)

    # General Parameters
    num_neurons=NeuronPattern["Training_Neuron"][0].shape[1]

    # Snapshot
    # pse
    MEAN_pse_ss,_ = ps.power_spectrum_error(tc.tensor(SS_Signal), tc.tensor(ZI_Signal))

    # Kullback Leibler Divergence
    Dim_kl = int(np.floor(num_neurons/3))
    neu_list = np.array([1,2,3])
    kl_dim = np.ones([Dim_kl,1])*np.nan
    for j in range(Dim_kl):
        kl_dim[j] = kl.calc_kl_from_data(tc.tensor(SS_Signal[:,neu_list]),
                                            tc.tensor(ZI_Signal[:,neu_list]))
        neu_list += 3
    MEAN_kl_ss = kl_dim.mean()

    #Non-Stationary simulation
    # pse
    MEAN_pse_ns,_ = ps.power_spectrum_error(tc.tensor(NS_Signal), tc.tensor(ZI_Signal))

    # Kullback Leibler Divergence
    Dim_kl = int(np.floor(num_neurons/3))
    neu_list = np.array([1,2,3])
    kl_dim = np.ones([Dim_kl,1])*np.nan
    for j in range(Dim_kl):
        kl_dim[j] = kl.calc_kl_from_data(tc.tensor(NS_Signal[:,neu_list]),
                                            tc.tensor(ZI_Signal[:,neu_list]))
        neu_list += 3
    MEAN_kl_ns = kl_dim.mean()

    #Hyper-Parameters of the Model
    Model_Hyper={}
    #Identification Hidden Units
    Model_Hyper["HU"] = hyper['dim_hidden']
    #Identification Parameter Lambda 1
    Model_Hyper["L1"] = hyper['reg_lambda1'][0]
    #Identification Parameter Lambda 2
    Model_Hyper["L2"] = hyper['reg_lambda2'][0]
    #Identification Parameter Lambda 2
    Model_Hyper["L3"] = hyper['reg_lambda3'][0]
    #Identification Sequence Length
    Model_Hyper["SL"] = hyper['seq_len']
    
    #Evaluation of the Model
    Model_Eval={}
    #Mean Correlation of the Session vs Model
    Model_Eval["PSE_SS"] = MEAN_pse_ss
    #Mean MSE testing in 100 sections for each trial
    Model_Eval["PSE_NS"] = MEAN_pse_ns
    #Mean PSE of the whole session. 
    Model_Eval["KLx_SS"] = MEAN_kl_ss
    #Mean Kullback leibler divergence of the whole session
    Model_Eval["KLx_NS"] = MEAN_kl_ns

    return Model_Hyper,Model_Eval


#%% MAIN SCRIPT

############################################# Set Paths #######################################################
# Select Path for Data (Training and Test Trials)
data_path = 'D:\\_work_cestarellas\\Analysis\\PLRNN\\noautoencoder\\neuralactivity\\OFC\\CE17\\L6\\Test0\\datasets' 
# Select Path for Models (Folder containing the specific models to test)
model_path = 'D:\\_work_cestarellas\\Analysis\\PLRNN\\noautoencoder\\results\\Tuning_OFC_CE17_221008'
# Select Path for saving Data:
save_path = 'D:\\_work_cestarellas\\Analysis\\PLRNN\\noautoencoder\\results\\Tuning_OFC_CE17_221008\\Evaluation_Sheets'
# Select the name for the save file (session name):
save_name='CE17_221008'

############################################ Load data ##########################################################

# Load Training & Test Data
train_n,train_i = func.load_data(data_path,'Training')
Data_info={"Training_Neuron":train_n,"Training_Input":train_i}

######################################## Test measurements #######################################################

# Computation of testing measurements for the models in your model_path
model_list=next(os.walk(model_path))[1]
#Initialization of evaluations lists
PSE_S = []
PSE_N = []
KLx_S = []
KLx_N = []
#Initialization of hyperparameter lists
Model_name=[]
RunNumber=[]
hidden=[]
lm1=[]
lm2=[]
lm3=[]
sl=[]

for i in tqdm(model_list,"Testing Models: "):
    pathway=os.path.join(model_path,i).replace('\\','/')
    runs=next(os.walk(pathway))[1] # taking only the folders with the models
    for j in runs:
        Hyper,Eval= LongTerm_eval(pathway,j,data_path,Data_info)
        # List of evaluations
        PSE_S.append(Eval["PSE_SS"])
        PSE_N.append(Eval["PSE_NS"])
        KLx_S.append(Eval["KLx_SS"])
        KLx_N.append(Eval["KLx_NS"])
        # List of Hyper-parameters
        # Folder's name of the model
        Model_name.append(i)
        # Number of the run
        RunNumber.append(j)
        #Identification Hidden Units
        hidden.append(Hyper["HU"])
        #Identification Parameter Lambda 1
        lm1.append(Hyper["L1"])
        #Identification Parameter Lambda 2
        lm2.append(Hyper["L2"])
        #Identification Parameter Lambda 3
        lm3.append(Hyper["L3"])
        #Identification Sequence Length
        sl.append(Hyper["SL"])


############################################### Saving ############################################################

# Saving Data as DataFrame
LimitData={"Models":Model_name,"Runs":RunNumber,
           "Hiddn_Units":hidden,"Sequence_Length":sl,
           "Lambda1":lm1,"Lambda2":lm2,"Lambda3":lm3,
           "PSE_SS":PSE_S,"PSE_NS":PSE_N,"KLx_SS":KLx_S,
           "KLx_NS":KLx_N
          }
Limitdf=pd.DataFrame(LimitData)

# Check/Create Path
if os.path.exists(save_path):
    os.chdir(save_path)
else:
    os.makedirs(save_path)
    os.chdir(save_path)
save_file='LimitingBehaviour_'+save_name+'.csv'
Limitdf.to_csv(save_file,index=False)
# %%
