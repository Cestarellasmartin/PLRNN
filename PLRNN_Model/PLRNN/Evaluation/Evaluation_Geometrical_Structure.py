# %% Import Libraries
import os
import pickle
import numpy as np
import pandas as pd
import torch as tc
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
from tqdm import tqdm

import evaluation_Function_GSA as gsa
from bptt.models import Model
from function_modules import model_anafunctions as func

plt.rcParams['font.size'] = 20



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  LOAD DATA & MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#data_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17/L6/Test1/datasets/' 
data_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17/L6/Test_WholePop/datasets/' 

#model_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/Model_Selected/'
model_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/Final_Models/OFC/CE17_L6_221008'

#model_name = 'Test_01_HU_256_l1_0.001_l2_64_l3_00_SL_400_encdim_64/001'
model_name = 'Population_HU_512_l1_0.001_l2_08_l3_00_SL_400_encdim_155/001'

mpath=os.path.join(model_path,model_name).replace('\\','/')

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




#%% FUNCTION KULLBACK LEIBLER DIVERGENCE

#########################################################################################################
###################################  KULLBACK LEIBLER DIVERGENCE ########################################
#########################################################################################################

# General Measurement: 
# Differentation of KL divergence in the limiting behaviour for different regions of the behaviour:
# - Before Session
# - Inter-Trial Interval (ITI)
# - After Session

# This is computed by two methods: SnapShot & NonStationary

# Method I: SnapShot
# We generate a Time Series (Limiting behaviour - simulations after 50000 steps). The Time Series generated
# is a combination of the last section for each trial simulated. 
# The length of the last section is the total length of the Real data

# Method II: NonStationary
# Generation of the Time Series concatenating the output of each trial as initial point of the next one
# Then we generate a continious Time Series without external inputs to observe the dynamics of the system
# In this method, we also compute the KL for train and test data

############################################## METHOD I ################################################

modeltrials = len(NeuronPattern["Training_Neuron"])                     # Number of Trials                           
Length_trialB = Metadata["BeforeActivity"].shape[0]                     # Length for each trial simulated 
Length_trialA = Metadata["AfterActivity"].shape[0]                      # Length for each trial simulated 
Input_channels= NeuronPattern["Training_Input"][0].shape[1]             # Number of External Inputs

# Selection Session regions without external inputs
print("Obtaining Inter-Trial Intervals")
ITI_Trial=[]
num_traintrials = len(NeuronPattern["Training_Neuron"])
for i in tqdm(range(num_traintrials),"Obtaining Data"):
     pos=np.where(np.sum(NeuronPattern["Training_Input"][i],1)==0)
     ITI_Trial.append(NeuronPattern["Training_Neuron"][i][pos[0],:])
ZI_Signal,_= gsa.concatenate_list(ITI_Trial,0)
Length_trialITI = ZI_Signal.shape[0]                                    # Length for each trial simulated 

print("::: Simulating SnapShot data :::")
print("Before Session")
Before_Model = gsa.SnapShot_generation(NeuronPattern["Training_Neuron"],Length_trialB,Input_channels,modeltrials,m)
print("After Session")
After_Model = gsa.SnapShot_generation(NeuronPattern["Training_Neuron"],Length_trialA,Input_channels,modeltrials,m)
print("ITI Session")
ITI_Model = gsa.SnapShot_generation(NeuronPattern["Training_Neuron"],Length_trialITI,Input_channels,modeltrials,m)

assert Before_Model[0].shape[0]==Metadata["BeforeActivity"].shape[0]
assert After_Model[0].shape[0]==Metadata["AfterActivity"].shape[0]
assert ITI_Model[0].shape[0]==ZI_Signal.shape[0]

print('::: Computing KL distance ::: Limiting Behaviour :::')
Total_Neurons = NeuronPattern["Training_Neuron"][0].shape[1]                            # Total Number of Neurons
subspace_neurons = 20                                                                   # Number of Neurons to Compute KL divergence
Iteration_KL = 5                                                                            # Number of New Combinations for the subspace of neurons

# Generate List of the Simulations and the Real Data for KL_LimitBehaviour function
Data=[Before_Model,After_Model,ITI_Model,Metadata["BeforeActivity"],Metadata["AfterActivity"],ZI_Signal]
# Compute KL divergence

KL_LB = gsa.KL_LimitBehaviour(Data,Total_Neurons,modeltrials,subspace_neurons,Iteration_KL)
# Save KL divergence 
np.save("KL_trials",KL_LB)

# KL per Trial
Mean_KL = KL_LB.mean(2)
Sem_KL = KL_LB.std(2)/np.sqrt(Iteration_KL)
# Generate DataFrame to plot the Result
KL_df = pd.DataFrame({"Before":Mean_KL[:,0],"ITI":Mean_KL[:,1],"After":Mean_KL[:,2]})

############################################## METHOD II ################################################

# Determining lenght of the simulation per trial
Length_trialB_NS = round(Metadata["BeforeActivity"].shape[0]/modeltrials)                     # Length for each trial simulated 
Length_trialA_NS = round(Metadata["AfterActivity"].shape[0]/modeltrials)                      # Length for each trial simulated 

 # Generation of free trajectories - NON-STATIONARY
print("::: Generation NonStationary ::: Before Session")
Before_Signal = gsa.NonStationary_generation(NeuronPattern["Training_Neuron"],Length_trialB_NS,Input_channels,modeltrials,m)

print("::: Generation NonStationary ::: After Session")
After_Signal = gsa.NonStationary_generation(NeuronPattern["Training_Neuron"],Length_trialA_NS,Input_channels,modeltrials,m)

# Simulating Test and Train Data
print("::: Generation Train Data :::")
Train_Signal,_ = gsa.Train_generation(NeuronPattern["Training_Neuron"],NeuronPattern["Training_Input"],modeltrials,m) 

print("::: Generation Test Data :::")
Test_Signal,_ = gsa.Test_generation(Metadata,NeuronPattern["Testing_Neuron"],NeuronPattern["Testing_Input"],m)

# Selecting Real data with the same length than the generated one
Before_Real = Metadata["BeforeActivity"][:Before_Signal.shape[0],:]
After_Real = Metadata["AfterActivity"][:After_Signal.shape[0],:]
# Concatenating trials of real data to generate one array of the whole time serie
Train_Real,_ = gsa.concatenate_list(NeuronPattern["Training_Neuron"],0)
Test_Real,_ = gsa.concatenate_list(NeuronPattern["Testing_Neuron"],0)

# Generate List of the Simulations and the Real Data for TimeSerie_KL_distance function
Data = [Before_Signal,After_Signal,Train_Signal,Test_Signal,
        Before_Real,After_Real,Train_Real,Test_Real]

# Compute KL divergence for Time Series
Series_KL = gsa.TimeSerie_KL_distance(Data,Total_Neurons,subspace_neurons,Iteration_KL)

# Generate DataFrame to plot the Result
Ser_KLdf = pd.DataFrame({"Before":Series_KL[:,0],"After":Series_KL[:,1],"Train":Series_KL[:,2],"Test":Series_KL[:,3]})

############################################## FIGURES ################################################

# Snapshot Limiting Behaviour Per Trial
plt.figure()
plt.plot(Mean_KL[:,0],label="Before")
plt.plot(Mean_KL[:,1],label="ITI")
plt.plot(Mean_KL[:,2],label="After")
plt.title("SnapShot")
plt.ylabel("KL Divergence")
plt.xlabel("Trials")
plt.legend()
plt.show()

# Mean Value: Snapshot Limiting Behaviour Per Trial
plt.figure()
ax=KL_df.boxplot(color='blue',figsize=(5,5))
ax.set_ylabel("KL divergence")
ax.set_title("SnapShot")
plt.show()

# NonStationary Limiting behaviour and Train and Test Data
plt.figure()
ax=Ser_KLdf.boxplot(color='blue',figsize=(5,5))
ax.set_ylabel("KL divergence")
ax.set_title("NonStationary Time Series")
plt.show()

# %%
