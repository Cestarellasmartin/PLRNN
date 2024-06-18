'''
**Script: Test_Trials
**Author: Cristian Estarellas Martin
**Date: 10/2023

**Description:
Computation of test measurements to determine overfitted and general models
computed for Training and Testing Trials. The Script will generate a 
DataFrame classifying the models with the following hyperparameters:
- Hidden Dimensions
- Sequence Length
- Lambda 1
- Lambda 2
In case of other classification for the DataFrame, the hyperparameters are stored in the variable hyper_f
You can reconstruct the DataFrame Structure as you wish (in you own version)

*Measurements (for Training and Testing Trials):
- Power Spectrum Error (PSE) between real and generated traces
- Distance Trial-Trial Correlation: Trial-Trial Correlation is the correlation of the Time Series between all trials
to provide the correlation between trial X and trial X+-Y. The Distance compute the similarity of the Trial-Trial Correlation
between real and generated neurons.
- Mean Square Error (MSE) between real and generated traces


*Inputs:
Path of your data, model and save folder:
- data_path
- model_path
- save_path

*Output:
Dataframe with the hyperparameters selected for the model and the Measurements explained.
- Default name: TestTrial_data.csv
'''
#%%  Import Libraries
import numpy as np
import torch as tc
import matplotlib.pyplot as plt
from bptt.models import Model
from tqdm import tqdm
import os
import pickle
import torch.nn as nn
from function_modules import model_anafunctions as func
from evaluation import pse as ps
from evaluation import mse as ms
import pandas as pd
from dtaidistance import dtw
plt.rcParams['font.size'] = 20

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
    full_name = open(os.path.join(mpath,'hypers.pkl').replace("\\","/"),"wb")                      # Name for training data
    pickle.dump(hyper,full_name)            # Save train data
    #close save instance 
    full_name.close()

def MSE_Training(NeuronPattern,m):
    n_steps=100
    number_trials = len(NeuronPattern["Training_Neuron"])
    val = np.empty((n_steps,number_trials))
    for indices in range(number_trials):
        input= tc.from_numpy(NeuronPattern["Training_Input"][indices]).float()
        data = tc.from_numpy(NeuronPattern["Training_Neuron"][indices]).float()
        val[:,indices] = ms.n_steps_ahead_pred_mse(m, data, input, n_steps, indices)
    return(np.mean(val))

def MSE_Testing(NeuronPattern,m):
    n_steps=100
    number_trials = len(NeuronPattern["Testing_Neuron"])
    val = np.empty((n_steps,number_trials))
    for indices in range(number_trials):
        input= tc.from_numpy(NeuronPattern["Testing_Input"][indices]).float()
        data = tc.from_numpy(NeuronPattern["Testing_Neuron"][indices]).float()
        val[:,indices] = ms.n_steps_ahead_pred_mse(m, data, input, n_steps, indices)
    return(np.mean(val))

def Trial_Correlation(RealData,ModelS):
    #num_trials = len(NeuronPattern["Training_Neuron"])
    RDataT = [tc.from_numpy(RealData[w_index]).float() for w_index in range(len(RealData))]
    num_trials = len(RealData)
    TrialCorrR = tc.empty((num_trials,num_trials))
    TrialCorrG = tc.empty((num_trials,num_trials))
    #Compute Correlation between Trials
    for i in tqdm(range(num_trials)):
        T1 = RDataT[i]
        for j in range(num_trials):
            T2 = RDataT[j]
            shape1 = T1.shape[0]
            shape2 = T2.shape[0]
            limit_points = shape1
            if shape1>shape2:
                    limit_points = shape2
            elif shape2>shape1:
                    limit_points = shape1
            TrialCorrR[i,j]=func.mean_trial_correlation(T1[:limit_points,:],T2[:limit_points,:])
            TrialCorrG[i,j]=func.mean_trial_correlation(ModelS[i][:limit_points,:],ModelS[j][:limit_points,:])

    #Generate Numpy Matrx
    CorrR = TrialCorrR.detach().numpy()
    CorrG = TrialCorrG.detach().numpy()
    Warping_Distance = []
    for i in range(num_trials):
            DataR = np.array([CorrR[i,j] for j in range(num_trials) if j!=i])
            DataG = np.array([CorrG[i,j] for j in range(num_trials) if j!=i])
            Warping_Distance.append(dtw.distance(DataR, DataG))
    #Mean Warping Distance of the Correlation 
    Correlation_Distance = np.array(Warping_Distance).mean()
    return Correlation_Distance

def Test_Trials(mpath,num_epochs,NeuronPattern,Metadata):
    
    print("::Loading Model::")
    m = Model()
    m.init_from_model_path(mpath, epoch=num_epochs)
    m.eval()

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
    
    # W Testing parameters
    print("::Generating W testing parameters::")
    print('Set of test trials: ',Metadata["TestTrials"])
    # Organisign test trial location in the Training Trials
    t_prev = [i for i in Metadata["Training2Test"]]
    t_post = [i+1 for i in Metadata["Training2Test"]]
    # Ouput Test Trial position
    print('trials before test trial: ',t_prev)
    print('trials after test trial: ',t_post)
    # Computing W matrices for test trials
    W2_test = np.empty((W2.shape[0],W2.shape[1],len(Metadata["TestTrials"])))
    W1_test = np.empty((W1.shape[0],W1.shape[1],len(Metadata["TestTrials"])))
    for i in range(len(t_prev)):
        W2_test[:,:,i] = (W2[:,:,t_prev[i]]+W2[:,:,t_post[i]])/2.0
        W1_test[:,:,i] = (W1[:,:,t_prev[i]]+W1[:,:,t_post[i]])/2.0
    
    #Generate Latent states for Test Trials (TT)
    TT = []
    #Generate Latent states for Test Trials
    W1_ind = [tc.from_numpy(W1_test[:,:,i]).float() for i in range(len(t_prev))]
    W2_ind = [tc.from_numpy(W2_test[:,:,i]).float() for i in range(len(t_prev))]
    test_tc = [tc.from_numpy(NeuronPattern["Testing_Neuron"][i]).float() for i in range(len(t_prev))]
    for i in range(len(W1_ind)):
        data_test=tc.from_numpy(NeuronPattern["Testing_Neuron"][i]).float()
        input_test=tc.from_numpy(NeuronPattern["Testing_Input"][i]).float()
        T0=int(len(NeuronPattern["Testing_Neuron"][i]))
        X, _ = m.generate_test_trajectory(data_test[0:11,:],W2_ind[i],W1_ind[i],input_test, T0,i)
        TT.append(X)

    #Generate Latent states for Train Trials
    ModelS=[]
    for w_index in tqdm(range(len(NeuronPattern["Training_Neuron"]))):
        data_trial=tc.from_numpy(NeuronPattern["Training_Neuron"][w_index]).float()          # tensor of neuronal data for initial trial data
        input_trial = tc.from_numpy(NeuronPattern["Training_Input"][w_index]).float()
        length_sim = input_trial.shape[0]
        X, _ = m.generate_free_trajectory(data_trial,input_trial,length_sim,w_index)
        ModelS.append(X[:,:])

    #Computing correlations between Trials
    Test_Corr_Dist = Trial_Correlation(NeuronPattern["Testing_Neuron"],TT)
    Train_Corr_Dist = Trial_Correlation(NeuronPattern["Training_Neuron"],ModelS)

    #Power Spectrum Error
    pse_test = []
    pse_testlist=[]
    for i in range(len(TT)):
        pse,pse_list = ps.power_spectrum_error(TT[i], test_tc[i])
        pse_test.append(pse)
        pse_testlist.append(pse_list)

    data_n=[tc.from_numpy(i).float() for i in NeuronPattern["Training_Neuron"]]
    data_i=[tc.from_numpy(i).float() for i in NeuronPattern["Training_Input"]]    
    pse_train, _ = func.trial_pse(data_n,data_i,m)

    PSE = (np.array(pse_train),np.array(pse_test))
    
    #Mean Square Error
    MSEtrain = MSE_Training(NeuronPattern,m)
    MSEtest = MSE_Testing(NeuronPattern,m)
    
    return Test_Corr_Dist, Train_Corr_Dist, PSE, MSEtrain, MSEtest

def Test_model_trials(ex_path,num_epochs,NeuronPattern,Metadata):
    print("::Loading Model::")
    m = Model()
    m.init_from_model_path(ex_path, epoch=num_epochs)
    m.eval()

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

    # W Testing parameters
    print("::Generating W testing parameters::")
    print('Set of test trials: ',Metadata["TestTrials"])

    t_prev = [i for i in Metadata["Training2Test"]]
    t_post = [i+1 for i in Metadata["Training2Test"]]

    print('trials before test trial: ',t_prev)
    print('trials after test trial: ',t_post)

    # Computing W matrices for test trials
    W2_test = np.empty((W2.shape[0],W2.shape[1],len(Metadata["TestTrials"])))
    W1_test = np.empty((W1.shape[0],W1.shape[1],len(Metadata["TestTrials"])))
    for i in range(len(t_prev)):
        W2_test[:,:,i] = (W2[:,:,t_prev[i]]+W2[:,:,t_post[i]])/2.0
        W1_test[:,:,i] = (W1[:,:,t_prev[i]]+W1[:,:,t_post[i]])/2.0
    #Generate Latent states

    TT = []
    #Generate Latent states
    W1_ind = [tc.from_numpy(W1_test[:,:,i]).float() for i in range(len(t_prev))]
    W2_ind = [tc.from_numpy(W2_test[:,:,i]).float() for i in range(len(t_prev))]
    test_tc = [tc.from_numpy(NeuronPattern["Testing_Neuron"][i]).float() for i in range(len(t_prev))]
    for i in range(len(W1_ind)):
        data_test=tc.from_numpy(NeuronPattern["Testing_Neuron"][i]).float()
        input_test=tc.from_numpy(NeuronPattern["Testing_Input"][i]).float()
        T0=int(len(NeuronPattern["Testing_Neuron"][i]))
        X, _ = m.generate_test_trajectory(data_test[0:11,:],W2_ind[i],W1_ind[i],input_test, T0,i)
        TT.append(X)
    return TT

def TestTrial_plot(Ractivity,Mactivity,Input,trialnum):
    vec_time=np.array([i*0.02 for i in range(Input[trialnum].shape[0])])                         # temporal vector for x axe
    sec_1= Ractivity[trialnum].shape[0]*0.02                                                      # time end of trial 1
    labneu=np.random.permutation(Ractivity[trialnum].shape[1])
    # Figure 6
    fig, axs = plt.subplots(8, figsize=(14, 10),sharex=True,sharey=False)
    for i in range(7):
        l1, =axs[i].plot(vec_time,Mactivity[trialnum][:,labneu[i]],c='r')
        l0, =axs[i].plot(vec_time,Ractivity[trialnum][:,labneu[i]],c='b')
        axs[i].set_ylabel('N'+str(labneu[i]))
    l2, =axs[7].plot(vec_time,Input[trialnum][:,0],c='purple')
    l3, =axs[7].plot(vec_time,Input[trialnum][:,1],c='g')
    l4, =axs[7].plot(vec_time,Input[trialnum][:,2],c='orange')
    axs[7].set_ylabel('INPUT')
    axs[7].axvline(sec_1, c='k')
    for a in axs:
        plt.setp(a.get_yticklabels(), visible=False)
        a.spines['top'].set_visible(False)
    plt.xlabel('Time (s)')
    plt.legend([l0,l1,l2,l3,l4],["Real Neuron","Generated Neuron","I1:Decision","I2:Gamble Reward","I3:Safe Reward"],bbox_to_anchor=(1,-0.8),ncol=3)
    axs[0].set_title("Test "+str(trialnum))
    plt.show()

#%% MAIN SCRIPT

############################################# Set Paths #######################################################
# Select Path for Data (Training and Test Trials)
data_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17/L6/Test0/datasets/' 
# Select Path for Models (Folder containing the specific models to test)
model_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/Tuning_OFC_CE17_L6_221008/A_reg/'
# Select Path for saving Data:
save_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/Tuning_OFC_CE17_L6_221008/A_reg'


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

# Computation of testing measurements for the models in your model_path
model_list=os.listdir(model_path)
num_epochs=100000        #select the last epoch generated by the model
CorrDistance=[]
PSE=[]
NMSE = []
Condition=[]
RunNumber=[]
Models=[]
hidden=[]
lm1=[]
lm2=[]
sl=[]

for i in tqdm(model_list):
    pathway=os.path.join(model_path,i)
    runs=os.listdir(pathway)
    for j in runs:
        mpath=os.path.join(pathway,j).replace('\\','/')
        Hyper_mod(mpath,data_path)
        hyper_f = openhyper(mpath)
        TestCorre,TrainCorre, PowerSE, MSEtrain, MSEtest = Test_Trials(mpath,num_epochs,NeuronPattern,Metadata)
        NMSE = NMSE + [MSEtrain,MSEtest]
        CorrDistance = CorrDistance + [TrainCorre,TestCorre]
        PSE=PSE+[PowerSE[0].mean(),PowerSE[1].mean()]
        #DataFrame Parameters
        Condition=Condition+["train","test"]
        RunNumber=RunNumber+[j,j]
        Models=Models+[i,i]
        hyper_f = openhyper(mpath)
        #Identification Hidden Units
        hidden_val = hyper_f['dim_hidden']
        hidden=hidden+[hidden_val,hidden_val]
        #Identification Parameter Lambda 1
        lm1_val = hyper_f['reg_lambda1'][0]
        lm1=lm1+[lm1_val,lm1_val]
        #Identification Parameter Lambda 2
        lm2_val = hyper_f['reg_lambda2'][0]
        lm2=lm2+[lm2_val,lm2_val]
        #Identification Sequence Length
        sl_val = hyper_f['seq_len']
        sl=sl+[sl_val,sl_val]

############################################### Saving ############################################################

# Saving Data as DataFrame
TestData={"Models":Models,"Runs":RunNumber,"Condition":Condition,"NMSE":NMSE,"CorrDistance":CorrDistance,"PSE":PSE,
          "Hidden":hidden,"Lambda1":lm1,"Lambda2":lm2,"SequenceLength":sl}
Testdf=pd.DataFrame(TestData)

# Check/Create Path
if os.path.exists(save_path):
    os.chdir(save_path)
else:
    os.makedirs(save_path)
    os.chdir(save_path)

Testdf.to_csv('TestTrial_JG15_190724_extra.csv',index=False)

# %%
