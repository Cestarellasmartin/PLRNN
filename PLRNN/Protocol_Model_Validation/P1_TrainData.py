# %% Import Libraries
import os
import pickle
import numpy as np
import pandas as pd
import torch as tc
import matplotlib.pyplot as plt

#import matplotlib.patches as mpatches
from tqdm import tqdm

from bptt.models import Model
from evaluation import klx_gmm as kl
from evaluation import mse as ms
from evaluation import pse as ps
from function_modules import model_anafunctions as func

plt.rcParams['font.size'] = 20
#%%%%%%%%%%%%% FUNCTIONS

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

def Training_eval(m_pathway,run,data_path,NeuronPattern):
    mpath=os.path.join(m_pathway,run).replace('\\','/')
    Hyper_mod(mpath,data_path)
    #### Load model
    hyper = openhyper(mpath)
    save_files=os.listdir(mpath)
    save_models=[s for s in save_files if "model" in s]
    num_epochs = len(save_models)*hyper["save_step"]
    m = Model()
    m.init_from_model_path(mpath, epoch=num_epochs)

    At, _, _, _, _, _ = m.get_latent_parameters()
    # Transform tensor to numpy format
    A = At.detach().numpy()

    # General Parameters
    num_trials=len(NeuronPattern["Training_Neuron"])
    num_neurons=NeuronPattern["Training_Neuron"][0].shape[1]

    # Generate Latent states for Train Trials
    ModelS=[]
    for w_index in range(num_trials):
        data_trial=tc.from_numpy(NeuronPattern["Training_Neuron"][w_index]).float()          # tensor of neuronal data for initial trial data
        input_trial = tc.from_numpy(NeuronPattern["Training_Input"][w_index]).float()
        length_sim = input_trial.shape[0]
        X, _ = m.generate_free_trajectory(data_trial,input_trial,length_sim,w_index)
        ModelS.append(X[:,:])

    # Correlation between Train model trial and Data per neuron
    DT=[tc.from_numpy(NeuronPattern["Training_Neuron"][i]).float() for i in range(len(NeuronPattern["Training_Neuron"]))]
    N = ModelS[1].size(1)                                                                          # number of neurons
    NT = len(ModelS)
    rs = tc.zeros((N,NT))                                                                       # initialization of the correlation variable

    for nt in range(NT):
        eps = tc.randn_like(ModelS[nt]) * 1e-5                                                          # generation of little noise to avoid problems with silent neurons
        X_eps_noise = ModelS[nt] + eps                                                                  # adding noise to the signal 
        for n in range(N):
            rs[n,nt] = func.pearson_r(X_eps_noise[:, n], DT[nt][:, n])                                      # computation of the pearson correlation
    rs = rs.detach().numpy()

    MEAN_Corre=rs.mean()

    # Mean Square Error between model trial and Date per neuron
    n_steps=100
    val_mse = np.empty((n_steps,num_trials))
    for indices in range(num_trials):
        input= tc.from_numpy(NeuronPattern["Training_Input"][indices]).float()
        data = tc.from_numpy(NeuronPattern["Training_Neuron"][indices]).float()
        val_mse[:,indices] = ms.n_steps_ahead_pred_mse(m, data, input, n_steps, indices)
    MEAN_mse = np.mean(val_mse)

    # Kullback Leibler Divergence

    Model_Signal,_= func.concatenate_list(ModelS,0)
    Data_Signal,_= func.concatenate_list(NeuronPattern["Training_Neuron"],0)

    Dim_kl = int(np.floor(num_neurons/3))
    neu_list = np.array([1,2,3])
    kl_dim = np.ones([Dim_kl,1])*np.nan
    for j in range(Dim_kl):
        kl_dim[j] = kl.calc_kl_from_data(tc.tensor(Model_Signal[:,neu_list]),
                                            tc.tensor(Data_Signal[:,neu_list]))
        neu_list += 3

    MEAN_kl = kl_dim.mean()

    # Power Spectrum Error
    MEAN_pse,pse_list = ps.power_spectrum_error(tc.tensor(Model_Signal), tc.tensor(Data_Signal))

    # Checking divergence of the model by A parameter
    #True: The system will diverge at some point
    #False: The system will never diverge
    A_divergence=sum(A>1)>0

    # FIGURES
    plt.figure()
    for it in range(NT):
        plt.hist(rs[:,it],alpha=0.3)
    plt.xlabel("Corr(Model vs Data)")
    plt.ylabel("neurons")
    plt.title("Distribution Train Trials")
    plot_name=os.path.join(m_pathway,run+"_Train_Distr_Corr.png").replace('\\','/')
    plt.savefig(plot_name, bbox_inches='tight')

    Higher=np.array([np.where(rs[:,i]>0.4)[0].shape[0]/N for i in range(NT)])
    Lower=np.array([np.where(rs[:,i]<0.4)[0].shape[0]/N for i in range(NT)])
    Trials = [i for i in range(NT)]
    Ratio_neurons = {
        ">0.4": Higher,
        "<0.4": Lower,
    }
    width = 0.5
    fig, ax = plt.subplots()
    bottom = np.zeros(NT)
    for boolean, weight_count in Ratio_neurons.items():
        p = ax.bar(Trials, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count
    ax.set_title("Neuron Groups Train Trials")
    ax.set_ylabel("Ratio Neurons")
    ax.set_xlabel("Trials")
    lgd=ax.legend(title="Correlation",bbox_to_anchor=(1.1, 1.05))
    plot_name=os.path.join(m_pathway,run+"_Train_Distr_Corr_Trial.png").replace('\\','/')
    plt.savefig(plot_name, bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.figure()
    plt.plot(val_mse.mean(0))
    plt.xlabel("Trials")
    plt.ylabel("MSE")
    plot_name=os.path.join(m_pathway,run+"_Train_MSE.png").replace('\\','/')
    plt.savefig(plot_name, bbox_inches='tight')

    plt.figure()
    plt.plot(range(NT),rs.mean(0))
    plt.xlabel("Trials")
    plt.ylabel("Mean Correlation")
    plt.ylim([0,1])
    plot_name=os.path.join(m_pathway,run+"_Train_Correlation.png").replace('\\','/')
    plt.savefig(plot_name, bbox_inches='tight')

    plt.figure()
    plt.plot(range(Dim_kl),kl_dim)
    plt.xlabel("Subsets of 3 neurons")
    plt.ylabel("KLx")
    plot_name=os.path.join(m_pathway,run+"_Train_KLx.png").replace('\\','/')
    plt.savefig(plot_name, bbox_inches='tight')

    neu=np.random.choice(num_neurons,3,replace=False)
    ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
    ax.plot(Model_Signal[:,neu[0]], Model_Signal[:,neu[1]],
            Model_Signal[:,neu[2]], 'red',linestyle='dashed',label="Generated")
    ax.plot(Data_Signal[:,neu[0]], Data_Signal[:,neu[1]], 
            Data_Signal[:,neu[2]], 'blue',linestyle='dashed',label="Real")
    lgd=ax.legend()
    ax.set_xlabel('Neu 1',labelpad =15)
    ax.set_ylabel('Neu 2',labelpad =15)
    ax.set_zlabel('Neu 3',labelpad =15)
    ax.set_title('Training')
    plot_name=os.path.join(m_pathway,run+"_Train_PhasePlane.png").replace('\\','/')
    plt.savefig(plot_name, bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.figure()
    plt.plot(range(N),pse_list)
    plt.xlabel("Neurons")
    plt.ylabel("PSE")
    plot_name=os.path.join(m_pathway,run+"_Train_PSE.png").replace('\\','/')
    plt.savefig(plot_name, bbox_inches='tight')

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
    #Identification A_regularization
    Model_Hyper["AR"]=hyper["A_reg"]
    
    #Evaluation of the Model
    Model_Eval={}
    #Mean Correlation of the Session vs Model
    Model_Eval["Correlation"] = MEAN_Corre
    #Mean MSE testing in 100 sections for each trial
    Model_Eval["MSE"] = MEAN_mse
    #Mean PSE of the whole session. 
    Model_Eval["PSE"] = MEAN_pse
    #Mean Kullback leibler divergence of the whole session
    Model_Eval["KLx"] = MEAN_kl
    #Checking the A parameters to know the long-term behaviour of the model
    Model_Eval["Divergence"] = A_divergence
    
    return Model_Hyper,Model_Eval


#%%%% Main

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  LOAD DATA & MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% MAIN SCRIPT

############################################# Set Paths #######################################################
# Select Path for Data (Training and Test Trials)
data_path = 'D:\\_work_cestarellas\\Analysis\\PLRNN\\noautoencoder\\neuralactivity\\OFC\\JG15_190724\\datasets' 
# Select Path for Models (Folder containing the specific models to test)
model_path = 'D:\\_work_cestarellas\\Analysis\\PLRNN\\noautoencoder\\results\\Tuning_OFC_JG15_190724'
# Select Path for saving Data:
save_path = 'D:\\_work_cestarellas\\Analysis\\PLRNN\\noautoencoder\\results\\Tuning_OFC_JG15_190724\\Evaluation_Sheets'
# Select the name for the save file (session name):
save_name='JG15_190724'

############################################ Load data ##########################################################

# Load Training & Test Data
train_n,train_i = func.load_data(data_path,'Training')
Data_info={"Training_Neuron":train_n,"Training_Input":train_i}

######################################## Test measurements #######################################################

# Computation of testing measurements for the models in your model_path
model_list=next(os.walk(model_path))[1]
#Initialization of evaluations lists
Correlation=[]
PSE=[]
NMSE = []
KLx=[]
Div=[]
#Initialization of hyperparameter lists
Model_name=[]
RunNumber=[]
hidden=[]
lm1=[]
lm2=[]
lm3=[]
sl=[]
a_reg=[]

for i in tqdm(model_list,"Testing Models: "):
    pathway=os.path.join(model_path,i).replace('\\','/')
    runs=next(os.walk(pathway))[1] # taking only the folders with the models
    for j in runs:
        Hyper,Eval= Training_eval(pathway,j,data_path,Data_info)
        # List of evaluations
        NMSE.append(Eval["MSE"])
        Correlation.append(Eval["Correlation"])
        PSE.append(Eval["PSE"])
        KLx.append(Eval["KLx"])
        Div.append(Eval["Divergence"])
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
        #Activation of regulation A matrix
        a_reg.append(Hyper["AR"])


############################################### Saving ############################################################

# Saving Data as DataFrame
TrainData={"Models":Model_name,"Runs":RunNumber,
           "Hiddn_Units":hidden,"Sequence_Length":sl,"Regulation_A":a_reg,
           "Lambda1":lm1,"Lambda2":lm2,"Lambda3":lm3,
           "Correlation":Correlation,"NMSE":NMSE,"PSE":PSE,
           "KLx":KLx,"Divergence":Div
          }
Traindf=pd.DataFrame(TrainData)

# Check/Create Path
if os.path.exists(save_path):
    os.chdir(save_path)
else:
    os.makedirs(save_path)
    os.chdir(save_path)
save_file='TrainEvaluation_'+save_name+'.csv'
Traindf.to_csv(save_file,index=False)
# %%
