# %% Import Libraries
import os
import pickle
import numpy as np
import pandas as pd
import torch as tc
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import signal
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from bptt.models import Model
from evaluation import klx_gmm as kl
from evaluation import pse as ps
from evaluation import mse as ms
from function_modules import model_anafunctions as func

plt.rcParams['font.size'] = 20

#%%%% Functions
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

def Testing_eval(m_pathway,run,data_path,NeuronPattern,Metadata):
    mpath=os.path.join(m_pathway,run).replace('\\','/')
    Hyper_mod(mpath,data_path)
    #### Load model
    hyper = openhyper(mpath)
    save_files=os.listdir(mpath)
    save_models=[s for s in save_files if "model" in s]
    num_epochs = len(save_models)*hyper["save_step"]
    m = Model()
    m.init_from_model_path(mpath, epoch=num_epochs)

    _, W1t, W2t, _, _, Ct = m.get_latent_parameters()

    # Transform tensor to numpy format
    W2 = W2t.detach().numpy().transpose(1,2,0)
    W1 = W1t.detach().numpy().transpose(1,2,0)
    C = Ct.detach().numpy()
    # General Parameters
    Ntraining_trials=len(NeuronPattern["Training_Neuron"])
    Ntest_trials = len(NeuronPattern["Testing_Neuron"])
    num_neurons=NeuronPattern["Training_Neuron"][0].shape[1]

    # Generate Latent states for Test Trials
    # Identificating Test Trials in the training trial set
    t_prev = [i for i in Metadata["Training2Test"]]
    t_post = [i+1 for i in Metadata["Training2Test"]]

    # Computing W matrices for test trials
    W2_test = np.empty((W2.shape[0],W2.shape[1],len(Metadata["TestTrials"])))
    W1_test = np.empty((W1.shape[0],W1.shape[1],len(Metadata["TestTrials"])))
    for i in range(len(t_prev)):
        W2_test[:,:,i] = (W2[:,:,t_prev[i]]+W2[:,:,t_post[i]])/2.0
        W1_test[:,:,i] = (W1[:,:,t_prev[i]]+W1[:,:,t_post[i]])/2.0
    #Generate Latent states
    ModelT = []
    #Generate Latent states
    W1_ind = [tc.from_numpy(W1_test[:,:,i]).float() for i in range(len(t_prev))]
    W2_ind = [tc.from_numpy(W2_test[:,:,i]).float() for i in range(len(t_prev))]
    for i in range(len(W1_ind)):
        data_test=tc.from_numpy(NeuronPattern["Testing_Neuron"][i]).float()
        input_test=tc.from_numpy(NeuronPattern["Testing_Input"][i]).float()
        T0=NeuronPattern["Testing_Neuron"][i].shape[0]
        X, _ = m.generate_test_trajectory(data_test[0:11,:],W2_ind[i],W1_ind[i],input_test, T0,i)
        ModelT.append(X)
    
    # Correlation between Train model trial and Data per neuron
    DT=[tc.from_numpy(NeuronPattern["Testing_Neuron"][i]).float() for i in range(len(NeuronPattern["Testing_Neuron"]))]
    rs = tc.zeros((num_neurons,Ntest_trials))                                                                       # initialization of the correlation variable

    for nt in range(Ntest_trials):
        eps = tc.randn_like(ModelT[nt]) * 1e-5                                                          # generation of little noise to avoid problems with silent neurons
        X_eps_noise = ModelT[nt] + eps                                                                  # adding noise to the signal 
        for n in range(num_neurons):
            rs[n,nt] = func.pearson_r(X_eps_noise[:, n], DT[nt][:, n])                                      # computation of the pearson correlation
    rs = rs.detach().numpy()

    MEAN_Corre=rs.mean()

    # Mean Square Error between model test and test data per neuron
    n_steps=100
    val_mse = np.empty((n_steps,Ntest_trials))
    for indices in range(Ntest_trials):
        val_mse[:,indices] = ms.test_trials_mse(m,DT[indices], ModelT[indices], n_steps)

    MEAN_mse = np.mean(val_mse)
    # Kullback Leibler Divergence

    Model_Signal,_= func.concatenate_list(ModelT,0)
    Data_Signal,_= func.concatenate_list(DT,0)

    Dim_kl = int(np.floor(num_neurons/3))
    neu_list = np.array([1,2,3])
    kl_dim = np.ones([Dim_kl,1])*np.nan
    for j in range(Dim_kl):
        kl_dim[j] = kl.calc_kl_from_data(tc.tensor(Model_Signal[:,neu_list]),
                                         tc.tensor(Data_Signal[:,neu_list]))
        neu_list += 3

    MEAN_kl = kl_dim.mean()

    # Power Spectrum Error
    MEAN_pse,pse_list = ps.power_spectrum_error(tc.tensor(Model_Signal),
                                                tc.tensor(Data_Signal))
    
    #  Correlation Test trials
    corre_list=[]
    l_train_t=Ntraining_trials# number of training trials
    norm_neu=[]
    Nnorm_neu=[]
    limit_validation=[]
    for i_neu in range(num_neurons):
        rs_training=[]
        # Train correlation
        for i_trial in range(l_train_t-1):
            L1=NeuronPattern["Training_Neuron"][i_trial][:,i_neu].shape[0]
            for i_extra in range(i_trial+1,l_train_t):
                L2=NeuronPattern["Training_Neuron"][i_extra][:,i_neu].shape[0]
                if L1==L2:
                    T1=tc.from_numpy(NeuronPattern["Training_Neuron"][i_trial][:,i_neu])
                    T2=tc.from_numpy(NeuronPattern["Training_Neuron"][i_extra][:,i_neu])
                    eps=tc.randn_like(T2)*1e-5
                    X_eps_noise=T2+eps
                    correlation=func.pearson_r(X_eps_noise,T1)
                    correlation=correlation.detach().numpy()
                    rs_training.append(correlation)
                elif L1<L2:
                    X_rsample = signal.resample(NeuronPattern["Training_Neuron"][i_extra][:,i_neu],
                                        NeuronPattern["Training_Neuron"][i_trial].shape[0])
                    T1 = tc.from_numpy(NeuronPattern["Training_Neuron"][i_trial][:,i_neu])
                    T2 = tc.from_numpy(X_rsample)
                    eps=tc.randn_like(T2)*1e-5
                    X_eps_noise=T2+eps
                    correlation=func.pearson_r(X_eps_noise,T1)
                    correlation=correlation.detach().numpy()
                    rs_training.append(correlation)
                else:
                    X_rsample = signal.resample(NeuronPattern["Training_Neuron"][i_trial][:,i_neu],
                                        NeuronPattern["Training_Neuron"][i_extra].shape[0])
                    T1 = tc.from_numpy(X_rsample)
                    T2 = tc.from_numpy(NeuronPattern["Training_Neuron"][i_extra][:,i_neu])
                    eps=tc.randn_like(T2)*1e-5
                    X_eps_noise=T2+eps
                    correlation=func.pearson_r(X_eps_noise,T1)
                    correlation=correlation.detach().numpy()
                    rs_training.append(correlation)
        data=np.array(rs_training)
        corre_list.append(data)
        norm_prob_plot = norm.ppf(np.linspace(0.01, 0.99, len(data)))  # Generate quantiles from a theoretical normal distribution
        sorted_data = np.sort(data)

        # Create a linear regression model instance
        model = LinearRegression()
        # Fit the model to the data
        model.fit(norm_prob_plot.reshape(-1,1), sorted_data.reshape(-1,1))
        # Predicting y values using the fitted model
        y_pred = model.predict(norm_prob_plot.reshape(-1,1))
        # Calculate R-squared
        r_squared = r2_score(sorted_data.reshape(-1,1), y_pred)
        # Testing normality of the correlation
        if r_squared > 0.98:
            norm_neu.append(i_neu)
            # Calculate sample mean and standard deviation
            sample_mean = np.mean(data)
            sample_std = np.std(data,ddof=1)  # ddof=1 for sample standard deviation
            limit_validation.append(sample_mean)#+sample_std)
        else:
            Nnorm_neu.append(i_neu)
            limit_validation.append(999)
    limit_validation=np.array(limit_validation)
    #%Testing correlation Test Trials
    ratio_tests=[]
    for i in norm_neu:
        pass_test=rs[i,:]>limit_validation[i]
        ratio_tests.append(sum(pass_test)/Ntest_trials)
    
    MEAN_ceval=np.array(ratio_tests).mean()
 
    # FIGURES
    plt.figure()
    for it in range(Ntest_trials):
        plt.hist(rs[:,it],alpha=0.3)
    plt.xlabel("Corr(Model vs Data)")
    plt.ylabel("neurons")
    plt.title("Distribution Test Trials")
    plot_name=os.path.join(m_pathway,run+"_Test_Distr_Corr.png").replace('\\','/')
    plt.savefig(plot_name, bbox_inches='tight')

    Higher=np.array([np.where(rs[:,i]>0.4)[0].shape[0]/num_neurons for i in range(Ntest_trials)])
    Lower=np.array([np.where(rs[:,i]<0.4)[0].shape[0]/num_neurons for i in range(Ntest_trials)])
    Trials = [i for i in range(Ntest_trials)]
    Ratio_neurons = {
        ">0.4": Higher,
        "<0.4": Lower,
    }
    width = 0.5
    fig, ax = plt.subplots()
    bottom = np.zeros(Ntest_trials)
    for boolean, weight_count in Ratio_neurons.items():
        p = ax.bar(Trials, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count
    ax.set_title("Neuron Groups Test Trials")
    ax.set_ylabel("Ratio Neurons")
    ax.set_xlabel("Trials")
    lgd=ax.legend(title="Correlation",bbox_to_anchor=(1.1, 1.05))
    plot_name=os.path.join(m_pathway,run+"_Test_Distr_Corr_Trial.png").replace('\\','/')
    plt.savefig(plot_name, bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.figure()
    plt.plot(range(Ntest_trials),rs.mean(0))
    plt.xlabel("Trials")
    plt.ylabel("Mean Correlation")
    plt.title("Test Trials")
    plt.ylim([0,1])
    plot_name=os.path.join(m_pathway,run+"_Test_Correlation.png").replace('\\','/')
    plt.savefig(plot_name, bbox_inches='tight')

    plt.figure()
    plt.plot(range(Dim_kl),kl_dim)
    plt.xlabel("Subsets of 3 neurons")
    plt.ylabel("KLx")
    plot_name=os.path.join(m_pathway,run+"_Test_KLx.png").replace('\\','/')
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
    ax.set_title('Test')
    plot_name=os.path.join(m_pathway,run+"_Test_PhasePlane.png").replace('\\','/')
    plt.savefig(plot_name, bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.figure()
    plt.plot(range(num_neurons),pse_list)
    plt.xlabel("Neurons")
    plt.ylabel("PSE")
    plot_name=os.path.join(m_pathway,run+"_Test_PSE.png").replace('\\','/')
    plt.savefig(plot_name, bbox_inches='tight')

    #Example Evaluation of test correlation
    rand_neu=np.random.randint(0,num_neurons)
    rand_test=np.random.randint(0,Ntest_trials)        
    plt.figure()
    plt.hist(corre_list[rand_neu],bins=20,range=(-1,1))
    plt.axvline(rs[rand_neu,rand_test],color='r')
    plt.axvline(limit_validation[rand_neu],color='k')
    plt.xlabel("Correlation")
    plt.ylabel("Pair trials")
    plt.title("Correlation Distribution Session")

    #Ratio of test trials accepted as good trials 
    plt.figure()
    plt.hist(np.array(ratio_tests),bins=20,range=(0,1))
    plt.xlabel("Ratio of accepted test trials")
    plt.ylabel("Neurons")

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
    Model_Eval["Correlation"] = MEAN_Corre
    #Mean MSE testing in 100 sections for each trial
    Model_Eval["MSE"] = MEAN_mse
    #Mean PSE of the whole session. 
    Model_Eval["PSE"] = MEAN_pse
    #Mean Kullback leibler divergence of the whole session
    Model_Eval["KLx"] = MEAN_kl
    #Validation of test correlation 
    Model_Eval["CEval"] =MEAN_ceval

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
test_n,test_i = func.load_data(data_path,'Test')

Data_info={"Training_Neuron":train_n,"Training_Input":train_i,
           "Testing_Neuron":test_n,"Testing_Input":test_i}

# Load Metadata
file=open(os.path.join(data_path,'Metadata.pkl'),'rb')
Metadata_info=pickle.load(file)
file.close()

######################################## Test measurements #######################################################

# Computation of testing measurements for the models in your model_path
model_list=next(os.walk(model_path))[1]
#Initialization of evaluations lists
Correlation=[]
PSE=[]
NMSE = []
KLx=[]
CEva=[]
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
        Hyper,Eval= Testing_eval(pathway,j,data_path,Data_info,Metadata_info)
        # List of evaluations
        NMSE.append(Eval["MSE"])
        Correlation.append(Eval["Correlation"])
        PSE.append(Eval["PSE"])
        KLx.append(Eval["KLx"])
        CEva.append(Eval["CEval"])
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
TestData={"Models":Model_name,"Runs":RunNumber,
           "Hiddn_Units":hidden,"Sequence_Length":sl,
           "Lambda1":lm1,"Lambda2":lm2,"Lambda3":lm3,
           "Correlation":Correlation,"NMSE":NMSE,"PSE":PSE,
           "KLx":KLx,"CEvaluation":CEva
          }
Testdf=pd.DataFrame(TestData)

# Check/Create Path
if os.path.exists(save_path):
    os.chdir(save_path)
else:
    os.makedirs(save_path)
    os.chdir(save_path)
save_file='TestEvaluation_'+save_name+'.csv'
Testdf.to_csv(save_file,index=False)
# %%
