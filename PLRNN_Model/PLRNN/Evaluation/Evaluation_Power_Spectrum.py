# %% Import Libraries
import os
import pickle
import random
import numpy as np
import torch as tc
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import evaluation_Function_PSA as psa
from bptt.models import Model
from evaluation import pse as ps
from function_modules import model_anafunctions as func

import scipy.io
import quantities as pq
plt.rcParams['font.size'] = 20

#%%
#########################################################################################################
##################################### Real Data - Power Spectrum ########################################
#########################################################################################################

# Study of the Gaussian convoluation of the spike activity in the Power Spectrum analysis.
# We will compare the convolution with:
# - the optimal kernel computed for the full recordings
# - the optimal kernel computed for the recordings belong to the session (task) 
# - the convolution with fixed kernel size with different values

path='D:/_work_cestarellas/Analysis/PLRNN/Session_Selected/OFC/CE17_L6'                                   # Pathway of the data (behaviour & Spike activity)
filename = "session.csv"
os.chdir(path)
# Load data
# Open the Behaviour file
BehData = pd.read_csv(filename)
#Load matlab file (Spike Activity)
data = scipy.io.loadmat('CE17_L6_SpikeActivity.mat')                                
STMtx = data["STMX"]

# set-up parameters: Sampling
SampleTime=np.nanmax(STMtx[-1,:])*pq.s                                                                    # Time of the total experiment in seconds
SampleRt=20000*pq.Hz                                                                                      # Sampling rate in Hertz
SampleRtPoints = 20000                                                                                    # Rate in Sampling Points
SamplePd=(1.0/SampleRt.simplified)                                                                        # Sampling period in seconds
SampleMsPd=SamplePd.rescale('ms')                                                                         # Sampling period in ms
SamplePt = (SampleTime*SampleRt).magnitude                                                                # Total number of data points
NumNeu=np.size(STMtx,axis=1)                                                                              # Total number of units/neurons

# Defining Behavioural Events
# Trial Info: version 07/2023: DataFrame with error in InterTrialIntervals
NumTrials = BehData.shape[0]                                                                              # number of trials of the session
StartTrial = np.array(BehData.wheel_stop/SampleRtPoints)                                                  # times of trial initiation
CueScreen = np.array(BehData.Cue_present/SampleRtPoints)                                                  # times of cue presented
RewardTime = np.array(BehData.reward/SampleRtPoints)                                                      # times of reward
# times of trial end: Not the best way: Problems with the dataFrame ITI data --> string...
EndTrial = np.array([StartTrial[i] for i in range(1,StartTrial.shape[0])])                       
EndTrial = np.append(EndTrial,np.array(BehData.trial_end[NumTrials-1]/SampleRtPoints)) 
assert EndTrial.shape[0]==StartTrial.shape[0],"End times length does not match with the Start times length"

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SPIKE CONVOLUTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print('Neural Convolution')
print("Number of Neurons: ",NumNeu)
print("Number of Trials: ",NumTrials)

# Bin_size for the instantaneous rate
InstBs = 0.02*pq.s                                                           # sample period
InstRt = (1/InstBs).rescale('Hz')                                            # sample rate
KernelSelect = np.linspace(10*InstBs.magnitude,10,num = 8)                   # possible kernel bandwidth

print('Neural Convolution')
print("Number of Neurons: ",NumNeu)
print("Number of Trials: ",NumTrials)

# Bin_size for the instantaneous rate
InstBs = 0.02*pq.s                                                           # sample period
InstRt = (1/InstBs).rescale('Hz')                                            # sample rate
KernelSelect = np.linspace(10*InstBs.magnitude,10,num = 8)                   # possible kernel bandwidth

# Computing convolution with optimal kernel from full recording data
NeuralConvolutionF,NeuralActiveF,NeuKernelF = psa.FullConvolution(STMtx,NumNeu,NumTrials,StartTrial,EndTrial,InstBs,KernelSelect) 
# Computing convolution with optimal kernel from recording data that belongs to the session
NeuralConvolutionS,NeuralActiveS,NeuKernelS = psa.SessionConvolution(STMtx,NumNeu,NumTrials,StartTrial,EndTrial,InstBs,KernelSelect)

# Plot to compare the Optimal Kernels computed in both methods
fig, axs = plt.subplots(1,2,figsize=(15,8))
fig.suptitle('Comparison Optimal Kernel Neurons')
axs[0].hist(NeuKernelF,bins = 50)
axs[0].set_ylabel("Neurons")
axs[0].set_xlabel("Kernel Size (s)")
axs[0].set_title("Full Recordings")
axs[1].hist(NeuKernelS,bins=500)
axs[1].set_xlim([0,10])
axs[1].set_ylabel("Neurons")
axs[1].set_xlabel("Kernel Size (s)")
axs[1].set_title("Session Recordings")

# Computing the Convolution with fixed Kernel size
KernelSize=0.8
NeuralConvF08 = psa.FixedConvolution(STMtx,NumNeu,InstBs,KernelSize)

KernelSize=0.2
NeuralConvF02 = psa.FixedConvolution(STMtx,NumNeu,InstBs,KernelSize)

KernelSize=0.08
NeuralConvF008 = psa.FixedConvolution(STMtx,NumNeu,InstBs,KernelSize)

#%% PLot of the Example of different convolutions in the same neuron
neuplot=32
ini_plot = 0
end_plot = -1
time = [i*0.02 for i in range(NeuralConvolutionF[int(StartTrial[ini_plot]/0.02):int(EndTrial[end_plot]/0.02),0].shape[0])]

plt.figure(figsize=(15,5))
plt.plot(time,NeuralConvF08[int(StartTrial[ini_plot]/0.02):int(EndTrial[end_plot]/0.02),neuplot],label='Fixed08')
plt.plot(time,NeuralConvF02[int(StartTrial[ini_plot]/0.02):int(EndTrial[end_plot]/0.02),neuplot],label='Fixed02')
plt.plot(time,NeuralConvF008[int(StartTrial[ini_plot]/0.02):int(EndTrial[end_plot]/0.02),neuplot],label='Fixed008')
plt.plot(time,NeuralConvolutionF[int(StartTrial[ini_plot]/0.02):int(EndTrial[end_plot]/0.02),neuplot],label='Full')
plt.plot(time,NeuralConvolutionS[int(StartTrial[ini_plot]/0.02):int(EndTrial[end_plot]/0.02),neuplot],label='Session')
plt.xlabel('Time (s)')
plt.ylabel('Zscore (FR)')
plt.title(" Example of Neurons and Convolution")
plt.legend()
plt.show()

#%% Generation of Random Spike Trains for each neuron

NewSTMtx = np.ones(STMtx.shape)*np.nan

for i in range(STMtx.shape[1]):
    Section=STMtx[~np.isnan(STMtx[:,i]),i]
    NewSection=np.sort(np.array([random.randrange(int(Section[0]*SampleRtPoints ),int(Section[-1]*SampleRtPoints ))/SampleRtPoints  for i in range(Section.shape[0])]))
    NewSTMtx[0:NewSection.shape[0],i]=NewSection

# Convolution of Random Spikes
RNeuralConvolutionF,RNeuralActiveF,RNeuKernelF = psa.FullConvolution(NewSTMtx,NumNeu,NumTrials,StartTrial,EndTrial,InstBs,KernelSelect) 
RNeuralConvolutionS,RNeuralActiveS,RNeuKernelS = psa.SessionConvolution(NewSTMtx,NumNeu,NumTrials,StartTrial,EndTrial,InstBs,KernelSelect)
KernelSize=0.2
RNeuralConvF02 = psa.FixedConvolution(NewSTMtx,NumNeu,InstBs,KernelSize)

#%% Computing Firing Rate of each Neuron per trial and identifying the neurons with spike activity in the 95 % of the session
FR_Neurons = np.ones((NumTrials,NumNeu))*np.nan
for i in range(NumNeu):
    for j in range(NumTrials):
        Initial_time = StartTrial[j]
        Final_time = EndTrial[j]
        FR_Neurons[j,i] = np.sum((STMtx[:,i]>Initial_time) & (STMtx[:,i]<Final_time))/(Final_time-Initial_time)

# Taking active neurons during the session
BestNeurons=np.where(np.sum(FR_Neurons==0,0)/FR_Neurons.shape[0]*100<=5)[0]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#################################### TESTING POWER SPECTRUM #############################################
#########################################################################################################

Act_ConvFull = NeuralConvolutionF[:,BestNeurons]
Act_ConvSess = NeuralConvolutionS[:,BestNeurons]
Act_ConvFi02 = NeuralConvF02[:,BestNeurons]

RAct_ConvFull = RNeuralConvolutionF[:,BestNeurons]
RAct_ConvSess = RNeuralConvolutionS[:,BestNeurons]
RAct_ConvFi02 = RNeuralConvF02[:,BestNeurons]

Total_Neurons=BestNeurons.shape[0]
Ini=int(StartTrial[0]/0.02)
End=int(EndTrial[-1]/0.02)

FullData = Act_ConvFull[Ini:End,:]
FullData = FullData-FullData.mean(0)
FullRandom = RAct_ConvFull[Ini:End,:]
FullRandom = FullRandom-FullRandom.mean(0)

SessionData = Act_ConvSess[Ini:End,:]
SessionData = SessionData-SessionData.mean(0)
SessionRandom = RAct_ConvSess[Ini:End,:]
SessionRandom = SessionRandom-SessionRandom.mean(0)

SmallKernel = Act_ConvFi02[Ini:End,:]
SmallKernel = SmallKernel-SmallKernel.mean(0)
SmallRandom = RAct_ConvFi02[Ini:End,:]
SmallRandom = SmallRandom-SmallRandom.mean(0)


small = []
full = []
ses = []
small_rand = []
full_rand = []
ses_rand = []
for dim in range(SessionData.shape[1]):
    ses.append(ps.get_power_spectrum(SessionData[:,dim]))
    full.append(ps.get_power_spectrum(FullData[:,dim]))
    small.append(ps.get_power_spectrum(SmallKernel[:,dim]))
    ses_rand.append(ps.get_power_spectrum(SessionRandom[:,dim]))
    full_rand.append(ps.get_power_spectrum(FullRandom[:,dim]))
    small_rand.append(ps.get_power_spectrum(SmallRandom[:,dim]))

#%% PLot Example
SF = 1/0.02
frequency=[SF/ses[0].shape[1]*i for i in range(ses[0].shape[1])]

neuplot = 10
fig, axs = plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
fig.suptitle('Individual Neuron: '+str(neuplot))
axs[0,0].plot(frequency,ses[neuplot][0,:])
axs[0,0].set_title("Kernel Session")

axs[0,1].plot(frequency,full[neuplot][0,:])
axs[0,1].set_title("Kernel Full")

axs[0,2].plot(frequency,small[neuplot][0,:])
axs[0,2].set_title("Kernel Small: 0.2")

axs[1,0].plot(frequency,ses_rand[neuplot][0,:])
axs[1,0].set_title("Random")

axs[1,1].plot(frequency,full_rand[neuplot][0,:])
axs[1,1].set_title("Random")

axs[1,2].plot(frequency,small_rand[neuplot][0,:])
axs[1,2].set_xlim([0,1])
axs[1,2].set_title("Random")


# Mean Pop
PS_Session_mean = np.array(ses).squeeze().T.mean(1)
PS_Session_std = np.array(ses).squeeze().T.std(1)/np.sqrt(Total_Neurons)
PS_Full_mean = np.array(full).squeeze().T.mean(1)
PS_Full_std = np.array(full).squeeze().T.std(1)/np.sqrt(Total_Neurons)
PS_Small_mean = np.array(small).squeeze().T.mean(1)
PS_Small_std = np.array(small).squeeze().T.std(1)/np.sqrt(Total_Neurons)

RPS_Session_mean = np.array(ses_rand).squeeze().T.mean(1)
RPS_Session_std = np.array(ses_rand).squeeze().T.std(1)/np.sqrt(Total_Neurons)
RPS_Full_mean = np.array(full_rand).squeeze().T.mean(1)
RPS_Full_std = np.array(full_rand).squeeze().T.std(1)/np.sqrt(Total_Neurons)
RPS_Small_mean = np.array(small_rand).squeeze().T.mean(1)
RPS_Small_std = np.array(small_rand).squeeze().T.std(1)/np.sqrt(Total_Neurons)


fig, axs = plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
fig.suptitle('Average Population')
axs[0,0].plot(frequency,PS_Session_mean)
axs[0,0].set_title("Kernel Session")

axs[0,1].plot(frequency,PS_Full_mean)
axs[0,1].set_title("Kernel Full")

axs[0,2].plot(frequency,PS_Small_mean)
axs[0,2].set_title("Kernel Small: 0.2")

axs[1,0].plot(frequency,RPS_Session_mean)
axs[1,0].set_title("Random")

axs[1,1].plot(frequency,RPS_Full_mean)
axs[1,1].set_title("Random")

axs[1,2].plot(frequency,RPS_Small_mean)
axs[1,2].set_xlim([0,1])
axs[1,2].set_title("Random")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
########################################  LOAD DATA & MODEL #############################################
#########################################################################################################

#data_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17/L6/Test1/datasets/' 
data_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17/L6/Test_WholePop/datasets/' 

#model_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/Model_Selected/'
model_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/Final_Models/OFC/CE17_L6_221008'

#model_name = 'Test_01_HU_256_l1_0.001_l2_64_l3_00_SL_400_encdim_64/001'
model_name = 'Population_HU_512_l1_0.001_l2_08_l3_00_SL_400_encdim_155/001'
mpath=os.path.join(model_path,model_name).replace('\\','/')

psa.Hyper_mod(mpath,data_path)

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


#  W parameters and its Correlation trial-trial 
print(repr(m), f"\nNumber of Parameters: {m.get_num_trainable()}")                                                      # get trainable parameters
At, W1t, W2t, h1t, h2t, Ct = m.get_latent_parameters()

# Transform tensor to numpy format
A = At.detach().numpy()
W2 = W2t.detach().numpy().transpose(1,2,0)
W1 = W1t.detach().numpy().transpose(1,2,0)
h1 = h1t.detach().numpy()
h2 = h2t.detach().numpy()
C = Ct.detach().numpy()

#Setup constant values
num_trials=W2.shape[-1]                                                                                                 #number of trials
num_neurons=W2.shape[0]                                                                                                 #number of neurons
num_inputs=C.shape[1]                                                                                                   #number of inputs


# Generation Training Data
TrainS=[]
for w_index in tqdm(range(len(NeuronPattern["Training_Neuron"]))):
    data_trial=tc.from_numpy(NeuronPattern["Training_Neuron"][w_index]).float()                                         # tensor of neuronal data for initial trial data
    input_trial = tc.from_numpy(NeuronPattern["Training_Input"][w_index]).float()
    length_sim = input_trial.shape[0]
    X, _ = m.generate_free_trajectory(data_trial,input_trial,length_sim,w_index)
    TrainS.append(X[:,:])
Train_Signal,_ = psa.concatenate_list(TrainS,0)

# Generation Testing Data
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
TestS = []
#Generate Latent states for Test Trials
W1_ind = [tc.from_numpy(W1_test[:,:,i]).float() for i in range(len(t_prev))]
W2_ind = [tc.from_numpy(W2_test[:,:,i]).float() for i in range(len(t_prev))]
test_tc = [tc.from_numpy(NeuronPattern["Testing_Neuron"][i]).float() for i in range(len(t_prev))]
for i in range(len(W1_ind)):
        data_test=tc.from_numpy(NeuronPattern["Testing_Neuron"][i]).float()
        input_test=tc.from_numpy(NeuronPattern["Testing_Input"][i]).float()
        T0=int(len(NeuronPattern["Testing_Neuron"][i]))
        X, _ = m.generate_test_trajectory(data_test[0:11,:],W2_ind[i],W1_ind[i],input_test, T0,i)
        TestS.append(X)
Test_Signal,_=psa.concatenate_list(TestS,0)

# GENERATION SNAPSHOT
data_trial=tc.from_numpy(NeuronPattern["Training_Neuron"][0][-1,:]).float()                                                         # tensor of neuronal data for initial trial data
Length_trial = round(Metadata["BeforeActivity"].shape[0]/len(NeuronPattern["Training_Neuron"]))
Input_channels= NeuronPattern["Training_Input"][0].shape[1]
Inputs=tc.zeros(Length_trial,Input_channels,dtype=tc.float32)

Warm_time=50000
Length_data=Length_trial+1+Warm_time
Input_channels= NeuronPattern["Training_Input"][0].shape[1]
Inputs=tc.zeros(Length_data,Input_channels,dtype=tc.float32)

# Generation of free trajectories for limiting behaviour - SNAPSHOT
print("::snapshot::")
SnapS=[]
for w_index in tqdm(range(len(NeuronPattern["Training_Neuron"]))):
    data_trial=tc.from_numpy(NeuronPattern["Training_Neuron"][w_index]).float()                                                     # tensor of neuronal data for initial trial data
    X, _ = m.generate_free_trajectory(data_trial,Inputs,Length_data,w_index)
    SnapS.append(X[-Length_trial:,:])

SS_Signal,_ = psa.concatenate_list(SnapS,0)

#%%
test_ps = []
train_ps = []
snap_ps = []
Test_Signal = Test_Signal-Test_Signal.mean(0)
Train_Signal = Train_Signal-Train_Signal.mean(0)
SS_Signal = SS_Signal-SS_Signal.mean(0)

for dim in range(SS_Signal.shape[1]):
    test_ps.append(ps.get_power_spectrum(Test_Signal[:,dim]))
    train_ps.append(ps.get_power_spectrum(Train_Signal[:,dim]))
    snap_ps.append(ps.get_power_spectrum(SS_Signal[:,dim]))
    

# %% Example Plot

freq_train = [SF/train_ps[0].shape[1]*i for i in range(train_ps[0].shape[1])]
plt.figure()
plt.plot(frequency,full[0][0,:],label="Original")
plt.plot(freq_train,train_ps[0][0,:],label="Model")
plt.xlabel("Frequency(Hz)")
plt.ylabel("Power")
plt.title("Individual Neuron")
plt.legend()
plt.xlim([0,2])

plt.figure()
plt.plot(frequency,full_rand[0][0,:],label="Random")
plt.plot(freq_train,train_ps[0][0,:],label="Model")
plt.xlabel("Frequency(Hz)")
plt.ylabel("Power")
plt.title("Individual Neuron")
plt.legend()
plt.xlim([0,2])

#%% Selection Session regions without external inputs
Zero_ITrial=[]
num_traintrials = len(NeuronPattern["Training_Neuron"])
for i in tqdm(range(num_traintrials)):
     pos=np.where(np.sum(NeuronPattern["Training_Input"][i],1)==0)
     Zero_ITrial.append(NeuronPattern["Training_Neuron"][i][pos[0],:])

ZT_Signal,_= psa.concatenate_list(Zero_ITrial,0)
limits_ZT = ZT_Signal.shape[0]
ZT_Signal = ZT_Signal-ZT_Signal.mean(0)

zero_ps = []
for dim in range(ZT_Signal.shape[1]):
    zero_ps.append(ps.get_power_spectrum(ZT_Signal[:,dim]))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
######################################  POWER SPECTRUM ERROR ############################################
#########################################################################################################

# PSE: Train vs Original
limit1=Train_Signal.shape[0]
_,TvO = ps.power_spectrum_error(Train_Signal, FullData[:limit1,:])

# PSE: Train vs Random
_,TvRO = ps.power_spectrum_error(Train_Signal, FullRandom[:limit1,:])

# PSE: ITI vs Zero Inputs
limit2=SS_Signal.shape[0]
_,IvZ = ps.power_spectrum_error(SS_Signal, ZT_Signal[:limit2,:])

# PSE: ITI vs Random
_,IvR = ps.power_spectrum_error(SS_Signal, FullRandom[:limit2,:])

# PSE: ITI Real vs Random
limit3 = ZT_Signal.shape[0]
_,RealIvR = ps.power_spectrum_error(ZT_Signal, FullRandom[:limit3,:])

PSE_frame = pd.DataFrame({"MvsD":np.array(TvO),"MvsR":np.array(TvRO),"ITIvsD":np.array(IvZ),
                          "ITIvsR":np.array(IvR),"RealITIvsR":np.array(RealIvR)})

xnum = [i for i in range(PSE_frame.shape[1])]
xlabel = ["MvsD","MvsR","MITIvsD","MITIvsR","RealITIvsR"]
plt.figure()
plt.bar(np.arange(PSE_frame.shape[1]), PSE_frame.mean(), yerr=[PSE_frame.std()/np.sqrt(PSE_frame.shape[0]), PSE_frame.std()/np.sqrt(PSE_frame.shape[0])], capsize=6)
plt.xticks(xnum,xlabel,rotation=45)
plt.ylabel("PS_Error")


# %%
