'''
**Script: Preprocess__RNN_Gambling
**Author: Cristian Estarellas Martin
**Date: 11/2023

**Description:
Preprocess of multi unit activity (MUA) for Gambling Task. 
The Script is focused on the optimization of the preprocess of the spike activity for the PLRNN model.
- Convolution of the neuronal activity with a gaussian kernel
- Determining optimal kernel for each neuron
- Concatenating experimental trials for the PLRNN 

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


#%% Libraries
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import neo
import scipy.io
import scipy.stats as stats
import elephant as el
import quantities as pq

plt.rcParams['font.size'] = 20
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%
def Metadata_Extraction(Info):
    MI=Info["Metadata"][0][0]
    Metadata = {}
    Metadata["Type"] = MI[0][0]
    Metadata["Code"] = MI[1][0]
    Metadata["Experimenter"] = MI[2][0]
    Metadata["Age"] = MI[3][0]
    Metadata["Recordings"] = MI[4][0]
    Metadata["Region"] = MI[5][0]
    Metadata["Laboratory"] = MI[6][0]
    Metadata["Date"] = MI[7][0]
    Metadata["Initial_Time"] = MI[8][0]
    Metadata["Final_Time"] = MI[8][1]
    Metadata["SR"] = float(MI[9][0][0])
    Metadata["Experiment"] = MI[10][0]
    Metadata["Spike_Sorting"] = MI[11][0]
    return Metadata

def SessionConvolution(SM,NumNeu,NumTrials,StartTrial,EndTrial,InstBs,KernelSelect):
    MaxTime=(np.nanmax(SM[-1,:])+0.5)*pq.s
    NeuKernel = np.zeros(NumNeu)
    InstRate = []
    NeuralActive = []
    for i in tqdm(range(NumNeu)):
        # Optimal Kernel for the Whole Session
        Train=neo.SpikeTrain(times = SM[~np.isnan(SM[:,i]),i]*pq.s,t_stop=MaxTime)    # selection of the Spikes in the session
        SpikeFull = Train[(Train>StartTrial[0])*(Train<EndTrial[-1])]
        OptimalFullKernel = el.statistics.optimal_kernel_bandwidth(SpikeFull.magnitude)         # computing convolution for the session
        #SpikeInfo = defaultdict(list)                                                      # creating 
        KernelSigma = np.zeros(NumTrials)                                                   # initialization Sigma value of Kernel per each trial
        MeanFiringRate = np.zeros(NumTrials)                                                # initialization of Mean Firing rate per each trial
        # Optimal Kernel for each Trial
        for i_t in range(NumTrials):
            SpikeSet = Train[(Train>StartTrial[i_t])*(Train<EndTrial[i_t])]
            SpikeSet.t_start = StartTrial[i_t]*pq.s
            SpikeSet.t_stop = EndTrial[i_t]*pq.s
            MeanFiringRate[i_t] = el.statistics.mean_firing_rate(SpikeSet)                  # Mean firing rate per trial
            if len(SpikeSet)>4:
                SetKernel = el.statistics.optimal_kernel_bandwidth(SpikeSet.magnitude,bandwidth=KernelSelect)
                KernelSigma[i_t] = SetKernel['optw']
            else:
                KernelSigma[i_t] = OptimalFullKernel['optw']
        # Obtaining mean values from trials        
        NeuKernel[i] = KernelSigma.mean()
        NeuralActive.append(MeanFiringRate)
        #Final convolution for each unit/neuron
        InstRate.append(el.statistics.instantaneous_rate(Train, sampling_period=InstBs,kernel = el.kernels.GaussianKernel(NeuKernel[i]*pq.s)))    

    NeuronTime = np.linspace(0,MaxTime.magnitude,num = int(MaxTime/InstBs))
    NeuralConvolution = np.array(InstRate[:]).squeeze().T
    NeuralConvolution = stats.zscore(NeuralConvolution)

    assert NeuralConvolution.shape[0]==NeuronTime.shape[0], 'Problems with the the bin_size of the convolution'
    return NeuralConvolution,NeuralActive,NeuKernel

def concatenate_list(lis,ax):
    res = lis[0]    # first element of the list
    Ls = [lis[0].shape[ax]] #first length of the list
    for i in range(1, len(lis)):
        Ls.append(lis[i].shape[ax])
        res = np.concatenate((res, lis[i]), axis=ax)
    return res, Ls

def Data_format(X,S,name,save_directory):
    os.chdir(save_directory)
    # check length (number of trials)
    assert len(X) == len(S)

    # check for consistency
    N = X[0].shape[-1]
    K = S[0].shape[-1]
    for x, s in zip(X, S):
        assert isinstance(x, np.ndarray)
        assert isinstance(s, np.ndarray)
        assert x.shape[-1] == N
        assert s.shape[-1] == K
        assert x.shape[0] == s.shape[0]

    # name
    #name = "Training"

    # create datasets folder if it does not already exist
    os.makedirs("datasets", exist_ok=True)

    # store with pickle
    with open(f"datasets/{name}_data.npy", "wb") as fp:
        pickle.dump(X, fp)

    with open(f"datasets/{name}_inputs.npy", "wb") as fp:
        pickle.dump(S, fp)

    # loading
    X_ = np.load(f"datasets/{name}_data.npy", allow_pickle=True)
    S_ = np.load(f"datasets/{name}_inputs.npy", allow_pickle=True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOADING DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

path='D:/_work_cestarellas/Analysis/PLRNN/Session_Selected/OFC/DM01_7_220524'   # Pathway of the data (behaviour & Spike activity)
save_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/DM01_7_DP'

os.chdir(path)
list_files = os.listdir(path)

for i in list_files:
    if i.find('Behaviour')>0:
        Behaviour_name = i
    if i.find('Metadata')>0:
        Metadata_name = i
    if i.find('SpikeActivity')>0:
        Spike_name = i

# Load data
# Metadata
Info_Experiment = scipy.io.loadmat(Metadata_name)
Metadata = Metadata_Extraction(Info_Experiment)
# Open the Behaviour file
Bdata = scipy.io.loadmat(Behaviour_name)
BehData = Bdata[list(Bdata.keys())[-1]]
# Load matlab file (Spike Activity)
Sdata = scipy.io.loadmat(Spike_name)                                
STMtx = Sdata[list(Sdata.keys())[-1]]/Metadata["SR"]

# Info Columns BehData:
# 0- Trial Start
# 1- Trial End
# 2- Duration (Seconds)
# 3- Block
# 4- Gamble Arm (Right = 1, Left = 0)
# 5- Probability big Reward
# 6- Probability Small Reward
# 7- Ammount Big Reward
# 8- Ammount Small Reward
# 9- Number of previously wheel not stopping
# 10- Not responding Trial
# 11- Chosen Side (Right = 1, Left = 0)
# 12- Chosen Arm (Gamble = 1, Safe = 0)
# 13- Reward Given
# 14- Start of the trial (Sampling points)
# 15- Cue Presentation (Sampling Points)
# 16- Start of the response window (Sampling Points)
# 17- Reward Period (Sampling Points)
# 18- End of the trial

####################################### PATH FOR SAVING DATA #######################################

# Check/Create Path
os.makedirs(save_path, exist_ok=True)
os.chdir(save_path)
os.makedirs("plots", exist_ok=True)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BEHAVIOUR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Classification of trials following the behabiour
GambleRewardTrials = np.where((BehData[:,12]==1) & (BehData[:,13]==1))[0]
GambleNoRewardTrials =  np.where((BehData[:,12]==1) & (BehData[:,13]==0))[0]
SafeRewardTrials = np.where((BehData[:,12]==0) & (BehData[:,13]==1))[0]
SafeNoRewardTrials = np.where((BehData[:,12]==0) & (BehData[:,13]==0))[0]
NoRespondingTrials = np.where(BehData[:,10]==1)[0]

# Blocks
Block_Prob = np.unique(BehData[:,5])
BlockTrials = [np.where(Block_Prob[i]==BehData[:,5])[0][0] for i in range(len(Block_Prob))]
# Smoothing the data for plotting
ScaleDecision=BehData[:,12]+(BehData[:,12]-1)
SigmaDecision=1
Binx =0.5
KernelWindow = np.arange(-3*SigmaDecision, 3*SigmaDecision, Binx)
KernelDecision = np.exp(-(KernelWindow/SigmaDecision)**2/2)
DecisionConvolution=np.convolve(ScaleDecision,KernelDecision,mode='same')
DecisionNormalized=(DecisionConvolution/np.nanmax(np.abs(DecisionConvolution))+1)/2

# FIGURE: Plot behaviour performance
plt.figure(figsize=(25,5))
plt.plot(DecisionNormalized)
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
for i in BlockTrials:
    plt.axvline(i,linestyle='dashed',color='r',)     
plt.ylim([-0.2,1.2])
plt.yticks(ticks=[1.0,0.5,0.0])
plt.xlabel('Trials')
plt.ylabel("Animal's choice")
plt.title('Full Session')
plt.savefig('plots/Full_Session.png')



if NoRespondingTrials[0]<20:
    first_trial = NoRespondingTrials[0]+1
    last_trial = NoRespondingTrials[1]-1
else:
    first_trial = 0
    last_trial = NoRespondingTrials[0]-1
    
first_trial = 0
last_trial = NoRespondingTrials[3]-1

# FIGURE: Plot behaviour performance
plt.figure(figsize=(25,5))
plt.plot(DecisionNormalized)
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
for i in BlockTrials:
    plt.axvline(i,linestyle='dashed',color='r',)     
plt.ylim([-0.2,1.2])
plt.yticks(ticks=[1.0,0.5,0.0])
plt.xlim([first_trial,last_trial])
plt.xlabel('Trials')
plt.ylabel("Animal's choice")
plt.title('Session for modelling')
plt.savefig('plots/Model_Session.png')


#Generation of Behavioural Matrix for the section selected
Beh_Matrix = BehData[first_trial:last_trial+1,:]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Data Convolution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# set-up parameters: Sampling
SampleTime=np.nanmax(STMtx[-1,:])*pq.s                                                                    # Time of the total experiment in seconds
SampleRt=20000*pq.Hz                                                                                      # Sampling rate in Hertz
SampleRtPoints = 20000                                                                                    # Rate in Sampling Points
SamplePd=(1.0/SampleRt.simplified)                                                                        # Sampling period in seconds
SampleMsPd=SamplePd.rescale('ms')                                                                         # Sampling period in ms
SamplePt = (SampleTime*SampleRt).magnitude                                                                # Total number of data points

# Defining Behavioural Events
# Trial Info: version 07/2023: DataFrame with error in InterTrialIntervals
StartTrial = np.array(Beh_Matrix[:,14]/SampleRtPoints)                                                  # times of trial initiation
CueScreen = np.array(Beh_Matrix[:,15]/SampleRtPoints)                                                  # times of cue presented
RewardTime = np.array(BehData[:,17]/SampleRtPoints)                                                      # times of reward
EndTrial = np.array([StartTrial[i] for i in range(1,StartTrial.shape[0])])                       
EndTrial = np.append(EndTrial,np.array(Beh_Matrix[-1,18]/SampleRtPoints)) 
assert EndTrial.shape[0]==StartTrial.shape[0],"End times length does not match with the Start times length"

Duration = EndTrial-StartTrial

plt.figure()
plt.hist(Duration,bins=200)
plt.xlabel('Duration (s)')
plt.ylabel('Trials')
plt.savefig('plots/Duration_Histogram.png')

BehTrials = [i for i in range(first_trial,last_trial)]
FullNumNeu = STMtx.shape[1]
# Computing Firing Rate of each Neuron per trial and identifying the neurons with spike activity in the 95 % of the session
FR_Neurons = np.ones((len(BehTrials),FullNumNeu))*np.nan
for i in range(FullNumNeu):
    for j in range(len(BehTrials)):
        Initial_time = StartTrial[BehTrials[j]]
        Final_time = EndTrial[BehTrials[j]]
        FR_Neurons[j,i] = np.sum((STMtx[:,i]>Initial_time) & (STMtx[:,i]<Final_time))/(Final_time-Initial_time)

# Taking active neurons during the session
BestNeurons=np.where(np.sum(FR_Neurons==0,0)/FR_Neurons.shape[0]*100<=5)[0]

NumNeu=len(BestNeurons)                                                                              # Total number of units/neurons
NumTrials= last_trial
print('Neural Convolution')
print("Number of Neurons: ",NumNeu)
print("Number of Trials: ",NumTrials)

# Bin_size for the instantaneous rate
InstBs = 0.02*pq.s                                                           # sample period
InstRt = (1/InstBs).rescale('Hz')                                            # sample rate
KernelSelect = np.linspace(10*InstBs.magnitude,4,num = 8)                   # possible kernel bandwidth

SpikeMatrix=STMtx[:,BestNeurons]
# Computing convolution with optimal kernel from recording data that belongs to the session
NeuralConvolutionS,NeuralActiveS,NeuKernelS = SessionConvolution(SpikeMatrix,NumNeu,NumTrials,StartTrial,EndTrial,InstBs,KernelSelect)


fig, axs = plt.subplots(1,2,figsize=(12,7))
fig.suptitle('Preprocess Info')
axs[0].hist(NeuKernelS,bins = 50)
axs[0].set_ylabel("Neurons")
axs[0].set_xlabel("Kernel Size (s)")
axs[0].set_title("Session")
axs[1].hist(FR_Neurons.mean(0),bins=50)
axs[1].set_ylabel("Neurons")
axs[1].set_xlabel("Firing Rate (Hz)")
axs[1].set_title("Average Session")
plt.savefig('plots/Preprocess_Distribution.png')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FILTER INFO PREPROCESS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Selecting neurons with the optimal kernel smaller than 1 second
Neu_sel=np.where(NeuKernelS<1)[0]

NeuralConvolutionS = NeuralConvolutionS[:,Neu_sel]

#%%
temp_vec = np.linspace(0,10000,10000)*0.02
plt.figure()
plt.plot(temp_vec,NeuralConvolutionS[:10000,:])
plt.xlabel("Time (s)")
print("Neurons: ",NeuralConvolutionS.shape[1])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DATA PREPROCESS FOR PLRNN MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LastTrial = Beh_Matrix.shape[0]
# Number of External Inputs
NumberExternalInputs = 3
# Channels for each External Input
ActionChannel = 0
BigRewardChannel = 1
SmallRewardChannel = 2
# Values of External Channels
ActionValue = 1
BigRewardValue = Beh_Matrix[:,12]*4*Beh_Matrix[:,13]
SmallRewardValue = np.abs(Beh_Matrix[:,12]-1)*1*Beh_Matrix[:,13]

#Limit trial end model
DataTrials =[]          # List of Spiking Activity. Each element is a trial
InputTrials = []        # List of External Inputs. Each element is a trial            
NumberBinTrials = []    # Number of bins per trial. Trial Length in sampling points
TrialList = [i for i in range (LastTrial)]

for i in TrialList:
    StartPosition = int(StartTrial[i]/InstBs.magnitude)     # start trial index
    ActionPosition = int(CueScreen[i]/InstBs.magnitude)     # cuen beginning index
    RewardPosition = int(RewardTime[i]/InstBs.magnitude)    # reward trial index
    EndPosition = int(EndTrial[i]/InstBs.magnitude)         # end trial index
    NumberBinTrials.append(EndPosition-StartPosition)
    DataTrials.append(NeuralConvolutionS[StartPosition:EndPosition,:])
    if np.isin(i,NoRespondingTrials):
        InputMatrix = np.zeros((NeuralConvolutionS[StartPosition:EndPosition,:].shape[0],NumberExternalInputs))
    else:
        # initialization input matrix
        InputMatrix = np.zeros((NeuralConvolutionS[StartPosition:EndPosition,:].shape[0],NumberExternalInputs))
        # trial decision
        InputMatrix[(ActionPosition-StartPosition):(RewardPosition-StartPosition),ActionChannel]= ActionValue # trial Action
        # trial big reward
        InputMatrix[(RewardPosition-StartPosition):((RewardPosition-StartPosition)+25),BigRewardChannel]= BigRewardValue[i] 
        # trial small reward
        InputMatrix[(RewardPosition-StartPosition):((RewardPosition-StartPosition)+25),SmallRewardChannel]= SmallRewardValue[i]  
    InputTrials.append(InputMatrix)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONCATENATING TRIALS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Total time in seconds of the concatenated trials
TimeTraningTrial = 20
# TimeTrainingTrial in sampling points
PointsTrainingTrial = TimeTraningTrial/InstBs.magnitude 
# Create a list with the limit of the trials to concatenate: Between InitialTrial and FinalTrial elements
SizeTrial = 0
ListChangeTrial = []
for i in range(len(NumberBinTrials)):
    SizeTrial += NumberBinTrials[i]
    if SizeTrial>PointsTrainingTrial:
        SizeTrial = 0
        SizeTrial +=NumberBinTrials[i]
        ListChangeTrial.append(i-1)
    elif SizeTrial==PointsTrainingTrial:
        ListChangeTrial.append(i)
        SizeTrial = 0

InitialTrial = [0] + [i+1 for i in ListChangeTrial[:-1]]    # List of Initial Trial
FinalTrial = ListChangeTrial                                # List of End Trials
if FinalTrial[-1]<LastTrial:
    InitialTrial=InitialTrial+[FinalTrial[-1]+1]
    FinalTrial = FinalTrial+[LastTrial]
assert len(InitialTrial)==len(FinalTrial), "Error len between InitialTrial and FinalTrial"

# List of concatenate Trials
DataNeuronModel=[] #Empty list for Neuronal Activity
DataInputModel=[]  #Empty list for External Inputs

for i in range(len(InitialTrial)):
    X,S=concatenate_list(DataTrials[InitialTrial[i]:FinalTrial[i]+1],0)
    Xi,Si=concatenate_list(InputTrials[InitialTrial[i]:FinalTrial[i]+1],0)
    DataNeuronModel.append(X)
    DataInputModel.append(Xi)

label_xticks = [ i for i in range(len(InitialTrial))]
# FIGURE: Contatenated trials (black dashed lines)
plt.figure(figsize=(25,5))
plt.plot(DecisionNormalized)
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
for i in BlockTrials:
    plt.axvline(i,color='r',)   
for i in InitialTrial:
    plt.axvline(i,linestyle = 'dashed',color='k')  
plt.xlim([0,LastTrial])
plt.ylim([-0.2,1.2])
plt.yticks(ticks=[1.0,0.5,0.0])
plt.xticks(InitialTrial[0:-1:10],label_xticks[0:-1:10])
plt.xlabel('Concatenated Trials')
plt.ylabel("Animal's choice")
plt.savefig('plots/Session_ConcatenatedTrials.png')

# %%%%%%%%%%%%%%%%%%%%%%%% SELECTION TRAINING AND TEST TRIALS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ModelTrials = [i for i in range(len(InitialTrial))]


# Select specific trials that you will use to test the model
print("Selection of Test Trials:")
# Input number of clusters that you will select
TrialsAmount = int(input("Enter the number of Test Trials you want:"))
TestTrials = []
# iterating till the range
for i in range(0, TrialsAmount):
    ele = int(input("Select Trial Number:")) # Select the number of the clusters that you decided
    # adding the element
    TestTrials.append(ele) 

Training2Test = list(np.array(TestTrials)-np.array([i+1 for i in range(len(TestTrials))]))
TrainingTrials = [i for i in ModelTrials if i not in TestTrials]

DataNeuronTest = [DataNeuronModel[i] for i in TestTrials] 
DataInputTest = [DataInputModel[i] for i in TestTrials] 

DataNeuronTraining = [DataNeuronModel[i] for i in TrainingTrials]
DataInputTraining = [DataInputModel[i] for i in TrainingTrials]

DataNeuronSession = DataNeuronModel
DataInputSession = DataInputModel

assert len(DataNeuronTest)==len(DataInputTest), "Input and Neurons doest not match in Test data"
assert len(DataNeuronTraining)==len(DataInputTraining), "Input and Neurons doest not match in Training data"

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  SAVE DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Training Data
full_name = open("Training.pkl","wb")                                    # Name for training data
pickle.dump([DataNeuronTraining,DataInputTraining],full_name)            # Save train data
full_name.close()                                                        #close save instance 


# Test Data
full_name = open("Test.pkl","wb")                                        # Name for Testing data
pickle.dump([DataNeuronTest,DataInputTest],full_name)                    # Save Test data
full_name.close()                                                        #close save instance 


# Full Session Data
full_name = open("Full_Session.pkl","wb")                                        # Name for Testing data
pickle.dump([DataNeuronSession,DataInputSession],full_name)                    # Save Test data
full_name.close()                                                        #close save instance 

"""
MetaData: Dictionary with the following information
-Number of Neurons: Amount of neurons 
-List of Neurons: Index of each Neuron used from raw data
-TestTrials: the number of Test trials selected
-TrainingTrials: the number of Training Trials
-Training2Test: The Position of the Test trials in the Training Trials
-BeforeActivity: Neuronal Activity before the session
-AfterActivity: Neuronal Activity after the session
"""
MetaData_model = {}
MetaData_model["NumberNeurons"] = BestNeurons.shape[0]
MetaData_model["ListNeurons"] = BestNeurons
MetaData_model["TestTrials"] = TestTrials
MetaData_model["TrainingTrials"] = TrainingTrials
MetaData_model["Training2Test"] = Training2Test

full_name = open("Metadata.pkl","wb")                      # Name for training data
pickle.dump(MetaData_model,full_name)            # Save train data
#close save instance 
full_name.close()

#%%% Save Excel info
# Info to know how to tune the hyper-parameters of the PLRNN model
dict_info={}
dict_info["Number_Neurons"] = len(BestNeurons)
dict_info["Number_Total_Trials"] = len(DataNeuronSession)
dict_info["Number_Training_Trials"] = len(DataNeuronTraining)
dict_info["Number_Test_Trials"] = len(DataNeuronTest)
dict_info["Duration_Concatenated_Trials"] = TimeTraningTrial
dict_info["Mean_bins_per_Trial"] = np.array([DataNeuronSession[i].shape[0] for i in range(len(DataNeuronSession))]).mean()

df = pd.DataFrame(data=dict_info,index=[0])
df=df.T
df.to_excel('Info_Experiment_for_PLRNN.xlsx')

#%%%%%%% GENERATE DATA FORMAT %%%%%%

Data_format(DataNeuronTest,DataInputTest,"Test",save_path)
Data_format(DataNeuronTraining,DataInputTraining,"Training",save_path)
Data_format(DataNeuronSession,DataInputSession,"FullSession",save_path)
# %%
