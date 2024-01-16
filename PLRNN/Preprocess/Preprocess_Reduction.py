'''
**Script: Preprocess_Gambling
**Author: Cristian Estarellas Martin
**Date: 10/2023

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
import seaborn as sns
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

import neo
import scipy.io
import scipy.stats as stats
import elephant as el
import quantities as pq
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plt.rcParams['font.size'] = 20
#%% Necessary Functions

# FUNCTION Concatonate trials
def concatenate_list(lis,ax):
    res = lis[0]    # first element of the list
    Ls = [lis[0].shape[ax]] #first length of the list
    for i in range(1, len(lis)):
        Ls.append(lis[i].shape[ax])
        res = np.concatenate((res, lis[i]), axis=ax)
    return res, Ls

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

# FUNCTION Spike Convolution of Neuronal Activity
def SpikeConvolution(STMtx,NumNeu,NumTrials,StartTrial,EndTrial,KernelSelect):
# Main: Compute the covulation of the spike times to give the spontaneous firing rate for each neuron.
#       The convolution is acroos the total recording time
#       The function compute the mean kernel window across all possible trials and uses this value.
#       If the spikes are not enough for computing the optimal kernel in the trial, the optimal kernel is the one computed for the whole session
# Input:
#   STMtx (Array-2D): Spike time of each neuron (column). Each row is a time in seconds.
#   NumNeu (Int): Number of Neurons
#   NumTrials (Int): Number of trials 

    MaxTime=(np.nanmax(STMtx[-1,:])+0.05)*pq.s
    NeuKernel = np.zeros(NumNeu)
    InstRate = []
    NeuralActive = []
    for i in tqdm(range(NumNeu)):
        # Optimal Kernel for the Whole Session
        Train=neo.SpikeTrain(times = STMtx[~np.isnan(STMtx[:,i]),i]*pq.s,t_stop=MaxTime)    # selection of the Spikes in the session
        OptimalFullKernel = el.statistics.optimal_kernel_bandwidth(Train.magnitude)         # computing convolution for the session
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
    return NeuralConvolution,NeuralActive

# FUNCTION Mean Firing Rate of Neurons Before and After (5 seconds)
def RatesOutSession(BeforeTimes,AfterTimes,STMtx,SampleRtPoints,NumNeu):
# Main: Compute the firing rate of each neuron in a temporal window of size DeltaTime before and after the session
# IMPORTANT: Ordre of temporal series: Free Time - Free Reward
# Input:
#   BeforeTimes(DataFrame):DataFrame where the column 0 Indicates when receive free reward
#   AfterTimes(DataFrame):DataFrame where the column 0 Indicates when receive free reward
#   STMtx (Array-2D): Spike time of each neuron (column). Each row is a time in seconds.
#   SampleRtPoints(Int): Sampling Rate point adquisition
#   NumNeu (Int): Number of Neurons
# Output:
#   NueronFiringRateBefore(Array-2D): Firing rate of each neuron (columns) in temporal windows of 5 seconds (rows). Before the session
#   NeuronFiringRateAfter(Array-2D): Firing rate of each neuron (columns) in temporal windows of 5 seconds (rows). After the session
#  
    # Generate array of times every 5 seconds
    DeltaTime = 5 # Temporal window
    BeforeListTimes = np.arange(0,BeforeTimes.iloc[-1][0]/SampleRtPoints,DeltaTime,dtype=float)
    AfterListTimes = np.arange(EndTrial[-1],AfterTimes.iloc[-1][0]/SampleRtPoints,DeltaTime,dtype=float)

    # Array of begining and ending of 5 seconds
    ArrayBefore = np.zeros((2,BeforeListTimes.shape[0]-1))  # initialization array 
    ArrayAfter = np.zeros((2,AfterListTimes.shape[0]-1))    # initialization array
    # Filling Arrays
    ArrayBefore[0,:] = BeforeListTimes[0:-1]
    ArrayBefore[1,:] = BeforeListTimes[1:]
    ArrayAfter[0,:] = AfterListTimes[0:-1]
    ArrayAfter[1,:] = AfterListTimes[1:]

    NeuronsFiringRateBefore = np.zeros((ArrayBefore.shape[1],NumNeu))
    NeuronsFiringRateAfter = np.zeros((ArrayAfter.shape[1],NumNeu))

    for i in range(ArrayBefore.shape[1]):
        Condition1 = STMtx>ArrayBefore[0,i]
        Condition2 = STMtx<ArrayBefore[1,i]
        NeuronsFiringRateBefore[i,:]=np.sum(Condition1*Condition2,axis=0)/DeltaTime

    for i in range(ArrayAfter.shape[1]):
        Condition1 = STMtx>ArrayAfter[0,i]
        Condition2 = STMtx<ArrayAfter[1,i]
        NeuronsFiringRateAfter[i,:]=np.sum(Condition1*Condition2,axis=0)/DeltaTime
    
    return NeuronsFiringRateBefore,NeuronsFiringRateAfter

def RateSession(Spikes,Trial0,Trial1):
    N = Spikes.shape[1]      # number of neurons
    T = Trial0.shape[0]      # number of trials
    rate = np.zeros((T,N))
    DeltaTime=Trial1-Trial0
    for t in range(T):
        Condition1 = Spikes>Trial0[t]
        Condition2 = Spikes<Trial1[t]
        rate[t,:]=np.sum(Condition1*Condition2,axis=0)/DeltaTime[t]
    return rate

# FUNCTION Kmeans cluster evaluation
def Cluster_Evaluation(N_cluster,PopulationRateScore):
# Main: Determine the Optimal number of clusters from the range of N_clusters and compute the model for the optimal value
# Input: 
#   N_cluster(Int): Number of maximum clusters to test
#   PopulationRateScore: Coefficients of PCA components for the PCA_analysis_3C function
# Output:
#   Modelk: K-mean model for optimal number of clusters tested by Silhouette method
#   OptimalClusterK: Optimal number of clusters

    # Cluster Evaluation
    Silhouette = []
    for i in range(2,N_cluster):
        Modelk = KMeans(n_clusters=i,max_iter=2000,tol=0.1).fit(PopulationRateScore)
        Silhouette.append(silhouette_score(PopulationRateScore,Modelk.labels_))

    # Selection of optimal cluster
    OptimalClusterK = Silhouette.index(max(Silhouette))+2
    Modelk = KMeans(n_clusters=OptimalClusterK).fit(PopulationRateScore)
    return Modelk,OptimalClusterK

# FUNCTION PCA with three PC
def PCA_analysis_3C(NeuronsFiringRate,LimitBefore,LimitSession):
    PCAFiringRate = PCA(n_components=3)
    PCAFiringRate.fit(NeuronsFiringRate)
    PopulationRateState = PCAFiringRate.transform(NeuronsFiringRate)
    PopulationRateScore = PCAFiringRate.components_.T

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(PopulationRateState[0:LimitBefore,0],PopulationRateState[0:LimitBefore,1],PopulationRateState[0:LimitBefore,2])
    ax.scatter(PopulationRateState[LimitBefore+1:LimitSession,0],PopulationRateState[LimitBefore+1:LimitSession,1],PopulationRateState[LimitBefore+1:LimitSession,2])
    ax.scatter(PopulationRateState[LimitSession+1:,0],PopulationRateState[LimitSession+1:,1],PopulationRateState[LimitSession+1:,2])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    return PopulationRateScore,PopulationRateState


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOADING DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

path='D:/_work_cestarellas/Analysis/PLRNN/session_selection_v0/OFC/CE17/L6'   # Pathway of the data (behaviour & Spike activity)
filename = "session.csv"
os.chdir(path)
# Load data
# Open the Behaviour file
BehData = pd.read_csv(filename)
#Load matlab file (Spike Activity)
data = scipy.io.loadmat('STMtx.mat')                                
STMtx = data["STMX"]


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SETTING PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%

# set-up parameters: Sampling
SampleTime=np.nanmax(STMtx[-1,:])*pq.s                                       # Time of the total experiment in seconds
SampleRt=20000*pq.Hz                                                         # Sampling rate in Hertz
SampleRtPoints = 20000                                                       # Rate in Sampling Points
SamplePd=(1.0/SampleRt.simplified)                                           # Sampling period in seconds
SampleMsPd=SamplePd.rescale('ms')                                            # Sampling period in ms
SamplePt = (SampleTime*SampleRt).magnitude                                   # Total number of data points
NumNeu=np.size(STMtx,axis=1)                                                 # Total number of units/neurons

# Defining Behavioural Events
# Trial Info: version 07/2023: DataFrame with error in InterTrialIntervals
NumTrials = BehData.shape[0]                                                 # number of trials of the session
StartTrial = np.array(BehData.wheel_stop/SampleRtPoints)                     # times of trial initiation
CueScreen = np.array(BehData.Cue_present/SampleRtPoints)                     # times of cue presented
RewardTime = np.array(BehData.reward/SampleRtPoints)                         # times of reward
# times of trial end: Not the best way: Problems with the dataFrame ITI data --> string...
EndTrial = np.array([StartTrial[i] for i in range(1,StartTrial.shape[0])])                       
EndTrial = np.append(EndTrial,np.array(BehData.trial_end[NumTrials-1]/SampleRtPoints)) 
assert EndTrial.shape[0]==StartTrial.shape[0],"End times length does not match with the Start times length"

# Block Probabilities
BlockTrials,BlockLabels = Block_Classification(BehData["probability G"])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BEFORE AND AFTER SESSION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load Data Behaviour Before and After the Session
BeforeTimes = pd.read_csv("Free_rewards_begin.csv")
AfterTimes = pd.read_csv("Free_rewards_end.csv")

# Comput Mean Firing Rate Before and After within a temporal window
FiringRates = RatesOutSession(BeforeTimes,AfterTimes,STMtx,SampleRtPoints,NumNeu)

# Split Before and After Activity
NeuronsFiringRateBefore=FiringRates[0]
NeuronsFiringRateAfter=FiringRates[1]
# Firing Rate per Trial
NeuronsFiringRateSession = RateSession(STMtx,StartTrial,EndTrial)

NeuronsFiringRate = np.concatenate((NeuronsFiringRateBefore,NeuronsFiringRateSession,NeuronsFiringRateAfter),axis=0)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PCA CLASSIFICATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NeuronsFiringRate = stats.zscore(NeuronsFiringRate)
# Trials Transitions
LimitBefore = NeuronsFiringRateBefore.shape[0]                      #Trial Transition between Before and Session
LimitSession = LimitBefore+NeuronsFiringRateSession.shape[0]        #Trial Transision between Session and After

PopulationRateScore,PopulationRateState = PCA_analysis_3C(NeuronsFiringRate,LimitBefore,LimitSession)

#%%%%%%%%%%%%%%%%%%%%%%%%%% NEURON CLUSTER CLASSIFICATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N_cluster = 20 # Number of clusters for testing Kmean classification
Modelk, OptimalClusterK=Cluster_Evaluation(N_cluster,PopulationRateScore)

# Classifying the neurons in the different clusters
NeuronClusters = []
for i in range(OptimalClusterK):
    NeuronClusters.append(np.where(Modelk.labels_==i)[0])

# FIGURE: Plot of the different clusters in the PCA coefficient space
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
for i in range(OptimalClusterK):
    ax.scatter(PopulationRateScore[NeuronClusters[i],0],PopulationRateScore[NeuronClusters[i],1],PopulationRateScore[NeuronClusters[i],2])
ax.set_xlabel('Coeff1',labelpad=20)
ax.set_ylabel('Coeff2',labelpad=20)
ax.set_zlabel('Coeff3',labelpad=20)
plt.show()

# FIGURE: Mean Zscore firing rate of each cluster
plt.figure(figsize=(8,5))
for i in range(OptimalClusterK):
    KernelSize = 10
    KernelSignal = np.ones(KernelSize) / KernelSize
    SmoothSignal = np.convolve(NeuronsFiringRate[:,NeuronClusters[i]].mean(1), KernelSignal, mode='same')
    plt.plot(SmoothSignal,label=str(i))
plt.axvline(LimitBefore,c='k')
plt.axvline(LimitSession,c='k')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0),title="Clusters")
plt.xlabel("Trials")
plt.ylabel("FR (Zscore)")
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SPECIFIC CLUSTERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c1 = NeuronClusters[0]
c2 = NeuronClusters[1]
# cleaning clusters: All neurons must have at least one spike per trial
c1_reg=np.where(sum((NeuronsFiringRateSession[:,c1]==0))==0)[0]
c2_reg=np.where(sum((NeuronsFiringRateSession[:,c2]==0))==0)[0]
c1 = c1[c1_reg]
c2 = c2[c2_reg]
flag=False
if flag:
    np.save("D:\\_work_cestarellas\\Analysis\\PLRNN\\noautoencoder\\neuralactivity\\OFC\\CE17\\L6\\Test_reduction\\cluster_1.npy",c1)
    np.save("D:\\_work_cestarellas\\Analysis\\PLRNN\\noautoencoder\\neuralactivity\\OFC\\CE17\\L6\\Test_reduction\\cluster_2.npy",c2)


#%% Exploring Cluster Selected
# Generate Continuous signal
MaxTime=(np.nanmax(STMtx[-1,:])+0.05)*pq.s
NeuKernel = np.zeros(NumNeu)
InstRate = []
NeuralActive = []
# Bin_size for the instantaneous rate
InstBs = 0.02*pq.s                                                           # sample period
InstRt = (1/InstBs).rescale('Hz')                                            # sample rate
KernelSelect = np.linspace(10*InstBs.magnitude,10,num = 8)                   # possible kernel bandwidth
KernelWidth=0.4
for i in tqdm(range(NumNeu)):
    # Optimal Kernel for the Whole Session
    Train=neo.SpikeTrain(times = STMtx[~np.isnan(STMtx[:,i]),i]*pq.s,t_stop=MaxTime)    # selection of the Spikes in the session
    SpikeSet = Train[(Train>StartTrial[0])*(Train<EndTrial[-1])]
    OptimalFullKernel = el.statistics.optimal_kernel_bandwidth(SpikeSet.magnitude)         # computing convolution for the session
    MeanFiringRate = np.zeros(NumTrials)                                                # initialization of Mean Firing rate per each trial
    # Optimal Kernel for each Trial
    for i_t in range(NumTrials):
        SpikeSet1 = Train[(Train>StartTrial[i_t])*(Train<EndTrial[i_t])]
        SpikeSet1.t_start = StartTrial[i_t]*pq.s
        SpikeSet1.t_stop = EndTrial[i_t]*pq.s
        MeanFiringRate[i_t] = el.statistics.mean_firing_rate(SpikeSet1)                  # Mean firing rate per trial
    # Obtaining mean values from trials        
    NeuralActive.append(MeanFiringRate)
    #Final convolution for each unit/neuron
    InstRate.append(el.statistics.instantaneous_rate(Train, sampling_period=InstBs,kernel = el.kernels.GaussianKernel(KernelWidth*pq.s)))    

NeuronTime = np.linspace(0,MaxTime.magnitude,num = int(MaxTime/InstBs))
NeuralConvolution = np.array(InstRate[:]).squeeze().T
NeuralConvolution = stats.zscore(NeuralConvolution)

assert NeuralConvolution.shape[0]==NeuronTime.shape[0], 'Problems with the the bin_size of the convolution'


SpikeS=NeuralConvolution[:,c1]
#%% Example
time_length=np.linspace(0,SpikeS.shape[0]*0.02,SpikeS.shape[0])
plt.figure()
plt.plot(time_length,SpikeS[:,:])
plt.axvline(EndTrial[100]-StartTrial[0])
plt.axvline(EndTrial[101]-StartTrial[0])
plt.axvline(EndTrial[102]-StartTrial[0])
plt.axvline(EndTrial[103]-StartTrial[0])
plt.axvline(EndTrial[104]-StartTrial[0])
plt.xlim([570,620])
#%%
Tini=StartTrial-StartTrial[0]
Tend=EndTrial-StartTrial[0]
var_neu=np.full((StartTrial.shape[0], SpikeS.shape[1]), np.nan)
for i in range(NumTrials):
    t0=round(Tini[i]/0.02)
    t1=round(Tend[i]/0.02)
    var_neu[i,:]=np.var(SpikeS[t0:t1,:],axis=0)


#%% Create a heatmap
Var_norm=stats.zscore(var_neu)
plt.figure(figsize=(12,8))
sns.heatmap(Var_norm.T,cmap='viridis',vmax=12)
#%% Filter 2
c1f=np.where(var_neu.mean(axis=0)>0.5)[0]

plt.figure()
plt.plot(time_length,SpikeS[:,c1f])
plt.axvline(EndTrial[100]-StartTrial[0])
plt.axvline(EndTrial[101]-StartTrial[0])
plt.axvline(EndTrial[102]-StartTrial[0])
plt.axvline(EndTrial[103]-StartTrial[0])
plt.axvline(EndTrial[104]-StartTrial[0])
plt.xlim([570,620])
SpikeSS=SpikeS[:,c1f]



#%%
var_neu=np.full((StartTrial.shape[0], SpikeSS.shape[1]), np.nan)
for i in range(NumTrials):
    t0=round(Tini[i]/0.02)
    t1=round(Tend[i]/0.02)
    var_neu[i,:]=np.var(SpikeSS[t0:t1,:],axis=0)





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BEHAVIOUR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Classification of trials following the behabiour
GambleRewardTrials = BehData.index[(BehData['gamble']==1) & (BehData['REWARD']==1)]
GambleNoRewardTrials =  BehData.index[(BehData['gamble']==1) & (BehData['REWARD']==0)]
SafeRewardTrials = BehData.index[(BehData['safe']==1) & (BehData['REWARD']==1)]
SafeNoRewardTrials = BehData.index[(BehData['safe']==1) & (BehData['REWARD']==0)]
NoRespondingTrials = BehData.index[BehData["F"]==1]


# Smoothing the data for plotting
ScaleDecision=BehData["gamble"]+(BehData["gamble"]-1)
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
plt.xlim([0,185])
plt.yticks(ticks=[1.0,0.5,0.0])
plt.xlabel('Trials')
plt.ylabel("Animal's choice")
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STUDYING BOTH CLUSTERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
L_trial=BlockTrials[1]
#Block 1
T_GR1 = GambleRewardTrials[GambleRewardTrials<BlockTrials[0]]
T_GNR1 = GambleNoRewardTrials[GambleNoRewardTrials<BlockTrials[0]]
T_SR1 = SafeRewardTrials[SafeRewardTrials<BlockTrials[0]]
T_SNR1 = SafeNoRewardTrials[SafeNoRewardTrials<BlockTrials[0]]
#Block 2
T_GR2 = GambleRewardTrials[(GambleRewardTrials>=BlockTrials[0]) & (GambleRewardTrials<L_trial)]
T_GNR2 = GambleNoRewardTrials[(GambleNoRewardTrials>=BlockTrials[0]) & (GambleNoRewardTrials<L_trial)]
T_SR2 =SafeRewardTrials[(SafeRewardTrials>=BlockTrials[0]) & (SafeRewardTrials<L_trial)]
T_SNR2 =SafeNoRewardTrials[(SafeNoRewardTrials>=BlockTrials[0]) & (SafeNoRewardTrials<L_trial)]

FR_c1=NeuronsFiringRateSession[:,c1]
FR_c2=NeuronsFiringRateSession[:,c2]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SPIKE CONVOLUTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print('Neural Convolution')
print("Number of Neurons: ",NumNeu)
print("Number of Trials: ",NumTrials)

# Bin_size for the instantaneous rate
InstBs = 0.02*pq.s                                                           # sample period
InstRt = (1/InstBs).rescale('Hz')                                            # sample rate
KernelSelect = np.linspace(10*InstBs.magnitude,10,num = 8)                   # possible kernel bandwidth

NeuralConvolution,NeuralActive = SpikeConvolution(STMtx,NumNeu,NumTrials,StartTrial,EndTrial,KernelSelect) 


#%%%%%%%%%%%%%%%%%%%%%%%% SELECTION OF NEURONS FOR THE MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Selection of clusters:")
# Input number of clusters that you will select
ClusterAmount = int(input("Enter the number of clusters you want:"))
ClusterSelected = []
NeuronsSelected = np.empty(0,dtype=int)
# iterating till the range
for i in range(0, ClusterAmount):
    ele = int(input("Select Cluster Number:")) # Select the number of the clusters that you decided
    # adding the element
    ClusterSelected.append(ele) 
    NeuronsSelected = np.concatenate((NeuronsSelected,NeuronClusters[ele]))

NeuronsSelected.sort()
# Generation of the activity matrix for selected neurons
SpikesModel = NeuronsFiringRateSession[:,NeuronsSelected]
NeuronTrajectory = NeuralConvolution[:,NeuronsSelected] 
# Filter 2
# Taking active neurons during the session
BestNeurons=np.where(np.sum(SpikesModel==0,0)/SpikesModel.shape[0]*100<=5)[0]
# Spiking Neurons for the training model
ModelData = NeuronTrajectory[:,BestNeurons]




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DATA PREPROCESS FOR PLRNN MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Decision of the limit of trials to use. WE DON'T USE TRIALS AFTER NO-RESPONDING TRIALS
LastTrial = NoRespondingTrials[0]-3
# Number of External Inputs
NumberExternalInputs = 3
# Channels for each External Input
ActionChannel = 0
BigRewardChannel = 1
SmallRewardChannel = 2
# Values of External Channels
ActionValue = 1
BigRewardValue = BehData["gamble"]*4*BehData["REWARD"]
SmallRewardValue = BehData["safe"]*1*BehData["REWARD"]

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
    DataTrials.append(ModelData[StartPosition:EndPosition,:])
    # initialization input matrix
    InputMatrix = np.zeros((ModelData[StartPosition:EndPosition,:].shape[0],NumberExternalInputs))
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

# FIGURE: Contatenated trials (black dashed lines)
label_xticks = [ i for i in range(len(InitialTrial))]
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
plt.xticks(InitialTrial[0:-1:2],label_xticks[0:-1:2])
plt.xlabel('Trials')
plt.ylabel("Animal's choice")
plt.show()



# %%%%%%%%%%%%%%%%%%%%%%%% SELECTION TRAINING AND TEST TRIALS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ModelTrials = [i for i in range(len(InitialTrial))]
# Select specific trials that you will use to test the model
#TestTrials = [10, 15, 33, 38]
#TestTrials = [6, 25, 36, 45]
TestTrials = [13, 20, 40, 51]

Training2Test = list(np.array(TestTrials)-np.array([i+1 for i in range(len(TestTrials))]))
TrainingTrials = [i for i in ModelTrials if i not in TestTrials]

DataNeuronTest = [DataNeuronModel[i] for i in TestTrials] 
DataInputTest = [DataInputModel[i] for i in TestTrials] 

DataNeuronTraining = [DataNeuronModel[i] for i in TrainingTrials]
DataInputTraining = [DataInputModel[i] for i in TrainingTrials]

assert len(DataNeuronTest)==len(DataInputTest), "Input and Neurons doest not match in Test data"
assert len(DataNeuronTraining)==len(DataInputTraining), "Input and Neurons doest not match in Training data"

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  SAVE DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17/L6/Test3'
# Check/Create Path
if os.path.exists(save_path):
    os.chdir(save_path)
else:
    os.makedirs(save_path)
    os.chdir(save_path)

# Training Data
full_name = open("Training.pkl","wb")                                    # Name for training data
pickle.dump([DataNeuronTraining,DataInputTraining],full_name)            # Save train data
full_name.close()                                                        #close save instance 


# Test Data
full_name = open("Test.pkl","wb")                                        # Name for Testing data
pickle.dump([DataNeuronTest,DataInputTest],full_name)                    # Save Test data
full_name.close()                                                        #close save instance 

"""
MetaData: Dictionary with the following information
-Number of Neurons: Amount of neurons 
-TestTrials: the number of Test trials selected
-TrainingTrials: the number of Training Trials
-Training2Test: The Position of the Test trials in the Training Trials
-BeforeActivity: Neuronal Activity before the session
-AfterActivity: Neuronal Activity after the session
"""
MetaData = {}
MetaData["NumberNeurons"] = BestNeurons.shape[0]
MetaData["TestTrials"]=TestTrials
MetaData["TrainingTrials"] =TrainingTrials
MetaData["Training2Test"] = Training2Test

EndBeforeTime = int(StartTrial[0]/InstBs.magnitude)
StartAfterTime =  int(EndTrial[-1]/InstBs.magnitude)       # end trial index
MetaData["BeforeActivity"] = ModelData[:EndBeforeTime,:]
MetaData["AfterActivity"] = ModelData[StartAfterTime+1:,:]

full_name = open("Metadata.pkl","wb")                      # Name for training data
pickle.dump(MetaData,full_name)            # Save train data
#close save instance 
full_name.close()

# %%
