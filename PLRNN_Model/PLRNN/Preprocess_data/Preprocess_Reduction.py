'''
**Script: Preprocess_Gambling
**Author: Cristian Estarellas Martin
**Date: 10/2023 // UPDATE:06/2024

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
################################################
#%% Necessary Functions
################################################

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
# If you want to save the clusters selected you should change flag to TRUE
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
KernelWidth=0.1
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
    InstRate.append(el.statistics.instantaneous_rate(SpikeSet, sampling_period=InstBs,kernel = el.kernels.GaussianKernel(KernelWidth*pq.s)))    

NeuronTime = np.linspace(0,MaxTime.magnitude,num = int(MaxTime/InstBs))
NeuralConvolution = np.array(InstRate[:]).squeeze().T
S=round(StartTrial[0]/0.02)
E=round(EndTrial[-1]/0.02)
NeuralConvolution = NeuralConvolution[S:E,:]
NeuronTime=NeuronTime[S:E]
NeuralConvolution = stats.zscore(NeuralConvolution)

assert NeuralConvolution.shape[0]==NeuronTime.shape[0], 'Problems with the the bin_size of the convolution'

discard_neu=[]
for i in range(len(NeuralActive)):
    if np.where(NeuralActive[i]==0)[0].shape[0]>0:
        discard_neu.append(i)


#%%
# Selction of matrix for clustered neurons
c1_f=[i for i in c1 if i not in discard_neu]
SpikeS=NeuralConvolution[:,c1_f]
# Example of some neurons in consecutive trials
time_length=np.linspace(0,SpikeS.shape[0]*0.02,SpikeS.shape[0])
plt.figure()
plt.plot(time_length,SpikeS[:,23])
plt.xlim([600,608])

# Computing the variance of each neuron per trial
Tini=StartTrial-StartTrial[0]
Tend=EndTrial-StartTrial[0]
var_neu=np.full((StartTrial.shape[0], SpikeS.shape[1]), np.nan)
for i in range(NumTrials):
    t0=round(Tini[i]/0.02)
    t1=round(Tend[i]/0.02)
    var_neu[i,:]=np.var(SpikeS[t0:t1,:],axis=0)/(t1-t0)

# Summary of the Neural variance in the session
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
slope=[]
for i in range(var_neu.shape[1]):
    lr.fit(np.arange(var_neu[:,i].shape[0]).reshape(-1,1),var_neu[:,i])
    slope.append(lr.coef_[0])

slope=np.array(slope)
idx=np.argsort(slope)
var_sort=var_neu[:,idx]

plt.figure(figsize=(12,8))
sns.heatmap(var_sort.T,cmap='viridis',vmax=0.02)
plt.ylabel("Neurons")
plt.xlabel("Trials")
plt.title("Variance map")
plt.xlim([0,180])
# Mean Variance for each neuron
m_var=var_neu.mean(axis=0)
m_var.sort()
ratio_var=m_var/np.max(m_var)
plt.figure(figsize=(5,5))
plt.plot(ratio_var,'+')
plt.ylabel("Mean Variance")
plt.xlabel("Neurons")

#%% Filter 2
# Use neurons with higher variance
c2f=np.where(ratio_var>0.5)[0]
SpikeSS=SpikeS[:,c2f]

plt.figure()
plt.plot(time_length,SpikeSS)
plt.axvline(EndTrial[100]-StartTrial[0])
plt.axvline(EndTrial[101]-StartTrial[0])
plt.axvline(EndTrial[102]-StartTrial[0])
plt.axvline(EndTrial[103]-StartTrial[0])
plt.axvline(EndTrial[104]-StartTrial[0])
plt.xlim([570,620])

Tini=StartTrial-StartTrial[0]
Tend=EndTrial-StartTrial[0]
var_neu=np.full((StartTrial.shape[0], SpikeSS.shape[1]), np.nan)
for i in range(NumTrials):
    t0=round(Tini[i]/0.02)
    t1=round(Tend[i]/0.02)
    var_neu[i,:]=np.var(SpikeSS[t0:t1,:],axis=0)/(t1-t0)

# Summary of the Neural variance in the session

lr=LinearRegression()
slope=[]
for i in range(var_neu.shape[1]):
    lr.fit(np.arange(var_neu[:,i].shape[0]).reshape(-1,1),var_neu[:,i])
    slope.append(lr.coef_[0])

slope=np.array(slope)
idx=np.argsort(slope)
var_sort=var_neu[:,idx]

plt.figure(figsize=(12,8))
sns.heatmap(var_sort.T,cmap='viridis',vmax=0.02)
plt.ylabel("Neurons")
plt.xlabel("Trials")
plt.title("Variance map")
plt.xlim([0,180])

# Mean Variance for each neuron
m_var=var_neu.mean(axis=0)
m_var.sort()
ratio_var=m_var/np.max(m_var)
plt.figure(figsize=(5,5))
plt.plot(ratio_var,'+')
plt.ylabel("Mean Variance")
plt.xlabel("Neurons")

plt.figure()
plt.plot(time_length,SpikeSS[:,13])
plt.xlim([0,620])


#%% Exploring the new subset of neurons
Correlation_Neurons=np.corrcoef(SpikeSS, rowvar=False)
plt.figure(figsize=(12,8))
sns.heatmap(Correlation_Neurons.T,cmap='viridis')
plt.ylabel("Neurons")
plt.xlabel("Neurons")
plt.title("Correlation Session")

#Generating list for individual trials
NAct=[]
Tini=StartTrial-StartTrial[0]
Tend=RewardTime-StartTrial[0]
for i in range(NumTrials):
    t0=round(Tini[i]/0.02)
    t1=round(Tend[i]/0.02)
    NAct.append(SpikeSS[t0:t1,:])

# Compute the correlation between neurons in each trial
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
# Create a figure and axis using Seaborn
Corr_trial=[]
fig, ax = plt.subplots(figsize=(8, 6))
sns.set()
# Initialize the bar plot
Correlation_Neurons=np.corrcoef(NAct[0], rowvar=False)
heatmap = sns.heatmap(Correlation_Neurons, cmap='viridis', cbar=False, vmin=-1, vmax=1)
def update(frame):
    ax.clear()
    Correlation_Neurons=np.corrcoef(NAct[frame], rowvar=False)
    upp_idx=np.triu_indices_from(Correlation_Neurons,k=1)
    Corr_trial.append(Correlation_Neurons[upp_idx])
    heatmap = sns.heatmap(Correlation_Neurons, cmap='viridis', cbar=False, vmin=-1, vmax=1)
    # Set title and labels
    ax.set_title(f'Trial {frame}')
    ax.set_xlabel('Neurons')
    ax.set_ylabel('Neurons')
# Create the animation
animation = FuncAnimation(fig, update, frames=range(NumTrials), interval=500)

# Display the animation in the notebook
HTML(animation.to_jshtml())

#%%
pair_n=Corr_trial[0].shape[0]
C_trans=np.zeros((NumTrials,pair_n))
for i in range(NumTrials):
    C_trans[i,:]=Corr_trial[i]

PCACorrT = PCA(n_components=3)
PCACorrT.fit(C_trans)
CorrelationState = PCACorrT.transform(C_trans)
CorrelationScore = PCACorrT.components_.T

colors=np.linspace(0,180,180)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
scatter=ax.scatter(CorrelationState[0:180,0],CorrelationState[0:180,1],CorrelationState[0:180,2],c=colors,cmap='viridis')
cbar=plt.colorbar(scatter)
cbar.set_label('Trials')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

colors=np.linspace(0,180,180)
plt.figure(figsize=(10,10))
scatter2d=plt.scatter(CorrelationState[0:180,0],CorrelationState[0:180,1],c=colors,cmap='viridis')
cbar=plt.colorbar(scatter2d)
cbar.set_label('Trials')
plt.xlabel('PC1')
plt.ylabel('PC2')


Mcorr=C_trans.mean(axis=1)[0:180]
plt.figure()
plt.plot(Mcorr)
plt.xlabel('Trials')
plt.ylabel('Correlation')
plt.title('Average Population Correlations')
#Trial Selection: Decision
G_Trials = np.array(BehData.index[(BehData['gamble']==1)]).astype(int)
S_Trials = np.array(BehData.index[(BehData['safe']==1)]).astype(int)
G_Trials = G_Trials[np.where(G_Trials<180)]
S_Trials = S_Trials[np.where(S_Trials<180)]
# Distibution of Correlation for Gamble and Safe decisions
GMcorr=Mcorr[G_Trials]
SMcorr=Mcorr[S_Trials]
plt.figure()
plt.hist(GMcorr,alpha=0.5,density=True,label="Gamble")
plt.hist(SMcorr,alpha=0.5,density=True,label="Safe")
plt.xlabel("Correlation")
plt.ylabel("Density")
plt.title("Distribution for side decision")
plt.legend()

#Trial Selection: Previous output: Reward or No Reward
R_Trials = np.array(BehData.index[(BehData['REWARD']==1)]).astype(int)
NR_Trials = np.array(BehData.index[(BehData['REWARD']==0)]).astype(int)
R_Trials = R_Trials[np.where(R_Trials<179)]
NR_Trials = NR_Trials[np.where(NR_Trials<179)]
RMcorr=Mcorr[R_Trials+1]
NRMcorr=Mcorr[NR_Trials+1]
plt.figure()
plt.hist(RMcorr,alpha=0.5,density=True,label="Rewards")
plt.hist(NRMcorr,alpha=0.5,density=True,label="No Reward")
plt.xlabel("Correlation")
plt.ylabel("Density")
plt.title("Distribution for Reward trials")
plt.legend()

#Trial Selection: Previous output
GambleRewardTrials = BehData.index[(BehData['gamble']==1) & (BehData['REWARD']==1)]
GambleNoRewardTrials =  BehData.index[(BehData['gamble']==1) & (BehData['REWARD']==0)]
SafeRewardTrials = BehData.index[(BehData['safe']==1) & (BehData['REWARD']==1)]
SafeNoRewardTrials = BehData.index[(BehData['safe']==1) & (BehData['REWARD']==0)]
GambleRewardTrials=GambleRewardTrials[np.where(GambleRewardTrials<179)]
GambleNoRewardTrials=GambleNoRewardTrials[np.where(GambleNoRewardTrials<179)]
SafeRewardTrials=SafeRewardTrials[np.where(SafeRewardTrials<179)]
SafeNoRewardTrials=SafeNoRewardTrials[np.where(SafeNoRewardTrials<179)]

GRMcorr=Mcorr[GambleRewardTrials+1]
GNRMcorr=Mcorr[GambleNoRewardTrials+1]
SRMcorr=Mcorr[SafeRewardTrials+1]
SNRMcorr=Mcorr[SafeNoRewardTrials+1]
plt.figure()
plt.hist(GRMcorr,alpha=0.5,density=True,label="Reward")
plt.hist(GNRMcorr,alpha=0.5,density=True,label="No Reward")
plt.xlabel("Correlation")
plt.ylabel("Density")
plt.title("Distribution after Gamble decisions")
plt.legend()

plt.figure()
plt.hist(SRMcorr,alpha=0.5,density=True,label="Reward")
plt.hist(SNRMcorr,alpha=0.5,density=True,label="No Reward")
plt.xlabel("Correlation")
plt.ylabel("Density")
plt.title("Distribution after Safe decisions")
plt.legend()

#Example
Correlation_Neurons=np.corrcoef(NAct[22], rowvar=False)
plt.figure()
sns.heatmap(Correlation_Neurons, cmap='viridis', cbar=False, vmin=-1, vmax=1)
#%%
plt.figure()
plt.plot(NAct[28][:,3])
plt.plot(NAct[28][:,48])
plt.plot(NAct[28][:,46])
plt.figure()
plt.plot(NAct[22][:,3])
plt.plot(NAct[22][:,48])
plt.plot(NAct[22][:,46])

plt.figure()
plt.plot(NAct[23][:,3])
plt.plot(NAct[23][:,48])
plt.plot(NAct[23][:,46])
plt.figure()
plt.plot(NAct[26][:,3])
plt.plot(NAct[26][:,48])
plt.plot(NAct[26][:,46])

plt.figure()
plt.plot(NAct[70][:,3])
plt.plot(NAct[70][:,48])
plt.plot(NAct[70][:,46])
plt.figure()
plt.plot(NAct[75][:,3])
plt.plot(NAct[75][:,48])
plt.plot(NAct[75][:,46])
plt.figure()
plt.plot(NAct[78][:,3])
plt.plot(NAct[78][:,48])
plt.plot(NAct[78][:,46])

#%%
plt.figure()
color=np.linspace(0,NAct[21][:,3].shape[0],NAct[21][:,3].shape[0])
plt.scatter(NAct[21][:,3],NAct[21][:,48],c=color,cmap='viridis')
plt.figure()
color=np.linspace(0,NAct[22][:,3].shape[0],NAct[22][:,3].shape[0])
plt.scatter(NAct[22][:,3],NAct[22][:,48],c=color,cmap='viridis')
plt.figure()
color=np.linspace(0,NAct[23][:,3].shape[0],NAct[23][:,3].shape[0])
plt.scatter(NAct[23][:,3],NAct[23][:,48],c=color,cmap='viridis')

plt.figure()
color=np.linspace(0,NAct[56][:,3].shape[0],NAct[56][:,3].shape[0])
plt.scatter(NAct[56][:,3],NAct[56][:,48],c=color,cmap='viridis')
plt.figure()
color=np.linspace(0,NAct[75][:,3].shape[0],NAct[75][:,3].shape[0])
plt.scatter(NAct[75][:,3],NAct[75][:,48],c=color,cmap='viridis')
plt.figure()
color=np.linspace(0,NAct[95][:,3].shape[0],NAct[95][:,3].shape[0])
plt.scatter(NAct[95][:,3],NAct[95][:,48],c=color,cmap='viridis')
plt.figure()

gdec=55
sdec=86
color1=np.linspace(0,NAct[gdec][:,3].shape[0],NAct[gdec][:,3].shape[0])
color2=np.linspace(0,NAct[sdec][:,3].shape[0],NAct[sdec][:,3].shape[0])
plt.scatter(NAct[gdec][:,3],NAct[gdec][:,48],c=color1,cmap='viridis')
plt.scatter(NAct[sdec][:,3],NAct[sdec][:,48],c=color2,cmap='magma')

plt.figure()
for gdec in range(50,100):
    plt.scatter(NAct[gdec][:,3],NAct[gdec][:,48])
# Set title and labels
#%% Signal Properties Pre-Decision:
from scipy import signal
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# Classification of trials in Gamble or Safe decision
G_Trials = np.array(BehData.index[(BehData['gamble']==1)]).astype(int)
S_Trials = np.array(BehData.index[(BehData['safe']==1)]).astype(int)
G_Trials = G_Trials[np.where(G_Trials<180)]
S_Trials = S_Trials[np.where(S_Trials<180)]
# Selecting time series before reward point
Tini=StartTrial-StartTrial[0]
Tend=RewardTime-StartTrial[0]
NDec=[]
for i in range(NumTrials):
    t0=round(Tini[i]/0.02)
    t1=round(Tend[i]/0.02)
    NDec.append(SpikeSS[t0:t1,:])

NG=[NDec[i] for i in G_Trials]
NS=[NDec[i] for i in S_Trials]

norm_neu=[]
gambleC=[]
l_gamble=len(G_Trials)# number of training trials
num_neurons=SpikeSS.shape[1]
for i_neu in range(num_neurons):
    rs_training=[]
    # Train correlation
    for i_trial in range(l_gamble):
        L1=NG[i_trial][:,i_neu].shape[0]
        for i_extra in range(i_trial+1,l_gamble):
            L2=NG[i_extra][:,i_neu].shape[0]
            if L1==L2:
                T1=NG[i_trial][:,i_neu]
                T2=NG[i_extra][:,i_neu]
                eps=np.random.normal(0,1,len(T2))*0.001
                T2=T2+eps
                eps=np.random.normal(0,1,len(T2))*0.001
                T1=T1+eps
                correlation=np.corrcoef(T1,T2)[0,1]
                rs_training.append(correlation)
                if np.isnan(correlation):
                    plt.figure()
                    plt.plot(T1)
                    plt.plot(T2)
            elif L1<L2:
                X_rsample = signal.resample(NG[i_extra][:,i_neu],
                                    NG[i_trial].shape[0])
                T1 = NG[i_trial][:,i_neu]
                T2 = X_rsample
                eps=np.random.normal(0,1,len(T2))*0.001
                T2=T2+eps
                eps=np.random.normal(0,1,len(T2))*0.001
                T1=T1+eps
                correlation=np.corrcoef(T1,T2)[0,1]
                rs_training.append(correlation)
                if np.isnan(correlation):
                    plt.figure()
                    plt.plot(T1)
                    plt.plot(T2)
            else:
                X_rsample = signal.resample(NG[i_trial][:,i_neu],
                                    NG[i_extra].shape[0])
                T1 = X_rsample
                T2 = NG[i_extra][:,i_neu]
                eps=np.random.normal(0,1,len(T2))*0.001
                T2=T2+eps
                eps=np.random.normal(0,1,len(T2))*0.001
                T1=T1+eps
                correlation=np.corrcoef(T1,T2)[0,1]
                rs_training.append(correlation)
                if np.isnan(correlation):
                    plt.figure()
                    plt.plot(T1)
                    plt.plot(T2)
    data=np.array(rs_training)
    gambleC.append(data)
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
#%Testing correlation Test Trials


#%%
plt.figure()
i_neu=50
for i_trial in range(len(NG)):
    plt.plot(NG[i_trial][:,i_neu])

plt.figure()
plt.hist(gambleC[i_neu])


min_trials=np.min(np.array([NG[i].shape[0] for i in range(len(NG))]))
NG_r=np.zeros((min_trials,len(NG)))
for i in range(len(NG)):
    NG_r[:,i]=signal.resample(NG[i][:,i_neu],min_trials)

plt.figure()
for i_trial in range(len(NG)):
    plt.plot(NG_r[:,i_trial])

c_neu=np.corrcoef(NG_r)
plt.figure(figsize=(12,8))
sns.heatmap(c_neu,cmap='viridis',vmax=1,vmin=-1)
plt.ylabel("Trials")
plt.xlabel("Trials")
plt.title("Correlation Gamble Trials")




min_trials=np.min(np.array([NS[i].shape[0] for i in range(len(NS))]))
NS_r=np.zeros((min_trials,len(NS)))

for i in range(len(NS)):
    NS_r[:,i]=signal.resample(NS[i][:,i_neu],min_trials)

c_neu=np.corrcoef(NS_r)
plt.figure(figsize=(12,8))
sns.heatmap(c_neu,cmap='viridis',vmax=1,vmin=-1)
plt.ylabel("Trials")
plt.xlabel("Trials")
plt.title("Correlation Safe Trials")

plt.figure()
for i_trial in range(len(NS)):
    plt.plot(NS[i_trial][:,i_neu])


plt.figure()
for i_trial in range(len(NS)):
    plt.plot(NS_r[:,i_trial])
#%% Signal Properties Post-Decision:


plt.figure()
i_neu=0
for i_trial in range(16):
    plt.plot(NG[i_trial][:,i_neu])


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
