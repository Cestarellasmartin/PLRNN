import os
import pickle
import numpy as np
from tqdm import tqdm

import neo
import elephant as el
import quantities as pq
import scipy.stats as stats

def FullConvolution(STMtx,NumNeu,NumTrials,StartTrial,EndTrial,InstBs,KernelSelect):
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
    return NeuralConvolution,NeuralActive,NeuKernel

def SessionConvolution(STMtx,NumNeu,NumTrials,StartTrial,EndTrial,InstBs,KernelSelect):
    MaxTime=(np.nanmax(STMtx[-1,:])+0.05)*pq.s
    NeuKernel = np.zeros(NumNeu)
    InstRate = []
    NeuralActive = []
    for i in tqdm(range(NumNeu)):
        # Optimal Kernel for the Whole Session
        Train=neo.SpikeTrain(times = STMtx[~np.isnan(STMtx[:,i]),i]*pq.s,t_stop=MaxTime)    # selection of the Spikes in the session
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

def FixedConvolution(STMtx,NumNeu,InstBs,KernelSize):
    MaxTime=(np.nanmax(STMtx[-1,:])+0.05)*pq.s
    InstRate = []
    for i in tqdm(range(NumNeu)):
        # Optimal Kernel for the Whole Session
        Train=neo.SpikeTrain(times = STMtx[~np.isnan(STMtx[:,i]),i]*pq.s,t_stop=MaxTime)    # selection of the Spikes in the session
        #Final convolution for each unit/neuron
        InstRate.append(el.statistics.instantaneous_rate(Train, sampling_period=InstBs,kernel = el.kernels.GaussianKernel(KernelSize*pq.s)))    

    NeuronTime = np.linspace(0,MaxTime.magnitude,num = int(MaxTime/InstBs))
    NeuralConvolution = np.array(InstRate[:]).squeeze().T
    NeuralConvolution = stats.zscore(NeuralConvolution)

    assert NeuralConvolution.shape[0]==NeuronTime.shape[0], 'Problems with the the bin_size of the convolution'
    return NeuralConvolution

# FUNCTION Concatonate trials
def concatenate_list(lis,ax):
    res = lis[0]    # first element of the list
    Ls = [lis[0].shape[ax]] #first length of the list
    for i in range(1, len(lis)):
        Ls.append(lis[i].shape[ax])
        res = np.concatenate((res, lis[i]), axis=ax)
    return res, Ls

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

