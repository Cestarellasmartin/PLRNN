# %% Import Libraries
import os
import pickle
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import evaluation_Function_PSA as psa

import scipy.io
import scipy.stats as st
import scikit_posthocs as sp

import quantities as pq
from statannotations.Annotator import Annotator
plt.rcParams['font.size'] = 20

#%% Functions
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

def Plot_MixCorrelation(X,S):
    size=X.shape[0]
    CM=np.zeros((size,size))*np.nan
    for i in range(size):
        for j in range(size):
            if j<i:
                CM[i,j]=S[i,j]
            if j>=i:
                CM[i,j]=X[i,j]
    return CM

def Generation_RandomActivity(path):
    filename = "session.csv"
    os.chdir(path)
    # Load data
    # Open the Behaviour file
    BehData = pd.read_csv(filename)
    #Load matlab file (Spike Activity)
    data = scipy.io.loadmat('STMtx.mat')                                
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


    NewSTMtx = np.ones(STMtx.shape)*np.nan
    # Bin_size for the instantaneous rate
    InstBs = 0.02*pq.s                                                           # sample period
    InstRt = (1/InstBs).rescale('Hz')                                            # sample rate
    KernelSelect = np.linspace(10*InstBs.magnitude,10,num = 8)                   # possible kernel bandwidth

    for i in range(STMtx.shape[1]):
        Section=STMtx[~np.isnan(STMtx[:,i]),i]
        NewSection=np.sort(np.array([random.randrange(int(Section[0]*SampleRtPoints ),int(Section[-1]*SampleRtPoints ))/SampleRtPoints  for i in range(Section.shape[0])]))
        NewSTMtx[0:NewSection.shape[0],i]=NewSection

    # Convolution of Random Spikes
    RandomConvolution, _, _ = psa.FullConvolution(NewSTMtx,NumNeu,NumTrials,StartTrial,EndTrial,InstBs,KernelSelect) 
    return RandomConvolution

def Mean_ITI(X,ch_t):
    numtrials=len(X)
    numneurons=X[0].shape[1]
    X_Avg=[]
    for i in range(numtrials):
        X_mean = np.zeros((len(ch_t[i])-1,numneurons))
        for j in range(len(ch_t[i])-1):
            X_mean[j,:]=X[i][ch_t[i][j]:ch_t[i][j+1],:].mean(0)
        X_Avg.append(X_mean.mean(0))
    return X_Avg

def Mean_LB(X,ch_t,rand_iteration):
    numtrials=len(X)
    numneurons=X[0].shape[1]
    X_Avg=[]
    for i in range(numtrials):
        X_mean = np.zeros((len(ch_t[i])-1,numneurons))
        for j in range(len(ch_t[i])-1):
            Sum_X=np.zeros((1,numneurons))
            for ij in range(rand_iteration):
                diff_trials=ch_t[i][j+1]-ch_t[i][j]
                ini=random.randint(0,X[i].shape[0]-diff_trials)
                Sum_X=Sum_X+X[i][ini:ini+diff_trials,:].mean(0).numpy()        
            X_mean[j,:]=Sum_X/rand_iteration
        X_Avg.append(X_mean.mean(0))
    return X_Avg

def Mean_RA(X,ch_t,rand_iteration,numtrials,numneurons):
    X_Avg=[]
    for i in range(numtrials):
        X_mean = np.zeros((len(ch_t[i])-1,numneurons))
        for j in range(len(ch_t[i])-1):
            Sum_X=np.zeros((1,numneurons))
            for ij in range(rand_iteration):
                diff_trials=ch_t[i][j+1]-ch_t[i][j]
                ini=random.randint(0,X.shape[0]-diff_trials)
                Sum_X=Sum_X+X[ini:ini+diff_trials,0:numneurons].mean(0)       
            X_mean[j,:]=Sum_X/rand_iteration
        X_Avg.append(X_mean.mean(0))
    return X_Avg

def Correlation_Matrix(X):
    numneurons=X[0].shape[0]
    numtrials=len(X)
    Corr_Matrix=np.ones((numneurons,numneurons))*np.nan
    for i_neuron in range(numneurons):
        Series1=np.array([X[i][i_neuron] for i in range(numtrials)])
        for j_neuron in range(numneurons):
            Series2=np.array([X[i][j_neuron] for i in range(numtrials)])
            Corr_Matrix[i_neuron,j_neuron]=np.corrcoef(Series1,Series2)[0][1]
    return Corr_Matrix


def Stat_Comparison(T1,T2,T3,plot_info):
    alpha=0.05
    nt1=st.normaltest(T1,axis=0)[1]
    nt2=st.normaltest(T2,axis=0)[1]
    nt3=st.normaltest(T3,axis=0)[1]
    print([nt1,nt2,nt3])
    if (nt1<alpha) or (nt2<alpha) or (nt3<alpha):
        print("Non-parametric")
        F1=st.kruskal(T1,T2,T3)
        alpha=0.05
        if F1[1]<alpha:
            mc_Corr = sp.posthoc_dunn([T1,T2,T3],p_adjust='bonferroni')
            remove = np.tril(np.ones(mc_Corr.shape), k=0).astype("bool")
            mc_Corr[remove] = np.nan
            molten_df = mc_Corr.melt(ignore_index=False).reset_index().dropna()
            pairs = [(i[1]["index"]-1, i[1]["variable"]-1) for i in molten_df.iterrows()]
            p_values = [i[1]["value"] for i in molten_df.iterrows()]

            plt.figure()
            ax=sns.boxplot(data=[T1,T2,T3])
            annotator = Annotator(ax, pairs,data=[T1,T2,T3])
            annotator.configure(text_format="star", loc="inside")
            annotator.set_pvalues_and_annotate(p_values)
            plt.ylabel(plot_info["ylabel"])
            plt.xticks([0,1,2],labels=[plot_info["xtick1"],plot_info["xtick2"],plot_info["xtick3"]])
            plt.title(plot_info["title"])
        else:
            print("Not Significant change")
            plt.figure()
            ax=sns.boxplot(data=[T1,T2,T3])
            plt.ylabel(plot_info["ylabel"])
            plt.xticks([0,1,2],labels=[plot_info["xtick1"],plot_info["xtick2"],plot_info["xtick3"]])
            plt.title(plot_info["title"])

    if (nt1>alpha) or (nt2>alpha) or (nt3>alpha):
        print("Parametric")
        F1=st.f_oneway(T1,T2,T3)
        alpha=0.05
        if F1[1]<alpha:
            mc_Corr = sp.posthoc_tukey([T1,T2,T3])
            remove = np.tril(np.ones(mc_Corr.shape), k=0).astype("bool")
            mc_Corr[remove] = np.nan
            molten_df = mc_Corr.melt(ignore_index=False).reset_index().dropna()
            pairs = [(i[1]["index"]-1, i[1]["variable"]-1) for i in molten_df.iterrows()]
            p_values = [i[1]["value"] for i in molten_df.iterrows()]

            plt.figure()
            ax=sns.boxplot(data=[T1,T2,T3])
            annotator = Annotator(ax, pairs,data=[T1,T2,T3])
            annotator.configure(text_format="star", loc="inside")
            annotator.set_pvalues_and_annotate(p_values)
            plt.ylabel(plot_info["ylabel"])
            plt.xticks([0,1,2],labels=[plot_info["xtick1"],plot_info["xtick2"],plot_info["xtick3"]])
            plt.title(plot_info["title"])
        else:
            print("Not Significant change")
            plt.figure()
            ax=sns.boxplot(data=[T1,T2,T3])
            plt.ylabel(plot_info["ylabel"])
            plt.xticks([0,1,2],labels=[plot_info["xtick1"],plot_info["xtick2"],plot_info["xtick3"]])
            plt.title(plot_info["title"])
    
    

def correlation_distance(X,S):
    k_comb=0
    size=X.shape[0]
    Dist_Corr = []
    for i in range(size):
        for j in range(i+1,size):
            k_comb+=1
            Dist_Corr.append(np.abs(X[i,j]-S[i,j]))
    distance = np.array(Dist_Corr)
    return distance 

def Correlation_Comparison(M,D,R,plot_info):
    size=M.shape[0]
    vector1=[]
    vector2=[]
    alpha=0.05
    for i in range(size):
        for j in range(i+1,size):
            vector1.append(np.abs(M[i,j]-D[i,j]))
            vector2.append(np.abs(M[i,j]-R[i,j]))
    vector1=np.array(vector1)
    vector2=np.array(vector2)
    nt1=st.normaltest(vector1)[1]
    nt2=st.normaltest(vector2)[1]
    if (nt1<alpha) or (nt2<alpha):
        print("Non-Parametric")
        test_result=st.wilcoxon(vector1,vector2)
    else:
        print("Parametric")
        test_result=st.ttest_rel(vector1,vector2)
    
    pairs = [(0,1)]
    p_values = [test_result[1]]
    plt.figure()
    ax=sns.boxplot(data=[vector1,vector2])
    annotator = Annotator(ax, pairs,data=[vector1,vector2])
    annotator.configure(text_format="star", loc="inside")
    annotator.set_pvalues_and_annotate(p_values)     
    plt.ylabel(plot_info["ylabel"])
    plt.xticks([0,1],labels=[plot_info["xtick1"],plot_info["xtick2"]])
    plt.title(plot_info["title"])
            