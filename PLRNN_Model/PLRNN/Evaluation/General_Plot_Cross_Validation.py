# %% Import Libraries
import numpy as np
import torch as tc
import matplotlib.pyplot as plt
import os
from function_modules import model_anafunctions as func
import pandas as pd
import matplotlib.patches as mpatches
import pandas as pd
from mpl_toolkits import mplot3d
plt.rcParams['font.size'] = 20

#%% Set Paths: Data & Model
data_path = 'D:/_work_cestarellas/Analysis/PLRNN/Analysis_PLRNN/Data_Results/Cross_Validation_Data/CE17_221008'

# Load Test Trials Data
file_name = 'TestTrial_subset_CE17_221008.csv'
load_file=os.path.join(data_path,file_name).replace('\\','/')
Testdf = pd.read_csv(load_file)

# Load LimitBehaviour Data
file_name = 'LimitBehaviour_subset_CE17_221008.csv'
load_file=os.path.join(data_path,file_name).replace('\\','/')
Data = pd.read_csv(load_file)

#%% Effect of the Sequence Length

Test_SL = Testdf[(Testdf["Lambda2"]==1.0) & (Testdf["Hidden"]==512) & (Testdf["Condition"]=="test")]
Train_SL = Testdf[(Testdf["Lambda2"]==1.0) & (Testdf["Hidden"]==512) & (Testdf["Condition"]=="train")]
LB_SL = Data[(Data["Lambda2"]==1.0) & (Data["Hidden"]==256)]

variable_plot = "SequenceLength"
variable_label = "Sequence Length"

ax=Test_SL.boxplot(column="CorrDistance",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="CorrDistance",by=variable_plot,ax=ax,color='red')
ax.set_xlabel(variable_label)
ax.set_ylabel("Distance Correlation")
ax.set_ylim([0,1])
plt.suptitle('')
plt.title('')
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))

ax=Test_SL.boxplot(column="PSE",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="PSE",by=variable_plot,ax=ax,color='red')
ax.set_xlabel(variable_label)
ax.set_ylabel("PSE")
plt.suptitle('')
plt.title('')
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))

# Mean Square Error (MSE)
ax1=Test_SL.boxplot(column="NMSE",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="NMSE",by=variable_plot,ax=ax1,color='red')
ax1.set_xlabel(variable_label)
ax1.set_ylabel("MSE")
plt.title('')
plt.suptitle('')
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))

# Behaviour Limit Kullback Leibar Divergence
ax=LB_SL.boxplot(column="KLdist",by=variable_plot,color='blue',figsize=(5,5))
ax.set_xlabel(variable_label)
ax.set_ylabel("KL divergence")
plt.suptitle('')
plt.title('')

# Behaviour Limit Kullback Leibar Divergence
ax=LB_SL.boxplot(column="PSE_NS",by=variable_plot,color='blue',figsize=(5,5))
ax.set_xlabel(variable_label)
ax.set_ylabel("PSE_NS")
plt.suptitle('')
plt.title('')

# Behaviour Limit Kullback Leibar Divergence
ax=LB_SL.boxplot(column="PSE_SS",by=variable_plot,color='blue',figsize=(5,5))
ax.set_xlabel(variable_label)
ax.set_ylabel("PSE_SS")
plt.suptitle('')
plt.title('')

#%% Effect Lambda2
Test_SL = Testdf[(Testdf["Hidden"]==256.0) & (Testdf["Lambda1"]<100) & (Testdf["SequenceLength"]==400) & (Testdf["Condition"]=="test")]
Train_SL = Testdf[(Testdf["Hidden"]==256.0) & (Testdf["Lambda1"]<100) & (Testdf["SequenceLength"]==400) & (Testdf["Condition"]=="train")]
LB_SL = Data[(Data["Hidden"]==256.0) & (Data["Lambda1"]<100) & (Data["SequenceLength"]==400)]
Test_A = Testdf[(Testdf["Hidden"]==256.0) & (Testdf["Lambda1"]==100) & (Testdf["SequenceLength"]==400) & (Testdf["Condition"]=="test")]
Train_A = Testdf[(Testdf["Hidden"]==256.0) & (Testdf["Lambda1"]==100) & (Testdf["SequenceLength"]==400) & (Testdf["Condition"]=="train")]
LB_A = Data[(Data["Hidden"]==256.0) & (Data["Lambda1"]==100) & (Data["SequenceLength"]==400)]
variable_plot = "Lambda2"
variable_label = "Lambda 2"

ax=Test_SL.boxplot(column="CorrDistance",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="CorrDistance",by=variable_plot,ax=ax,color='red')
ax.axhline(np.array(Test_A["CorrDistance"])[0],color='blue')
ax.axhline(np.array(Train_A["CorrDistance"])[0],color='red')
ax.set_xlabel(variable_label)
ax.set_ylabel("Distance Correlation")
ax.set_ylim([0,1])
plt.suptitle('')
plt.title('')
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))

ax=Test_SL.boxplot(column="PSE",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="PSE",by=variable_plot,ax=ax,color='red')
ax.axhline(np.array(Test_A["PSE"])[0],color='blue')
ax.axhline(np.array(Train_A["PSE"])[0],color='red')
ax.set_xlabel(variable_label)
ax.set_ylabel("PSE")
plt.suptitle('')
plt.title('')
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))

# Mean Square Error (MSE)
ax1=Test_SL.boxplot(column="NMSE",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="NMSE",by=variable_plot,ax=ax1,color='red')
ax1.axhline(np.array(Test_A["NMSE"])[0],color='blue')
ax1.axhline(np.array(Train_A["NMSE"])[0],color='red')
ax1.set_xlabel(variable_label)
ax1.set_ylabel("MSE")
plt.title('')
plt.suptitle('')
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))

# Behaviour Limit Kullback Leibar Divergence
ax=LB_SL.boxplot(column="KLdist",by=variable_plot,color='blue',figsize=(5,5))
ax.axhline(np.array(LB_A["KLdist"])[0],color='blue')
ax.set_xlabel(variable_label)
ax.set_ylabel("KL divergence")
plt.suptitle('')
plt.title('')

# Behaviour Limit Kullback Leibar Divergence
ax=LB_SL.boxplot(column="PSE",by=variable_plot,color='blue',figsize=(5,5))
ax.axhline(np.array(LB_A["PSE"])[0],color='blue')
ax.set_xlabel(variable_label)
ax.set_ylabel("PSE")
plt.suptitle('')
plt.title('')

# # Behaviour Limit Kullback Leibar Divergence
# ax=LB_SL.boxplot(column="PSE_SS",by=variable_plot,color='blue',figsize=(5,5))
# ax.set_xlabel(variable_label)
# ax.set_ylabel("PSE_SS")
# plt.suptitle('')
# plt.title('')

#%% Effect Hidden Units
Test_SL = Testdf[(Testdf["Lambda2"]==8.0) & (Testdf["SequenceLength"]==400) & (Testdf["Condition"]=="test")]
Train_SL = Testdf[(Testdf["Lambda2"]==8.0) & (Testdf["SequenceLength"]==400) & (Testdf["Condition"]=="train")]
LB_SL = Data[(Data["Lambda2"]==8.0) & (Data["SequenceLength"]==400)]

variable_plot = "Hidden"
variable_label = "Hidden Units"

ax=Test_SL.boxplot(column="CorrDistance",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="CorrDistance",by=variable_plot,ax=ax,color='red')
ax.set_xlabel(variable_label)
ax.set_ylabel("Distance Correlation")
ax.set_ylim([0,1])
plt.suptitle('')
plt.title('')
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))

ax=Test_SL.boxplot(column="PSE",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="PSE",by=variable_plot,ax=ax,color='red')
ax.set_xlabel(variable_label)
ax.set_ylabel("PSE")
plt.suptitle('')
plt.title('')
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))

# Mean Square Error (MSE)
ax1=Test_SL.boxplot(column="NMSE",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="NMSE",by=variable_plot,ax=ax1,color='red')
ax1.set_xlabel(variable_label)
ax1.set_ylabel("MSE")
plt.title('')
plt.suptitle('')
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))

# Behaviour Limit Kullback Leibar Divergence
ax=LB_SL.boxplot(column="KLdist",by=variable_plot,color='blue',figsize=(5,5))
ax.set_xlabel(variable_label)
ax.set_ylabel("KL divergence")
plt.suptitle('')
plt.title('')

# Behaviour Limit Kullback Leibar Divergence
ax=LB_SL.boxplot(column="PSE_NS",by=variable_plot,color='blue',figsize=(5,5))
ax.set_xlabel(variable_label)
ax.set_ylabel("PSE_NS")
plt.suptitle('')
plt.title('')

# Behaviour Limit Kullback Leibar Divergence
ax=LB_SL.boxplot(column="PSE_SS",by=variable_plot,color='blue',figsize=(5,5))
ax.set_xlabel(variable_label)
ax.set_ylabel("PSE_SS")
plt.suptitle('')
plt.title('')
#%%
T1_plot=Testdf[(Testdf["Lambda2"]==1.0) & (Testdf["Condition"]=="test") & (Testdf["SequenceLength"]==400) & (Testdf["Hidden"]!=64)]
T8_plot=Testdf[(Testdf["Lambda2"]==8.0) & (Testdf["Condition"]=="test") & (Testdf["SequenceLength"]==400)]
T64_plot=Testdf[(Testdf["Lambda2"]==64.0) & (Testdf["Condition"]=="test") & (Testdf["SequenceLength"]==400)]
T256_plot=Testdf[(Testdf["Lambda2"]==256.0) & (Testdf["Condition"]=="test") & (Testdf["SequenceLength"]==400)]

ax=T1_plot.boxplot(column="NMSE",by="Hidden",color='blue',figsize=(5,5))
T8_plot.boxplot(column="NMSE",by="Hidden",ax=ax,color='red')
T64_plot.boxplot(column="NMSE",by="Hidden",ax=ax,color='green')
T256_plot.boxplot(column="NMSE",by="Hidden",ax=ax,color='purple')
ax.set_xlabel("Hidden")
ax.set_ylabel("NMSE")
plt.suptitle('')
plt.title('')
blue_patch = mpatches.Patch(color='blue', label='Lambda2: 1')
red_patch = mpatches.Patch(color='red', label='Lambda2: 8')
green_patch = mpatches.Patch(color='green', label='Lambda2: 64')
purple_patch = mpatches.Patch(color='purple', label='Lambda2: 256')
plt.legend(handles=[blue_patch, red_patch, green_patch, purple_patch],bbox_to_anchor=(1.5, 1.0))
