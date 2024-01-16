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
data_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/Tuning_OFC_CE17_L6_221008/Session_Test/Evaluation_Sheets'

# Load Train Trials Data
file_name = 'TrainEvaluation_CE17_L6.csv'
load_file=os.path.join(data_path,file_name).replace('\\','/')
Traindf = pd.read_csv(load_file)

# Load Test Trials Data
file_name = 'TestEvaluation_CE17_L6.csv'
load_file=os.path.join(data_path,file_name).replace('\\','/')
Testdf = pd.read_csv(load_file)

#%% Effect of Lambda2 for specific Hidden Units
hidden_num=128
Test_SL = Testdf[(Testdf["Hiddn_Units"]==hidden_num) & (Testdf["Sequence_Length"]==400)]
Train_SL = Traindf[(Traindf["Hiddn_Units"]==hidden_num) & (Traindf["Sequence_Length"]==400)]

variable_plot = "Lambda2"
variable_label = "Lambda2"
# Correlation
ax=Test_SL.boxplot(column="Correlation",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="Correlation",by=variable_plot,ax=ax,color='red')
ax.set_xlabel(variable_label)
ax.set_ylabel("Correlation")
ax.set_ylim([0,1])
plt.suptitle('')
plt.title('Hidden Units:'+str(hidden_num))
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))
# NMSE
ax=Test_SL.boxplot(column="NMSE",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="NMSE",by=variable_plot,ax=ax,color='red')
ax.set_xlabel(variable_label)
ax.set_ylabel("MSE")
plt.suptitle('')
plt.title('Hidden Units:'+str(hidden_num))
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))
# PSE
ax=Test_SL.boxplot(column="PSE",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="PSE",by=variable_plot,ax=ax,color='red')
ax.set_xlabel(variable_label)
ax.set_ylabel("PSE")
plt.suptitle('')
plt.title('Hidden Units:'+str(hidden_num))
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))
# KLx
ax=Test_SL.boxplot(column="KLx",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="KLx",by=variable_plot,ax=ax,color='red')
ax.set_xlabel(variable_label)
ax.set_ylabel("KLx")
plt.suptitle('')
plt.title('Hidden Units:'+str(hidden_num))
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))

# Test acceptance
# Correlation Test Evaluation
ax=Test_SL.boxplot(column="CEvaluation",by=variable_plot,color='blue',figsize=(5,5))
ax.set_xlabel(variable_label)
ax.set_ylabel("Ratio Passed Test")
plt.suptitle('')
plt.title('Hidden Units:'+str(hidden_num))
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))

#%% Effect of Hidden Units for specific Lamda2
Lambda_num=8.0
Test_SL = Testdf[(Testdf["Lambda2"]==Lambda_num) & (Testdf["Sequence_Length"]==400)]
Train_SL = Traindf[(Traindf["Lambda2"]==Lambda_num) & (Traindf["Sequence_Length"]==400)]

variable_plot = "Hiddn_Units"
variable_label = "Hidden Units"
# Correlation
ax=Test_SL.boxplot(column="Correlation",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="Correlation",by=variable_plot,ax=ax,color='red')
ax.set_xlabel(variable_label)
ax.set_ylabel("Correlation")
ax.set_ylim([0,1])
plt.suptitle('')
plt.title('Lambda2:'+str(Lambda_num))
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))
# NMSE
ax=Test_SL.boxplot(column="NMSE",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="NMSE",by=variable_plot,ax=ax,color='red')
ax.set_xlabel(variable_label)
ax.set_ylabel("MSE")
plt.suptitle('')
plt.title('Lambda2:'+str(Lambda_num))
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))
# PSE
ax=Test_SL.boxplot(column="PSE",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="PSE",by=variable_plot,ax=ax,color='red')
ax.set_xlabel(variable_label)
ax.set_ylabel("PSE")
plt.suptitle('')
plt.title('Lambda2:'+str(Lambda_num))
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))
# KLx
ax=Test_SL.boxplot(column="KLx",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="KLx",by=variable_plot,ax=ax,color='red')
ax.set_xlabel(variable_label)
ax.set_ylabel("KLx")
plt.suptitle('')
plt.title('Lambda2:'+str(Lambda_num))
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))

# Test acceptance
# Correlation Test Evaluation
ax=Test_SL.boxplot(column="CEvaluation",by=variable_plot,color='blue',figsize=(5,5))
ax.set_xlabel(variable_label)
ax.set_ylabel("Ratio Passed Test")
plt.suptitle('')
plt.title('Lambda2:'+str(Lambda_num))
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))
