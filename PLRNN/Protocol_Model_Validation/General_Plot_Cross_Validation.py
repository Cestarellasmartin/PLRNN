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
data_path = 'D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/results/OFC_red/Evaluation_Sheets'

# Load Train Trials Data
file_name = 'TrainEvaluation_CE17_red.csv'
load_file=os.path.join(data_path,file_name).replace('\\','/')
Traindf = pd.read_csv(load_file)

# Load Test Trials Data
file_name = 'TestEvaluation_CE17_red.csv'
load_file=os.path.join(data_path,file_name).replace('\\','/')
Testdf = pd.read_csv(load_file)

# Load Limiting Behaviour Data
file_name = 'LimitingBehaviour_CE17_red.csv'
load_file=os.path.join(data_path,file_name).replace('\\','/')
Limitdf = pd.read_csv(load_file)


#%% Effect of Lambda2 for specific Hidden Units
hidden_num=60
Test_SL = Testdf[(Testdf["Hiddn_Units"]==hidden_num)]
Train_SL = Traindf[(Traindf["Hiddn_Units"]==hidden_num)]
Limit_SL= Limitdf[(Limitdf["Hiddn_Units"]==hidden_num)]

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


# Limiting Behaviour: PSE
ax=Limit_SL.boxplot(column="PSE_SS",by=variable_plot,color='blue',figsize=(5,5))
ax.set_xlabel(variable_label)
ax.set_ylabel("PSE_SS")
plt.suptitle('')
plt.title('Hidden Units:'+str(hidden_num))
# Limiting Behaviour: KLx
ax=Limit_SL.boxplot(column="KLx_SS",by=variable_plot,color='blue',figsize=(5,5))
ax.set_xlabel(variable_label)
ax.set_ylabel("KLx_SS")
plt.suptitle('')
plt.title('Hidden Units:'+str(hidden_num))
plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))


#%% Effect of Hidden Units for specific Lamda2
Lambda_num=64
Test_SL = Testdf[(Testdf["Lambda2"]==Lambda_num)]
Train_SL = Traindf[(Traindf["Lambda2"]==Lambda_num)]
Limit_SL= Limitdf[(Limitdf["Lambda2"]==Lambda_num)]

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
# plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))
# NMSE
ax=Test_SL.boxplot(column="NMSE",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="NMSE",by=variable_plot,ax=ax,color='red')
ax.set_xlabel(variable_label)
ax.set_ylabel("MSE")
plt.suptitle('')
plt.title('Lambda2:'+str(Lambda_num))
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
# plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))
# PSE
ax=Test_SL.boxplot(column="PSE",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="PSE",by=variable_plot,ax=ax,color='red')
ax.set_xlabel(variable_label)
ax.set_ylabel("PSE")
plt.suptitle('')
plt.title('Lambda2:'+str(Lambda_num))
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
# plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))
# KLx
ax=Test_SL.boxplot(column="KLx",by=variable_plot,color='blue',figsize=(5,5))
Train_SL.boxplot(column="KLx",by=variable_plot,ax=ax,color='red')
ax.set_xlabel(variable_label)
ax.set_ylabel("KLx")
plt.suptitle('')
plt.title('Lambda2:'+str(Lambda_num))
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
# plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))

# Test acceptance
# Correlation Test Evaluation
ax=Test_SL.boxplot(column="CEvaluation",by=variable_plot,color='blue',figsize=(5,5))
ax.set_xlabel(variable_label)
ax.set_ylabel("Ratio Passed Test")
plt.suptitle('')
plt.title('Lambda2:'+str(Lambda_num))
red_patch = mpatches.Patch(color='red', label='Train')
blue_patch = mpatches.Patch(color='blue', label='Test')
# plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))

# Limiting Behaviour: PSE
ax=Limit_SL.boxplot(column="PSE_SS",by=variable_plot,color='blue',figsize=(5,5))
ax.set_xlabel(variable_label)
ax.set_ylabel("PSE_SS")
plt.suptitle('')
plt.title('Lambda2:'+str(Lambda_num))
# Limiting Behaviour: KLx
ax=Limit_SL.boxplot(column="KLx_SS",by=variable_plot,color='blue',figsize=(5,5))
ax.set_xlabel(variable_label)
ax.set_ylabel("KLx_SS")
plt.suptitle('')
plt.title('Lambda2:'+str(Lambda_num))
# plt.legend(handles=[blue_patch, red_patch],bbox_to_anchor=(1.5, 1.0))

