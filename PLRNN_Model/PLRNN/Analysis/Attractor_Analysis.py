# %% Import Libraries
import os
import pickle
import numpy as np
import pandas as pd
import torch as tc
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
from tqdm import tqdm

from bptt.models import Model
from function_modules import model_anafunctions as func

plt.rcParams['font.size'] = 20

import h5py
#%%% Loading Data

path_folder='D:\_work_cestarellas\Analysis\PLRNN\SCYFI\Attractors\CE17'
path_folder=path_folder.replace('\\','/')
file='FP_CE17_00_01.npz'
path_file = os.path.join(path_folder,file).replace('\\','/')

Dyn_Objects=np.load(path_file)
FP=Dyn_Objects["FP"]                # Matrix of Fixed Points (ZxF): Z-> laten states; F: -> Fixed Points
EV=Dyn_Objects["EV"]                # Matrix of Eigen Values (ZxF)

# %% Stability of FP

