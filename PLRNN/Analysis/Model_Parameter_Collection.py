'''
Cristian Estarellas.09-2024
Subtraction of model parameters for dynamical object analysis. SCY_FI algorithm


'''

# %% Import Libraries
import os
import pickle
import numpy as np
import torch as tc
from bptt.models import Model
#%% Main
mpath = str(input("Write model path:"))
mpath = mpath.replace('\\','/')
file=open(os.path.join(mpath,'hypers.pkl').replace("\\","/"),'rb')
hyper=pickle.load(file)
file.close()
last_epoch = hyper['n_epochs']-5

m = Model()
m.init_from_model_path(mpath, epoch=last_epoch)
m.eval()

print(repr(m), f"\nNumber of Parameters: {m.get_num_trainable()}") # get trainable parameters
At, W1t, W2t, h1t, h2t, Ct = m.get_latent_parameters()

# Transform tensor to numpy format
A = At.detach().numpy()
W2 = W2t.detach().numpy()
W1 = W1t.detach().numpy()
h1 = h1t.detach().numpy()
h2 = h2t.detach().numpy()
C = Ct.detach().numpy()

# Save Parameters
save_path="D:\_work_cestarellas\Analysis\PLRNN\SCYFI\data"
save_path = save_path.replace('\\','/')

name_file="Model_Parameters_CE17_red_003.pkl"

full_name = open(os.path.join(save_path,name_file).replace("\\","/"),"wb")                                    # Name for training data
pickle.dump([A,W2,W1,h1,h2,C],full_name)            # Save train data
full_name.close()    
# %%
