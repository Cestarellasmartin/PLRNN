from bptt.models import Model
import torch as tc
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

def plot_multiple_lines(trajectory, save_path, name):
    n_traces = trajectory.shape[1]
    
    # create traces for each line
    traces = [go.Scatter(y=trajectory[:,i], mode='lines', name=f'Line {i+1}') for i in range(n_traces)]

    # create data object with all traces
    data = traces

    # create layout for the plot
    layout = go.Layout(title='Multiple Line Plot', xaxis=dict(title='X-axis'), yaxis=dict(title='Y-axis'))

    # create figure object with data and layout, and display the plot
    fig = go.Figure(data=data, layout=layout)
    fig.write_html(save_path + "/" + name + ".html")

def load(path, obj=None):
    if obj is None:
        with open(path, 'rb') as f:
            ret = pickle.load(f)
    else:
        npath = os.path.join(path, obj)
        with open(npath, 'rb') as f:
            ret = pickle.load(f)
    return ret

def relu(z):
    return np.maximum(z, 0)

def get_named_parameters(model):
    named_parameters = {}
    for name, param in model.latent_model.named_parameters():
        name = name.split(".")[-1]
        named_parameters[name] = param
    return named_parameters

def get_params_as_list(Model):
    mparams = get_named_parameters(Model)
    params = [i.detach().numpy() for i in mparams.values()]
    return params

def extract_pm_set(params, i):
    return (params[0], params[1][i], params[2][i], params[3], params[4], params[5])

def latent_step(A, W1, W2, h1, h2, C, z, s=None):
    noise = np.random.multivariate_normal(cov=np.diag(np.ones((z.size,))*0.0001), mean=np.zeros((z.size,)), size=1)
    noise = np.zeros(noise.shape)
    if s is None: s = np.zeros(A.shape)
    Wz = W1@z
    z_act = relu(Wz+h1) - relu(Wz)
    z_next = A*z + W2@z_act + h2 + C@s + noise
    return z_next

def gen_traj(Model, idx, T, inps=None, z0=None):
    params = get_params_as_list(Model)
    if Model.latent_model.W_trial:
        pmset = extract_pm_set(params, idx)
    else:
        pmset = params[:6]
    dz, dh, ds = pmset[0].size, pmset[3].size, pmset[5].shape[1] #size(A), size(h1), shape_1(C)
    if z0 is None:
        z0 = np.random.rand(dz)*2 - 1
    if inps is None:
        inps = np.zeros((T, ds))
    Z = np.zeros((T, dz))
    # Z[0] = z0
    for t in range(1, T):
        Z[t,:] = latent_step(*pmset, Z[t-1], inps[t])
    return Z

def generate_all_trajectories(Model, T, inps=None, data=None):
    #TODO: Stationary case
    n_trials = Model.latent_model.ntrials
    Zs = []
    for i in range(n_trials):
        if data is None:
            z0 = None
        else:
            z0 = data[i][0,:]
        Zs.append(gen_traj(Model, i, T, inps, z0))
    return Zs

def get_Ws(Model):
    W1, W2 = get_params_as_list(Model)[1:3]
    return W1, W2







#%%
#Model path
# dir_path = r"/zi-flstorage/Max.Thurm/PhD/nonstationary-trial-data-autoencoder/results/example_data_lorenz/H512/001/"
# save_path = r"/zi-flstorage/Max.Thurm/PhD/nonstationary-trial-data-autoencoder/results/example_data_lorenz/H512/"

# dir_path = r"/zi-flstorage/Max.Thurm/PhD/nonstationary-trial-data-autoencoder/results/bursts_data/Anealing_try_01H32_lambda_0.1/001"
# save_path = r"/zi-flstorage/Max.Thurm/PhD/nonstationary-trial-data-autoencoder/results/bursts_data/Anealing_try_01H32_lambda_0.1/"

# dir_path = r"/zi-flstorage/Max.Thurm/PhD/nonstationary-trial-data-autoencoder/results/example_data/H256/001/"
# save_path = r"/zi-flstorage/Max.Thurm/PhD/nonstationary-trial-data-autoencoder/results/example_data/H256/"

# dir_path = r"/zi-flstorage/Max.Thurm/PhD/nonstationary-trial-data-autoencoder/results/bursts_data/Anealing3_try_01H08_lambda_0.01_process_noise_0.01/001"
# save_path = r"/zi-flstorage/Max.Thurm/PhD/nonstationary-trial-data-autoencoder/results/bursts_data/Anealing3_try_01H08_lambda_0.01_process_noise_0.01/"

dir_path = r"/home/maxingo.thurm/nonstationary-trial-data-autoencoder/results/BPTT_collection_17_data/Anealing_01H20_lambda_0.01/006"
save_path = r"/home/maxingo.thurm/nonstationary-trial-data-autoencoder/results/BPTT_collection_17_data/Anealing_01H20_lambda_0.01/"

#Load model
model = Model()
model.init_from_model_path(dir_path, 90000)

#%%
mparams = get_named_parameters(model)
params = get_params_as_list(model)
print(params[1].shape)
pmset = extract_pm_set(params, 0)
T = 1000
trajectory = generate_all_trajectories(model, T)
trajectory = np.concatenate(tuple([i.T for i in trajectory]), axis=1).T

# trajectory = model.D(tc.tensor(trajectory, dtype=tc.float32)).detach().numpy()



#Plotly plot
plot_multiple_lines(trajectory, save_path, 'test_data_set')

print()





# aeparams = get_named_parameters(model.E)
# xnd = aeparams['weight'].shape[-1]
