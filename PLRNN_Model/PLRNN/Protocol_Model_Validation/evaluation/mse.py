import torch as tc
from torch.linalg import pinv


def get_input(inputs, step, time_steps):
    #time_steps =
    if inputs is not None:
        input_for_step = inputs[step:(time_steps + step), :]
    else:
        input_for_step = None
    return input_for_step


def get_ahead_pred_obs(model, data, inputs, n_steps, indices):
    # dims
    T, dx = data.size()
    alpha = model.args['TF_alpha']

    # true data
    time_steps = T - n_steps
    x_data = data[:-n_steps, :].to(model.device)
    #s = inputs[:-n_steps, :].to(model.device)
    s = inputs.to(model.device)
    #s = tc.zeros((inputs.shape[0], inputs.shape[1])).to(model.device)


    # latent model
    lat = model.latent_model
    E, D = model.E, model.D

    # initial state. Uses x0 of the dataset
    # batch dim of z holds all T - n_steps time steps
    #if model.z0_model:
        #z = model.z0_model(x_data)
    #else:
        #dz = lat.d_z
        #z = tc.randn((time_steps, dz), device=model.device)

        # obs. model inv?
        #inv_obs = model.args['use_inv_tf']
        #B_PI = None
        #if inv_obs:
            #B = model.output_layer.weight
            #B_PI = pinv(B)
        #z = lat.teacher_force(z, x_data, B_PI)
    z_enc = model.E(x_data)
    z = z_enc
    if model.z0_model:
        z = model.z0_model(z_enc)
   # else:
        #z = model.E(x_data[])
        #dz = lat.d_z
        #z = tc.randn(size=(time_steps, dz), device=model.device)
        # z = self.teacher_force(z, x_[0], B_PI)
        #z = lat.teacher_force(z, z_enc, alpha)
    X_pred = tc.empty((n_steps, time_steps, dx), device=model.device)
    params = model.get_latent_parameters(indices)
    for step in range(n_steps):
        # latent step performs ahead prediction on every
        # time step here
        z = lat.latent_step(z, get_input(s, step+1, time_steps), *params)
        #x = model.output_layer(z)
        #X_pred[step] = x
        X_pred[step] = D(z[:, :z_enc.shape[1]]) ##because identity TF is used ?

    return X_pred

def construct_ground_truth(data, n_steps):
    T, dx = data.size()
    time_steps = T - n_steps
    X_true = tc.empty((n_steps, time_steps, dx))
    for step in range(1, n_steps + 1):
        X_true[step - 1] = data[step : time_steps + step]
    return X_true


def squared_error(x_pred, x_true):
    return tc.pow(x_pred - x_true, 2)

def test_trials_mse(model,data_true, data_model, n_steps):
    x_true=construct_ground_truth(data_true,n_steps).to(model.device)
    x_pred =construct_ground_truth(data_model,n_steps).to(model.device)
    mse = squared_error(x_pred, x_true).mean([1, 2]).cpu().numpy()
    return mse

@tc.no_grad()
def n_steps_ahead_pred_mse(model, data, inputs, n_steps, indices):
    x_pred = get_ahead_pred_obs(model, data, inputs, n_steps, indices)
    x_true = construct_ground_truth(data, n_steps).to(model.device)
    mse = squared_error(x_pred, x_true).mean([1, 2]).cpu().numpy()
    return mse
