import function_modules.model_anafunctions as func
import numpy as np

def measurement_generator(mpath1):
    A, W2, W1,h2, h1, C, m = func.W_matrix_generation(mpath1)
    num_hu=W2.shape[1]

    ## computing variance of each hidden unit [matrix 2D - latent states/neural dimensions and temporal dimension(trials)]
    # W2
    var_W2=np.asarray([W2[:,i,:].var(1).mean() for i in range(num_hu)])
    # W1
    var_W1=np.asarray([W1[i,:,:].var(1).mean() for i in range(num_hu)])

    variance = [var_W2.mean(),var_W1.mean()] # measurement 1

    var_info = []
    ## Computing informative and non-informative neurons
    # W2 
    h2=np.histogram(var_W2,range=(0,0.005),bins=20)

    units_noinfoW2 = np.where(var_W2<h2[1][1])[0]

    units_infoW2 = np.where(var_W2>h2[1][1])[0]

    var_info.append(var_W2[units_infoW2].mean()) 

    # W1
    h1=np.histogram(var_W1,range=(0,0.005),bins=20)

    units_noinfoW1 = np.where(var_W1<h1[1][1])[0]

    units_infoW1 = np.where(var_W1>h1[1][1])[0]

    var_info.append( var_W1[units_infoW1].mean()) 

    Noinfo_hu = [units_noinfoW2,units_noinfoW1]
    Info_hu = [units_infoW2,units_infoW1]

    # Computing distance between W parameters
    temporal_length = W2.shape[2]
    lat_states = W2.shape[0]
    # W2
    hidden_uni_W2 = len(units_infoW2)
    dist_W2 = np.empty((lat_states,hidden_uni_W2,temporal_length-1),dtype='float')

    for j in range(hidden_uni_W2):
        for i in range(temporal_length-1):
            dist_W2[:,j,i]=np.abs(W2[:,units_infoW2[j],i+1]-W2[:,units_infoW2[j],i])

    dist_stat_W2=np.empty((hidden_uni_W2,2),dtype='float')

    for i in range(dist_W2.shape[1]):
        dist_stat_W2[i,0]=dist_W2[:,i,:].mean()
        dist_stat_W2[i,1]=np.std(dist_W2[:,i,:])/np.sqrt(np.size(dist_W2[:,i,:]))

    # W1
    hidden_uni_W1 = len(units_infoW1)
    dist_W1 = np.empty((lat_states,hidden_uni_W1,temporal_length-1),dtype='float')

    for j in range(hidden_uni_W1):
        for i in range(temporal_length-1):
            dist_W1[:,j,i]=np.abs(W1[units_infoW1[j],:,i+1]-W1[units_infoW1[j],:,i])

    dist_stat_W1=np.empty((hidden_uni_W1,2),dtype='float')

    for i in range(dist_W1.shape[1]):
        dist_stat_W1[i,0]=dist_W1[:,i,:].mean()
        dist_stat_W1[i,1]=np.std(dist_W1[:,i,:])/np.sqrt(np.size(dist_W1[:,i,:]))

    distance_W = [dist_stat_W2[:,0].mean(),dist_stat_W1[:,0].mean()]
    
    return variance, var_info, Noinfo_hu, Info_hu, distance_W