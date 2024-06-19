import model_anafunctions as func
import multiprocessing as mp
import tqdm
import numpy as np
import torch as tc

def measurement_distance(mpath1,mpath2):
    A, W2, W1,h2, h1, C, m = func.W_matrix_generation(mpath1)
    num_neurons=W1.shape[0]
    num_trials = W2.shape[2]
    data_n,data_i = func.load_data(mpath2)
    neural_data = data_n
    data_n=[tc.from_numpy(i).float() for i in data_n]
    data_i=[tc.from_numpy(i).float() for i in data_i]

    # Simulation of long trials
    X=[]                                                                        # empty list to keep simulations for each trial
    ext_in=0                                                                    # number of extra bins withput inputs to add to the simulation
    if __name__ == '__main__':
        with mp.Pool() as pool:                                                              
            items=[(i,data_n[i],data_i[i],ext_in,m) for i in range(len(data_n))]    # list of the elements for the parallel computation of the function "ext_simul_par"
            for result in pool.starmap(func.Pext_simul, tqdm(items,desc='Simulatind trials in parallel')):
                X.append(result)
            
    # Rename simulated and experimental trials 
    strial = [X[i] for i in range(len(X))]
    etrial = [neural_data[i] for i in range(len(neural_data))]
    Dist = np.empty((num_trials,num_neurons),dtype=float)                                                          # empty list to keep simulations for each trial
    if __name__ == '__main__':
        with mp.Pool() as pool:
            alpha = 0.2
            items=[(strial[j][:,i],etrial[j][:,i],j,i,alpha) for i in range(strial[0].shape[1]) for j in range(len(strial))]    # list of the elements for the parallel computation of the function "ext_simul_par"
            for result in pool.starmap(func.DTW_para, tqdm(items,desc='Computing Distance')):
                Dist[result[0],result[1]] = result[2]
            
    distance_mean = Dist.mean(1).mean()
    distance_sem = Dist.mean(1).std()/np.sqrt(Dist.mean(1).shape[0])

    return distance_mean,distance_sem