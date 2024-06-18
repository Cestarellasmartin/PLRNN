
import random
import numpy as np
import torch as tc
from tqdm import tqdm
from evaluation import klx_gmm as kl

# FUNCTION Concatonate trials
def concatenate_list(lis,ax):
    res = lis[0]    # first element of the list
    Ls = [lis[0].shape[ax]] #first length of the list
    for i in range(1, len(lis)):
        Ls.append(lis[i].shape[ax])
        res = np.concatenate((res, lis[i]), axis=ax)
    return res, Ls


def SnapShot_generation(Training_Activity,Length_trial,Input_channels,modeltrials,m):
    Warm_time=50000
    Length_data=Length_trial+1+Warm_time
    Inputs=tc.zeros(Length_data,Input_channels,dtype=tc.float32)

    # Generation of free trajectories for limiting behaviour - SNAPSHOT
    print("::snapshot::")
    SS_Model=[]
    for w_index in tqdm(range(modeltrials)):
        data_trial=tc.from_numpy(Training_Activity[w_index]).float()          # tensor of neuronal data for initial trial data
        X, _ = m.generate_free_trajectory(data_trial,Inputs,Length_data,w_index)
        SS_Model.append(X[-Length_trial:,:])

    return SS_Model

def KL_LimitBehaviour(Data,Total_Neurons,modeltrials,subspace_neurons,Itera_KL):
    B_Model=Data[0]
    A_Model=Data[1]
    I_Model=Data[2]
    B_Signal=Data[3]
    A_Signal=Data[4]
    I_Signal=Data[5]
    list_NeuronID = [i for i in range(Total_Neurons)]
    Dim_kl = int(np.floor(Total_Neurons/subspace_neurons))
    KL_div = np.ones((modeltrials,3,Itera_KL))*np.nan
    for k in range(Itera_KL):
        print("Iteration",k+1,"of",Itera_KL)
        for i in tqdm(range(modeltrials)):
                random.shuffle(list_NeuronID)
                dim0 = 0
                dim3 = subspace_neurons
                kl_dim=np.ones((Dim_kl,3))*np.nan
                for j in range(Dim_kl):
                        #Before Session
                        kl_dim[j,0] = kl.calc_kl_from_data(B_Model[i][:,list_NeuronID[dim0:dim3]],tc.tensor(B_Signal[:,list_NeuronID[dim0:dim3]]).float())
                        #ITI
                        kl_dim[j,1] = kl.calc_kl_from_data(I_Model[i][:,list_NeuronID[dim0:dim3]],tc.tensor(I_Signal[:,list_NeuronID[dim0:dim3]]).float())
                        #After Session
                        kl_dim[j,2] = kl.calc_kl_from_data(A_Model[i][:,list_NeuronID[dim0:dim3]],tc.tensor(A_Signal[:,list_NeuronID[dim0:dim3]]).float())
                        dim0+=subspace_neurons
                        dim3+=subspace_neurons
                KL_div[i,:,k]=kl_dim.mean(0)
    return KL_div

def NonStationary_generation(Training_Activity,Length_trial,Input_channels,modeltrials,m):    
    Inputs=tc.zeros(Length_trial,Input_channels,dtype=tc.float32)
    data_trial=tc.from_numpy(Training_Activity[0][-1,:]).float()  
    num_neurons=Training_Activity[0].shape[1]
    NS_ModelT=[]
    for w_index in tqdm(range(modeltrials)):
        data_trial = tc.reshape(data_trial,(1,num_neurons))
        X, _ = m.generate_free_trajectory(data_trial,Inputs,Length_trial,w_index)
        data_trial=X[-1,:]
        NS_ModelT.append(X)
    # Concatenate Trials
    NS_Signal,_=concatenate_list(NS_ModelT,0)
    return NS_Signal    

# Generation Training Data
def Train_generation(Training_Activity,Training_Input,modeltrials,m):
    ModelS=[]
    for w_index in tqdm(range(modeltrials)):
        data_trial=tc.from_numpy(Training_Activity[w_index]).float()          # tensor of neuronal data for initial trial data
        input_trial = tc.from_numpy(Training_Input[w_index]).float()
        length_sim = input_trial.shape[0]
        X, _ = m.generate_free_trajectory(data_trial,input_trial,length_sim,w_index)
        ModelS.append(X[:,:])
    Model_Signal,_=concatenate_list(ModelS,0)
    return Model_Signal,ModelS

# Generation Testing Data
 # W Testing parameters
def Test_generation(Metadata,Test_Activity,Test_Input,m):
    print("::Generating W testing parameters::")
    print('Set of test trials: ',Metadata["TestTrials"])
    # Organisign test trial location in the Training Trials
    t_prev = [i for i in Metadata["Training2Test"]]
    t_post = [i+1 for i in Metadata["Training2Test"]]
    # Ouput Test Trial position
    print('trials before test trial: ',t_prev)
    print('trials after test trial: ',t_post)
    # Determine W matrices from the model
    _, W1t, W2t, _, _, _ = m.get_latent_parameters()
    # Transform tensor to numpy format
    W2 = W2t.detach().numpy().transpose(1,2,0)
    W1 = W1t.detach().numpy().transpose(1,2,0)
    # Computing W matrices for test trials
    W2_test = np.empty((W2.shape[0],W2.shape[1],len(Metadata["TestTrials"])))
    W1_test = np.empty((W1.shape[0],W1.shape[1],len(Metadata["TestTrials"])))
    for i in range(len(t_prev)):
            W2_test[:,:,i] = (W2[:,:,t_prev[i]]+W2[:,:,t_post[i]])/2.0
            W1_test[:,:,i] = (W1[:,:,t_prev[i]]+W1[:,:,t_post[i]])/2.0

    #Generate Latent states for Test Trials (TT)
    ModelS = []
    #Generate Latent states for Test Trials
    W1_ind = [tc.from_numpy(W1_test[:,:,i]).float() for i in range(len(t_prev))]
    W2_ind = [tc.from_numpy(W2_test[:,:,i]).float() for i in range(len(t_prev))]
    for i in tqdm(range(len(W1_ind))):
            data_test=tc.from_numpy(Test_Activity[i]).float()
            input_test=tc.from_numpy(Test_Input[i]).float()
            T0=int(len(Test_Activity[i]))
            X, _ = m.generate_test_trajectory(data_test[0:11,:],W2_ind[i],W1_ind[i],input_test, T0,i)
            ModelS.append(X)
    Model_Signal,_=concatenate_list(ModelS,0)
    return Model_Signal,ModelS

def TimeSerie_KL_distance(Data,Total_Neurons,subspace_neurons,Itera_KL):
    # Simulated Data
    B_Model = Data[0]
    A_Model = Data[1]
    Train_Model=Data[2]
    Test_Model = Data[3]
    # Real Data
    B_Signal = Data[4]
    A_Signal = Data[5]
    Train_Signal = Data[6]
    Test_Signal = Data[7]

    list_NeuronID = [i for i in range(Total_Neurons)]
    Dim_kl = int(np.floor(Total_Neurons/subspace_neurons))
    KL_div = np.ones((Itera_KL,int(len(Data)/2)))*np.nan
    for k in tqdm(range(Itera_KL),"Combinations:"):
        random.shuffle(list_NeuronID)
        dim0 = 0
        dim3 = subspace_neurons
        kl_dim=np.ones((Dim_kl,int(len(Data)/2)))*np.nan
        for j in range(Dim_kl):
                #Before Session
                kl_dim[j,0] = kl.calc_kl_from_data(tc.tensor(B_Model[:,list_NeuronID[dim0:dim3]]).float(),tc.tensor(B_Signal[:,list_NeuronID[dim0:dim3]]).float())
                #After Session
                kl_dim[j,1] = kl.calc_kl_from_data(tc.tensor(A_Model[:,list_NeuronID[dim0:dim3]]).float(),tc.tensor(A_Signal[:,list_NeuronID[dim0:dim3]]).float())
                #Train Session
                kl_dim[j,2] = kl.calc_kl_from_data(tc.tensor(Train_Model[:,list_NeuronID[dim0:dim3]]).float(),tc.tensor(Train_Signal[:,list_NeuronID[dim0:dim3]]).float())
                #Test Session
                kl_dim[j,3] = kl.calc_kl_from_data(tc.tensor(Test_Model[:,list_NeuronID[dim0:dim3]]).float(),tc.tensor(Test_Signal[:,list_NeuronID[dim0:dim3]]).float())
                dim0+=subspace_neurons
                dim3+=subspace_neurons
        KL_div[k,:]=kl_dim.mean(0)
    return KL_div
