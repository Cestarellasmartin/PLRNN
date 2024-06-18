using JLD2
using NPZ
using LinearAlgebra
using Pickle
using Plots
using SCYFI

data=Pickle.npyload("D:/_work_cestarellas/Analysis/PLRNN/SCYFI/data/Model_Parameters/Model_Parameters_Gambling.pkl")
file_name=readdir("D:\\_work_cestarellas\\Analysis\\PLRNN\\SCYFI\\Attractors\\CE17\\M0")
num_files=size(file_name)[1]
num_trials=size(data[2])[1]
FP_session=zeros((num_trials,num_files))
trial_stud=String[]
for it in 1:num_files
    ctrials=string(file_name[it][12],file_name[it][13])
    push!(trial_stud,ctrials)
    T_ana=parse(Int64,ctrials)
    Dyn_obj=npzread(string("D:/_work_cestarellas/Analysis/PLRNN/SCYFI/Attractors/CE17/M0/FP_CE17_00_",ctrials,".npz"))
    
    # Now we load the FP and search the FPs in the next trajectory_relu_matrix_list
    num_FP=size(Dyn_obj["FP"])[2]
    num_trials=size(data[2])[1]

    println("Number of Fixed Points: ",num_FP)
    println("Number of Trials: ", num_trials)
    println("Parameters from trial: ", T_ana)

    # Defining Parameters of the Trial Studied
    A = data[1]
    W₁ = data[2][T_ana,:,:]
    W₂ = data[3][T_ana,:,:]
    h₂ = data[4]
    h₁ = data[5]
    hidden_dim=size(h₂)[1]
    order=1

    # Testing Presence of the FP in the rest of the Trials
    Same_FP=zeros((num_trials,num_FP))  
    for j in 1:num_FP
        # Select FP
        FP=Dyn_obj["FP"][:,j]
        # get D matrices from
        trajectory_relu_matrix_list_1 = Array{Bool}(undef, hidden_dim, hidden_dim, order)
        trajectory_relu_matrix_list_2 = Array{Bool}(undef, hidden_dim, hidden_dim, order)
        trajectory_relu_matrix_list_1[:,:,1]=Diagonal((W₂*FP + h₂).>0)                    
        trajectory_relu_matrix_list_2[:,:,1]=Diagonal((W₂*FP).>0)

        #calculate FP in that subregion for next trial
        for i in 1:num_trials
            global Same_FP
            W₁₊ = data[2][i,:,:]
            W₂₊ = data[3][i,:,:]

            res_2=get_cycle_point_candidate(A, W₁₊, W₂₊, h₁, h₂, trajectory_relu_matrix_list_1[:,:,1],
            trajectory_relu_matrix_list_2[:,:,1], order)

            #is this candidate real fp?

            trajectory_relu_matrix_list_1_2 = Array{Bool}(undef, hidden_dim, hidden_dim, order)

            trajectory_relu_matrix_list_2_2 = Array{Bool}(undef, hidden_dim, hidden_dim, order)

            trajectory_relu_matrix_list_1_2[:,:,1] = Diagonal((W₂₊*res_2 + h₂).>0)                    

            trajectory_relu_matrix_list_2_2[:,:,1] = Diagonal((W₂₊*res_2).>0)


            if trajectory_relu_matrix_list_1_2[:,:,1]==trajectory_relu_matrix_list_1[:,:,1] && trajectory_relu_matrix_list_2_2[:,:,1]==trajectory_relu_matrix_list_2[:,:,1]
                Same_FP[i,j]=1
            end
        end
    end

    for j in 1:num_FP
        Presence_FP=sum(Same_FP[:,j])/num_trials*100
        println("Percentage of FP ", j," in the session: ",Presence_FP," %")
        Change_Trial=findlast(x->x==1,Same_FP[:,j])[1]
        println("Last trial where appears: ", Change_Trial)
    end
   FP_session[:,it]=Same_FP[:,1]     
end

pt1=plot()
for it in 1:num_files
    pt1 = plot!(1:num_trials,FP_session[:,it],labels=string("FP Trial ",trial_stud[it]))
end
pt1 = xlabel!("Trials")
pt1 = title!("Fixed Points of Session CE17")
display(pt1)