using JLD2
using NPZ
using LinearAlgebra
using Pickle
using SCYFI

include("D:/_work_cestarellas/Analysis/PLRNN/SCYFI/src/utilities/helpers.jl")
# Loading file from SCYFI output
data=Pickle.npyload("D:/_work_cestarellas/Analysis/PLRNN/SCYFI/data/Model_Parameters/Model_Parameters_Gambling.pkl")
Dyn_obj=npzread("D:/_work_cestarellas/Analysis/PLRNN/SCYFI/Attractors/CE17/FP_CE17_00_01.npz")

# Now we load the FP and search the FPs in the next trajectory_relu_matrix_list

num_FP=size(Dyn_obj["FP"])[2]
num_trials=size(data[2])[1]
T_ana = 1
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

FP_selected=1
FP=Dyn_obj["FP"][:,FP_selected]

trajectory_relu_matrix_list_1 = Array{Bool}(undef, hidden_dim, hidden_dim, order)
trajectory_relu_matrix_list_2 = Array{Bool}(undef, hidden_dim, hidden_dim, order)
trajectory_relu_matrix_list_1[:,:,1]=Diagonal((W₂*FP + h₂).>0)                    
trajectory_relu_matrix_list_2[:,:,1]=Diagonal((W₂*FP).>0)

Trial₊=28
println("Testing and Computing Trial: ",Trial₊)

W₁₊ = data[2][Trial₊,:,:]
W₂₊ = data[3][Trial₊,:,:]

# new candidate
res_2=get_cycle_point_candidate(A, W₁₊, W₂₊, h₁, h₂, trajectory_relu_matrix_list_1[:,:,1],
trajectory_relu_matrix_list_2[:,:,1], order)

#is this candidate real fp?
trajectory_relu_matrix_list_1_2 = Array{Bool}(undef, hidden_dim, hidden_dim, order)
trajectory_relu_matrix_list_2_2 = Array{Bool}(undef, hidden_dim, hidden_dim, order)
trajectory_relu_matrix_lists = Array{Bool}(undef, hidden_dim*2, hidden_dim, order)
trajectory_relu_matrix_list_1_2[:,:,1] =  Diagonal((W₂₊*res_2 + h₂).>0)                    
trajectory_relu_matrix_list_2_2[:,:,1] = Diagonal((W₂₊*res_2).>0)

# Matrix List with the info of the Quadrant
trajectory_relu_matrix_lists[:,:,1] = vcat(Diagonal((W₂₊*res_2 + h₂).>0) ,Diagonal((W₂*res_2).>0))
  
if trajectory_relu_matrix_list_1_2[:,:,1]==trajectory_relu_matrix_list_1[:,:,1] && trajectory_relu_matrix_list_2_2[:,:,1]==trajectory_relu_matrix_list_2[:,:,1]
    println("FP is same")
else
    println("FP is different")
    println("initialising in that quadrant")
    println("Number of initialisations in Pool: ",size(trajectory_relu_matrix_lists)[3])
    found_lower_orders = Array[]
    found_eigvals = Array[]
    cycles_found, eigvals = scy_fi(A, W₁₊, W₂₊, h₁, h₂, 1, Array[],trajectory_relu_matrix_lists,ClippedShallowPLRNN() , 
    outer_loop_iterations=1,inner_loop_iterations=5000,get_pool_from_traj=false,
    num_trajectories=1,
    len_trajectories=1)
    push!(found_lower_orders,cycles_found)
    push!(found_eigvals,eigvals)
    res=[found_lower_orders, found_eigvals]
    println(res)
    save("data/FP_CE17_00_28_NOINI.jld2","res",res)
end

