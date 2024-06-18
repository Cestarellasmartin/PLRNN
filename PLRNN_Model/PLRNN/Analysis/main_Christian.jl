using SCYFI
using LinearAlgebra
using Pickle
using JLD2

data=Pickle.npyload("data/Model_Parameters_Gambling.pkl")

res =find_cycles(data[1],data[2][1,:,:],data[3][1,:,:],data[5],data[4],1,outer_loop_iterations=500,inner_loop_iterations=1000,PLRNN=ClippedShallowPLRNN(), 
get_pool_from_traj=true,
num_trajectories=1000,
len_trajectories=1500)

println(res)
save("data/res_60.jld2","res",res)