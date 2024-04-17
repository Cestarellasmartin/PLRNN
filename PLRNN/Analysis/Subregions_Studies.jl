using JLD2
using NPZ
using LinearAlgebra
using Pickle
using Plots
using SCYFI
using Statistics

# Loading FP
Dyn_obj=npzread("D:/_work_cestarellas/Analysis/PLRNN/SCYFI/Attractors/CE17/M0/FP_CE17_00_01.npz")
num_FP = size(Dyn_obj["FP"])[2]   # number of FP
num_Z = size(Dyn_obj["FP"])[1]
FP=Dyn_obj["FP"][:,1]

# Loading Empirical Data
EmpData = Pickle.npyload("D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17/L6/Test0/datasets/Training_data.npy")  
InpData = Pickle.npyload("D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17/L6/Test0/datasets/Training_inputs.npy")
# Loading Parameters Model
Params=Pickle.npyload("D:/_work_cestarellas/Analysis/PLRNN/SCYFI/data/Model_Parameters/Model_Parameters_Gambling.pkl")


# Distance within the session
Trial_model=14
A = Params[1]
W₁ = Params[2][Trial_model,:,:]
W₂ = Params[3][Trial_model,:,:]
h₂ = Params[4]
h₁ = Params[5]

hidden_dim=size(h₂)[1]
order=1

# Select initial condition
zero_point=1
IC= EmpData[Trial_model][zero_point,:]
Time_Steps=1000
# get D matrices from
D1_data = Array{Bool}(undef, hidden_dim, hidden_dim, order)
D2_data = Array{Bool}(undef, hidden_dim, hidden_dim, order)
D1_data[:,:,1]=Diagonal((W₂*FP + h₂).>0)                    
D2_data[:,:,1]=Diagonal((W₂*FP).>0)

Distance_D1=Array{Int32}(undef,size(EmpData[Trial_model])[1]-zero_point+1)
Distance_D2=Array{Int32}(undef,size(EmpData[Trial_model])[1]-zero_point+1)
for i in zero_point:size(EmpData[Trial_model])[1]    
    IC= EmpData[Trial_model][i,:]
    Time_Steps=1000
    # get D matrices from
    D1_data₊ = Array{Bool}(undef, hidden_dim, hidden_dim, order)
    D2_data₊ = Array{Bool}(undef, hidden_dim, hidden_dim, order)
    D1_data₊[:,:,1]=Diagonal((W₂*IC + h₂).>0)                    
    D2_data₊[:,:,1]=Diagonal((W₂*IC).>0)
    
    Distance_D1[i-zero_point+1]=sum(abs.(D1_data-D1_data₊))
    Distance_D2[i-zero_point+1]=sum(abs.(D2_data-D2_data₊))

    if D1_data==D1_data₊ && D2_data==D2_data₊
        println("Same Subregion")
    end
end


plot(zero_point:size(EmpData[Trial_model])[1],Distance_D1,labels="D1")
plot!(zero_point:size(EmpData[Trial_model])[1],Distance_D2,labels="D2")
xlabel!("Time Steps")
ylabel!("Distance")
title!("Subregions distances from Subregion FP")


D1_mean=Distance_D1.-mean(Distance_D1)
D2_mean=Distance_D2.-mean(Distance_D2)

max_D1=maximum(D1_mean)
min_D1=minimum(D1_mean)

plot(zero_point:size(EmpData[Trial_model])[1],D1_mean,labels="D1ₘ")
# plot!(zero_point:size(EmpData[Trial_model])[1],D2_mean,labels="D2ₘ")
plot!(zero_point:size(EmpData[Trial_model])[1],InpData[Trial_model][:,1].*max_D1.+min_D1,labels="Cue")
plot!(zero_point:size(EmpData[Trial_model])[1],InpData[Trial_model][:,2]./4 .*max_D1.+min_D1,labels="GR")
plot!(zero_point:size(EmpData[Trial_model])[1],InpData[Trial_model][:,3].*max_D1.+min_D1,labels="SR")
xlabel!("Time Steps")
ylabel!("Distance")
title!("Relation between Distance and External Inputs")


####################################
# Testing Periodicity across session
####################################

# Subregion Initial Point
# Distance within the session
Trial_ini=1
A = Params[1]
W₁ = Params[2][Trial_ini,:,:]
W₂ = Params[3][Trial_ini,:,:]
h₂ = Params[4]
h₁ = Params[5]

hidden_dim=size(h₂)[1]
order=1

# Select initial condition

IC= EmpData[Trial_ini][1,:]
# get D matrices from
D1_ini = Array{Bool}(undef, hidden_dim, hidden_dim, order)
D2_ini = Array{Bool}(undef, hidden_dim, hidden_dim, order)
D1_ini[:,:,1]=Diagonal((W₂*FP + h₂).>0)                    
D2_ini[:,:,1]=Diagonal((W₂*FP).>0)


# Computing the rest of distances for the consecutive trials
Trial_end=40
Total_Steps=size(reduce(vcat,EmpData[Trial_ini:Trial_end]))[1]


Distance_D1=Array{Int32}(undef,Total_Steps)
Distance_D2=Array{Int32}(undef,Total_Steps)
global i_pos=0
for iw in Trial_ini:Trial_end
    W₂₊ = Params[3][iw,:,:]
    for its in 1:size(EmpData[iw])[1]    
        DP= EmpData[iw][its,:]
        # get D matrices 
        D1_data₊ = Array{Bool}(undef, hidden_dim, hidden_dim, order)
        D2_data₊ = Array{Bool}(undef, hidden_dim, hidden_dim, order)
        D1_data₊[:,:,1]=Diagonal((W₂₊*DP + h₂).>0)                    
        D2_data₊[:,:,1]=Diagonal((W₂₊*DP).>0)
        i_pos += 1
        Distance_D1[i_pos]=sum(abs.(D1_ini-D1_data₊))
        Distance_D2[i_pos]=sum(abs.(D2_ini-D2_data₊))

        if D1_ini==D1_data₊ && D2_ini==D2_data₊
            println("Same Subregion")
        end
    end
end


plot(1:Total_Steps,Distance_D1,labels="D1")
plot!(1:Total_Steps,Distance_D2,labels="D2")
xlabel!("Time Steps")
ylabel!("Distance")
title!("Subregions distance from Subregion")


# Specific Trial in a long trajectory
Ext_I=reduce(vcat,InpData[Trial_ini:Trial_end])

Trials_S0=Array{Int32}(undef,Trial_end-Trial_ini)
Trials_S1=Array{Int32}(undef,Trial_end-Trial_ini)

Trials_S0[1]=1
Trials_S1[1]=size(EmpData[Trial_ini])[1]

for i in (Trial_ini+1):(Trial_end-1)
    Trials_S0[i-Trial_ini+1]=Trials_S1[i-Trial_ini]
    Trials_S1[i-Trial_ini+1]=Trials_S1[i-Trial_ini]+size(EmpData[i])[1]
end

Trial_obs=25
Trial_aim=Trial_obs-Trial_ini+1
trials_extra=5
plot(Trials_S0[Trial_aim]:Trials_S1[Trial_aim+trials_extra],Distance_D1[Trials_S0[Trial_aim]:Trials_S1[Trial_aim+trials_extra]],labels="D1")
# plot!(Trials_S0[Trial_aim]:Trials_S1[Trial_aim+trials_extra],Distance_D2[Trials_S0[Trial_aim]:Trials_S1[Trial_aim+trials_extra]],labels="D2")
plot!(Trials_S0[Trial_aim]:Trials_S1[Trial_aim+trials_extra],Ext_I[Trials_S0[Trial_aim]:Trials_S1[Trial_aim+trials_extra],1].*25 .+100,labels="Cue")
plot!(Trials_S0[Trial_aim]:Trials_S1[Trial_aim+trials_extra],Ext_I[Trials_S0[Trial_aim]:Trials_S1[Trial_aim+trials_extra],2].*6 .+100,labels="GR")
plot!(Trials_S0[Trial_aim]:Trials_S1[Trial_aim+trials_extra],Ext_I[Trials_S0[Trial_aim]:Trials_S1[Trial_aim+trials_extra],3].*25 .+100,labels="GR")
xlabel!("Time Steps")
ylabel!("Distance")
title!("Subregions distance from Subregion")




###################
# Testing unstability fixed point
##################
Trial_test=1
A = Params[1]
W₁ = Params[2][Trial_test,:,:]
W₂ = Params[3][Trial_test,:,:]
h₂ = Params[4]
h₁ = Params[5]

hidden_dim=size(h₂)[1]
order=1

noise_level=10
FPₙ = FP+randn(size(FP)).*noise_level
Time_Steps=20000000
IC= EmpData[1][zero_point,:]
clipped=true
Gene=get_latent_time_series(Time_Steps,A,W₁,W₂,h₁,h₂,num_Z,z_0=IC,is_clipped=clipped)


# Select initial condition
zero_point=1
# get D matrices from
D1_data = Array{Bool}(undef, hidden_dim, hidden_dim, order)
D2_data = Array{Bool}(undef, hidden_dim, hidden_dim, order)
D1_data[:,:,1]=Diagonal((W₂*FP + h₂).>0)                    
D2_data[:,:,1]=Diagonal((W₂*FP).>0)

Distance_D1=Array{Int32}(undef,size(Gene)[1])
Distance_D2=Array{Int32}(undef,size(Gene)[1])
for i in zero_point:size(Gene)[1]    
    IC= Gene[i]
    # get D matrices from
    D1_data₊ = Array{Bool}(undef, hidden_dim, hidden_dim, order)
    D2_data₊ = Array{Bool}(undef, hidden_dim, hidden_dim, order)
    D1_data₊[:,:,1]=Diagonal((W₂*IC + h₂).>0)                    
    D2_data₊[:,:,1]=Diagonal((W₂*IC).>0)
    
    Distance_D1[i-zero_point+1]=sum(abs.(D1_data-D1_data₊))
    Distance_D2[i-zero_point+1]=sum(abs.(D2_data-D2_data₊))

    if D1_data==D1_data₊ && D2_data==D2_data₊
        println("Same Subregion",i)
    end
end

plot(1:size(Gene)[1],Distance_D1,labels="D1")
plot!(1:size(Gene)[1],Distance_D2,labels="D2")
xlabel!("Time Steps")
ylabel!("Distance")
title!("Subregions distances from Subregion FP")

Gene_series=reduce(hcat,Gene)
plot(Gene_series[1,:])

# Subregions in constant distances
it_const=20000
D1_cons = Array{Bool}(undef, hidden_dim, hidden_dim, order)
D2_cons = Array{Bool}(undef, hidden_dim, hidden_dim, order)
D1_cons[:,:,1]=Diagonal((W₂*Gene[it_const] + h₂).>0)                    
D2_cons[:,:,1]=Diagonal((W₂*Gene[it_const]).>0)

Distance_D1=Array{Int32}(undef,size(Gene)[1])
Distance_D2=Array{Int32}(undef,size(Gene)[1])
for i in it_const:size(Gene)[1]    
    IC= Gene[i]
    # get D matrices from
    D1_data₊ = Array{Bool}(undef, hidden_dim, hidden_dim, order)
    D2_data₊ = Array{Bool}(undef, hidden_dim, hidden_dim, order)
    D1_data₊[:,:,1]=Diagonal((W₂*IC + h₂).>0)                    
    D2_data₊[:,:,1]=Diagonal((W₂*IC).>0)
    
    Distance_D1[i-zero_point+1]=sum(abs.(D1_cons-D1_data₊))
    Distance_D2[i-zero_point+1]=sum(abs.(D2_cons-D2_data₊))

    if D1_cons==D1_data₊ && D2_cons==D2_data₊
        # println("Same Subregion",i)
    else
        println(i)
    end
end


# Determining a possible FP
# Matrix List with the info of the Quadrant
trajectory_relu_matrix_lists = Array{Bool}(undef, hidden_dim*2, hidden_dim, order)
trajectory_relu_matrix_lists[:,:,1] = vcat(Diagonal((W₂*Gene[it_const] + h₂).>0) ,Diagonal((W₂*Gene[it_const]).>0))
println("FP is different")
println("initialising in that quadrant")
println("Number of initialisations in Pool: ",size(trajectory_relu_matrix_lists)[3])
found_lower_orders = Array[]
found_eigvals = Array[]
cycles_found, eigvals = scy_fi(A, W₁, W₂, h₁, h₂, 1, Array[],trajectory_relu_matrix_lists,ClippedShallowPLRNN() , 
outer_loop_iterations=1,inner_loop_iterations=5000,get_pool_from_traj=false,
num_trajectories=1,
len_trajectories=1)
push!(found_lower_orders,cycles_found)
push!(found_eigvals,eigvals)
res=[found_lower_orders, found_eigvals]
println(res)

