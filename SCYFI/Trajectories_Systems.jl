using JLD2
using NPZ
using LinearAlgebra
using Pickle
using SCYFI
using Plots

# Loading FP
Dyn_obj=npzread("D:/_work_cestarellas/Analysis/PLRNN/SCYFI/Attractors/CE17/FP_CE17_00_28_NOINI.npz")
num_FP = size(Dyn_obj["FP"])[2]   # number of FP
num_Z = size(Dyn_obj["FP"])[1]
FP=Dyn_obj["FP"]

# Loading Empirical Data
EmpData = Pickle.npyload("D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17/L6/Test0/datasets/Training_data.npy")  

# Loading Parameters Model
Params=Pickle.npyload("D:/_work_cestarellas/Analysis/PLRNN/SCYFI/data/Model_Parameters/Model_Parameters_Gambling.pkl")
Trial_model=26
A = Params[1]
W₁ = Params[2][Trial_model,:,:]
W₂ = Params[3][Trial_model,:,:]
h₂ = Params[4]
h₁ = Params[5]


# Select initial condition
IC= EmpData[1][26,:]
Time_Steps=40000

clipped=true
Gene=get_latent_time_series(Time_Steps,A,W₁,W₂,h₁,h₂,num_Z,z_0=FP,is_clipped=clipped)
Gene_Emp=get_latent_time_series(Time_Steps,A,W₁,W₂,h₁,h₂,num_Z,z_0=IC,is_clipped=clipped)

Dist_fp=Array{Float32}(undef,Time_Steps)
Dist_emp=Array{Float32}(undef,Time_Steps)
for i in 1:Time_Steps
    Dist_fp[i]=sqrt(sum((Gene[i]-FP).^2))
    Dist_emp[i]=sqrt(sum((Gene_Emp[i]-FP).^2))
end

TS_FP=reduce(hcat,Gene)
TS_Emp=reduce(hcat,Gene_Emp)


plot(1:Time_Steps,[TS_Emp[1,:],TS_Emp[2,:],TS_Emp[3,:]],labels=["Z1" "Z2" "Z3"])
title!("Initial Condition from Data")
xlabel!("Bins")
ylabel!("Zscore")
savefig("D:/_work_cestarellas/Analysis/PLRNN/SCYFI/Attractors/CE17/Initial_Data.png")  

plot(1:Time_Steps,[TS_FP[1,:],TS_FP[2,:],TS_FP[3,:]],labels=["Z1" "Z2" "Z3"])
title!("Initial Condition from FP")
xlabel!("Bins")
ylabel!("Zscore")
savefig("D:/_work_cestarellas/Analysis/PLRNN/SCYFI/Attractors/CE17/Initial_FP.png")

plot(1:5000,Dist_fp[1:5000])
title!("Trajectory Distance to FP")
xlabel!("Bins")
ylabel!("Euclidean Distance")
savefig("D:/_work_cestarellas/Analysis/PLRNN/SCYFI/Attractors/CE17/Distance_first_Attractor_FP_Zoom.png")  

