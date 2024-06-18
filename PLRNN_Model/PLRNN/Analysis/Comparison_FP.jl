using JLD2
using NPZ
using LinearAlgebra
using Pickle
using Plots
using MultivariateStats
using StatsPlots
# Loading Dynamical objects
Dyn_obj₁=npzread("D:/_work_cestarellas/Analysis/PLRNN/SCYFI/Attractors/CE17/M0/FP_CE17_00_01.npz")
Dyn_obj₂=npzread("D:/_work_cestarellas/Analysis/PLRNN/SCYFI/Attractors/CE17/M0/FP_CE17_00_28.npz")

FP₁=Dyn_obj₁["FP"][:,1]
FP₂=Dyn_obj₂["FP"][:,1]
EV₁=Dyn_obj₁["EV"][:,1]
EV₂=Dyn_obj₂["EV"][:,1]

# Loading Parameters Model
Params=Pickle.npyload("D:/_work_cestarellas/Analysis/PLRNN/SCYFI/data/Model_Parameters/Model_Parameters_Gambling.pkl")

A = Params[1]
h₂ = Params[4]
h₁ = Params[5]
# Model Parameters for FP 1
Trial_model₁=27
W₁₁ = Params[2][Trial_model₁,:,:]
W₂₁ = Params[3][Trial_model₁,:,:]

# Model Parameters for FP 2
Trial_model₂=28
W₁₂ = Params[2][Trial_model₂,:,:]
W₂₂ = Params[3][Trial_model₂,:,:]

hidden_dim=size(h₂)[1]
order=1

# Select initial condition
zero_point=1
IC= EmpData[Trial_model][zero_point,:]
Time_Steps=1000
# get D matrices from
D1₁ = Array{Bool}(undef, hidden_dim, hidden_dim, order)
D2₁ = Array{Bool}(undef, hidden_dim, hidden_dim, order)
D1₁[:,:,1]=Diagonal((W₂₁*FP₁ + h₂).>0)                    
D2₁[:,:,1]=Diagonal((W₂₁*FP₁).>0)

D1₂ = Array{Bool}(undef, hidden_dim, hidden_dim, order)
D2₂ = Array{Bool}(undef, hidden_dim, hidden_dim, order)
D1₂[:,:,1]=Diagonal((W₂₂*FP₂ + h₂).>0)                    
D2₂[:,:,1]=Diagonal((W₂₂*FP₂).>0)

ΔD1=sum(abs.(D1₁-D1₂))
ΔD2=sum(abs.(D2₁-D2₂))

R1=real(EV₁)
R2=real(EV₂)

#Difference between FP
plot(FP₁-FP₂)
xlabel!("Z Dimensions")
ylabel!("ΔFP")

#Difference between EV
plot(R1-R2)
xlabel!("Z Dimensions")
ylabel!("ΔEV")

Eig₊ = [sum(R1.>0), sum(R2.>0)]./size(A)[1]
Eig₋ = [sum(R1.<0), sum(R2.<0)]./size(A)[1]

# In PyPlot backend, if we use chars like 'A':'L', ticks are displayed with "PyWrap".

groupedbar([Eig₊ Eig₋],
        bar_position = :stack,
        bar_width=0.2,
        xticks=(1:2),
        label=["Positive" "Negative"])
title!("Stability Fixed Points")
xlabel!("Found FP")
ylabel!("Ratio Dim Z")

