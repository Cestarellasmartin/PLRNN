# Author: Cristian Estarellas
# Julia version: 1.8.5
# Date: 06/2024

# Loading packages
using JLD2
using NPZ
using Pickle
using LinearAlgebra
using Plots
using MultivariateStats
using StatsPlots
using Statistics

# Loading Packages SCYFI
include("src/utilities/helpers.jl")

##########################################################################################################################################
#################### Organization of Fixed Points and Cycles limits determined in the trials of the session ##############################
##########################################################################################################################################

# Loading FP strucutre data from Lukas Einsenmann SCYFI algorithm
folder_path="D:\\_work_cestarellas\\Analysis\\PLRNN\\SCYFI\\data\\CE17_red_run_03"
files=readdir(folder_path)

num_trials=length(files)
FP_trials=Dict()
EV_trials=Dict()
KC_trials=Dict()
Stab_trials=Dict()

# Organization of FP and LC data
for tr in 1:num_trials
    path_trial=joinpath(folder_path,files[tr])
    data = JLD2.load(path_trial)
    FPinfo=data["res"]
    kcy = size(FPinfo[1])[1]                                               # Number of k-cyckles 
    FP=Array[]
    EV=Array[]
    kC=[]
    Stab=[]
    # Dyn Obj
    for ik in 1:kcy
        DO=FPinfo[1][ik]
        EO=FPinfo[2][ik]
        num_ob = size(DO)[1]                                               # Number of objects determined for cycle ik
        #If you have dynamical objects:
        if num_ob>0
            # K-cycle = 1
            if ik==1
                for iob in 1:num_ob
                    push!(FP,DO[iob][1])
                    push!(EV,EO[iob])
                    push!(kC,ik)
                    stability=abs.(EO[iob])
                    unstable = sum(stability.>=1)
                    stable = sum(stability .< 1)
                    if unstable==length(stability)
                        push!(Stab,1)
                    elseif stable==length(stability)
                        push!(Stab,-1)
                    else
                        push!(Stab,0)
                    end
                end
            else
            # K-cycle > 1
                for iob in 1:num_ob
                    flag=maximum(diff(DO[iob])).>0.00001
                    if sum(flag)>0
                        push!(FP,DO[iob])
                        push!(EV,EO[iob])
                        push!(kC,ik)
                        stability=abs.(EO[iob])
                        unstable = sum(stability.>=1)
                        stable = sum(stability .< 1)
                        if unstable==length(stability)
                            push!(Stab,1)
                        elseif stable==length(stability)
                            push!(Stab,-1)
                        else
                            push!(Stab,0)
                        end
                    else
                        push!(FP,DO[iob][1])
                        push!(EV,EO[iob])
                        push!(kC,1)
                        stability=abs.(EO[iob])
                        unstable = sum(stability.>=1)
                        stable = sum(stability .< 1)
                        if unstable==length(stability)
                            push!(Stab,1)
                        elseif stable==length(stability)
                            push!(Stab,-1)
                        else
                            push!(Stab,0)
                        end
                    end
                end
            end
        end
    end

    FP_trials[string(tr)]=FP
    EV_trials[string(tr)]=EV
    KC_trials[string(tr)]=kC
    Stab_trials[string(tr)]=Stab
end

# Plots of Generalization data structured
FP_num = fill(NaN,(num_trials,))
KCY_num = fill(NaN,(num_trials,))
ST_num = fill(NaN,(num_trials,))
UNST_num = fill(NaN,(num_trials,))
BIST_num = fill(NaN,(num_trials,))

for i in 1:num_trials
    if isempty(KC_trials[string(i)])
        FP_num[i] = 0
        KCY_num[i] = 0
        ST_num[i] = 0
        UNST_num[i] = 0
        BIST_num[i] = 0
    else
        FP_num[i] = sum(KC_trials[string(i)].==1)
        KCY_num[i] = sum(KC_trials[string(i)].>1)
        ST_num[i] = sum(Stab_trials[string(i)].==-1)
        UNST_num[i] = sum(Stab_trials[string(i)].==1)
        BIST_num[i] = sum(Stab_trials[string(i)].==0)
    end
end

limit_do=maximum(FP_num.+KCY_num)
groupedbar([FP_num KCY_num],
bar_position= :stack,
bar_width=0.5,
xlabel="Trials",ylabel="# of Different Objects",
title="FP and Cycles detected in the session",
label=["FP" "K-cycles"],
yticks=(1:limit_do),
legend=:outertopright)


groupedbar([ST_num UNST_num BIST_num],
bar_position= :stack,
bar_width=0.5,
color = [:red :black :limegreen],
xlabel="Trials",ylabel="# of Different Objects",
title="FP and Cycles detected in the session",
label=["Stable" "Unstable" "Saddle"],
yticks=(1:limit_do),
legend=:outertopright)

FP_num = ST_num+UNST_num+BIST_num

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

# Select Trial
TS=20

# Loading Model
data=Pickle.npyload("D:/_work_cestarellas/Analysis/PLRNN/SCYFI/data/Model_Parameters/Model_Parameters_CE17_red_003.pkl")
# Defining Parameters of the Trial Studied
A = data[1]
W₁ = data[2][TS,:,:]
W₂ = data[3][TS,:,:]
h₂ = data[4]
h₁ = data[5]
C = data[6]
hidden_dim=size(h₂)[1]

# Working with FPs
korder=1
pos_fp=findall(x->x==1,KC_trials[string(TS)])
num_fp = length(pos_fp)

# Quadrant Distance between Dynamical objects
Δ₁ = fill(NaN,(num_fp,num_fp))
Δ₂ = fill(NaN,(num_fp,num_fp))

# Quadrant FP1
for i0 in 1:num_fp
    FP=FP_trials[string(TS)][i0]
    # get D matrices from
    D₁ = Array{Bool}(undef, hidden_dim, hidden_dim, korder)
    D₂ = Array{Bool}(undef, hidden_dim, hidden_dim, korder)
    D₁[:,:,1]=Diagonal((W₂*FP + h₂).>0)                    
    D₂[:,:,1]=Diagonal((W₂*FP).>0)
    # Quadrant FP2
    
    for i1 in 1:num_fp
        FP2=FP_trials[string(TS)][i1]
        # get D matrices from
        DF₁ = Array{Bool}(undef, hidden_dim, hidden_dim, korder)
        DF₂ = Array{Bool}(undef, hidden_dim, hidden_dim, korder)
        DF₁[:,:,korder]=Diagonal((W₂*FP2 + h₂).>0)                    
        DF₂[:,:,korder]=Diagonal((W₂*FP2).>0)
        Δ₁[i0,i1] = sum(abs.(D₁-DF₁))
        Δ₂[i0,i1] = sum(abs.(D₂-DF₂))
    end
end

heatmap(Δ₁, color=:viridis, aspect_ratio=:equal, 
xlabel="FPs", ylabel="FP", title="W₂*FP+h₂",
colorbar_title="Distance")
xlims!(0.5, num_fp+0.5)

heatmap(Δ₂, color=:viridis, aspect_ratio=:equal, 
xlabel="FPs", ylabel="FP", title="W₂*FP",
colorbar_title="Distance")
xlims!(0.5, num_fp+0.5)

# Sorted matrix
sort_fp=sortperm(Δ₂[:,1])
cust_xtick=[string(x) for x in sort_fp]
heatmap(Δ₂[sort_fp,sort_fp], color=:viridis, aspect_ratio=:equal, 
xlabel="FPs", ylabel="FP", title="W₂*FP",
colorbar_title="Distance")
xticks!(1:17,cust_xtick)
yticks!(1:17,cust_xtick)
xlims!(0.5, num_fp+0.5)


sort_fp=sortperm(Δ₁[:,1])
cust_xtick=[string(x) for x in sort_fp]
heatmap(Δ₁[sort_fp,sort_fp], color=:viridis, aspect_ratio=:equal, 
xlabel="FPs", ylabel="FP", title="W₂*FP+h₂",
colorbar_title="Distance")
xticks!(1:17,cust_xtick)
yticks!(1:17,cust_xtick)
xlims!(0.5, num_fp+0.5)


##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
# PCA dimensional reduction

pca_model = fit(PCA, hcat(FP_trials[string(TS)]...)'; maxoutdim=2)
#Trial FP groups
### Change the FP (these groups are different for each trial tested)
group1=[1,2,5]
group2=[7]
group3=[3,4,6]
#Scatter plot of the FP distribution
Plots.scatter(projection(pca_model)[group1,1], projection(pca_model)[group1,2],labels="FP $group1",legend= :outerbottomright,
tickfontsize=12, legendfontsize=12)
Plots.scatter!(projection(pca_model)[group2,1], projection(pca_model)[group2,2],labels="FP $group2")
Plots.scatter!(projection(pca_model)[group3,1], projection(pca_model)[group3,2],labels="FP $group3")
Plots.xlabel!("PC1")
Plots.ylabel!("PC2")
Plots.title!("FPs reduced dimensions")


##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################


# Real data
EmpData = Pickle.npyload("D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17_reduction/datasets/Training_data.npy") 
EmpInput = Pickle.npyload("D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17_reduction/datasets/Training_inputs.npy")  

matrix_FP=hcat(FP_trials[string(TS)]...)'
n1=[8,14,7,13,3,2]
n2=[9,10,10,8,9,1]
#n1=[1,2,5]
#n2=[7,3,4]
pp=Vector{Plots.Plot}()
for i in 1:6
    ineu1=n1[i]#rand(1:14)
    ineu2=n2[i]#rand(1:14)
    p1=Plots.plot(
        Plots.scatter(matrix_FP[group1,ineu1],matrix_FP[group1,ineu2],tickfontsize=10,legend=false,
            xlabel="Neuron $ineu1",ylabel="Neuron $ineu2",title="Trial 20"),
    )
    Plots.scatter!(matrix_FP[group2,ineu1],matrix_FP[group2,ineu2],tickfontsize=12,legend=false)
    Plots.scatter!(matrix_FP[group3,ineu1],matrix_FP[group3,ineu2],tickfontsize=12,legend=false)
    Plots.plot!(EmpData[TS][:,ineu1],EmpData[TS][:,ineu2],tickfontsize=12,legend=false)
    
    push!(pp, p1)
end

sub=Plots.plot(pp[1],pp[2],pp[3],pp[4],pp[5],pp[6],layout=(2,3),size=(1024,720))
