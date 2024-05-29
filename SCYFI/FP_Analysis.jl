using JLD2
using NPZ
using Pickle
using LinearAlgebra
using Plots
using MultivariateStats
using StatsPlots
using Statistics

include("D:/_work_cestarellas/Analysis/PLRNN/SCYFI/src/utilities/helpers.jl")

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
# Organization of Fixed Points and Cycles limits determined in the trials of the session

folder_path="D:\\_work_cestarellas\\Analysis\\PLRNN\\SCYFI\\data\\CE17_red_000"
files=readdir(folder_path)
num_trials=length(files)
FP_trials=Dict()
EV_trials=Dict()
KC_trials=Dict()
Stab_trials=Dict()

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

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

# Select Trial
TS=20

# Loading Model
data=Pickle.npyload("D:/_work_cestarellas/Analysis/PLRNN/SCYFI/data/Model_Parameters/Model_Parameters_CE17_red_001.pkl")
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
# Working with K-cycles

# Work in progress...


##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

# Classification of FPs
# Select Trial
it=9
re =real(EV_trials[string(TS)][it])
ir =imag(EV_trials[string(TS)][it])
re_pos=findall(x->x==0,ir)
ir_pos=findall(x->x!=0,ir)

# Real eigenvalues
uns_node=sum(re[re_pos].>0)
circ=sum(re[re_pos].==0)
s_node=sum(re[re_pos].<0)

# Pair conjugate
pair = unique(abs.(ir[ir_pos]))
rest_unique=mod(length(ir_pos),length(pair))
if rest_unique==0
    num_pair=length(pair)
    num_pairs=sum(unique(re[ir_pos]).<0)
    num_pairu=sum(unique(re[ir_pos]).>0)
    num_pairc=sum(unique(re[ir_pos]).==0)
else
    num_pair=length(pair)-rest_unique
end


labels=["Stable node", "Unstable node", "Stable spiral","Unstable spiral"]
sizes =[s_node,uns_node,num_pairs*2,num_pairu*2]
sizes =sizes/sum(sizes)
pie(labels, sizes, title="FP $it", legend=false)

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
# PCA dimensional reduction

pca_model = fit(PCA, hcat(FP_trials[string(TS)]...)'; maxoutdim=2)
#Trial 1
group1=[1,2,6]
group2=[4,3]
group3=[5,7,8,9]
#Trial 14
#group1=[1,2]
#group2=[3]
#group3=[4,5]
#Trial 20
# group1=[1,2]
# group2=[4]
# group3=[5,3]
# group1=[1]
# group2=[1]
# group3=[1]
#Trial 40
# group1=[1,5,10,11,12,13,17]
# group2=[2,3,4,6]
# group3=[15,16,7,8,9,14]
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
# n1=[8,14,7,13,3,2]
# n2=[9,10,10,8,9,1]
n1=[14,9,3,11,3,2]
n2=[9,8,1,8,5,1]
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


# Distance of the Data in the Regions

# Quadrant Distance between Dynamical objects
series_length = size(EmpData[TS])[1]
Δ₁ = fill(NaN,(num_fp,series_length))
Δ₂ = fill(NaN,(num_fp,series_length))

# Quadrant FP1
for i0 in 1:num_fp
    FP=FP_trials[string(TS)][i0]
    # get D matrices from
    D₁ = Array{Bool}(undef, hidden_dim, hidden_dim, korder)
    D₂ = Array{Bool}(undef, hidden_dim, hidden_dim, korder)
    D₁[:,:,1]=Diagonal((W₂*FP + h₂).>0)                    
    D₂[:,:,1]=Diagonal((W₂*FP).>0)
    # Quadrant FP2
    
    for i1 in 1:series_length
        FP2=EmpData[TS][i1,:]
        # get D matrices from
        DF₁ = Array{Bool}(undef, hidden_dim, hidden_dim, korder)
        DF₂ = Array{Bool}(undef, hidden_dim, hidden_dim, korder)
        DF₁[:,:,korder]=Diagonal((W₂*FP2 + h₂).>0)                    
        DF₂[:,:,korder]=Diagonal((W₂*FP2).>0)
        Δ₁[i0,i1] = sum(abs.(D₁-DF₁))
        Δ₂[i0,i1] = sum(abs.(D₂-DF₂))
    end
end
pp=Vector{Plots.Plot}()
f_p=[1,4,8]
for i in 1:3
    id_fp=f_p[i]
    p1=Plots.plot(Δ₁[id_fp,:],ylabel="Distance",title="FP $id_fp",labels="Δ₁")
    Plots.plot!(Δ₂[id_fp,:],ylabel="Distance",labels="Δ₂")
    push!(pp,p1)
end
p2=Plots.plot(EmpInput[TS][:,1],ylabel="Inputs",labels="Cue")
Plots.plot!(EmpInput[TS][:,2],labels="Gamble Reward")
Plots.plot!(EmpInput[TS][:,3],labels="Safe Reward")

push!(pp,p2)
sub=Plots.plot(pp[1],pp[2],pp[3],pp[4],layout=(4,1),size=(800,800),xlabel="Time Steps")


# Long Simulations

# Select initial condition
IC= EmpData[TS][26,:]
Time_Steps=1000000
num_Z=14
clipped=true
Gene_Emp=get_latent_time_series(Time_Steps,A,W₁,W₂,h₁,h₂,num_Z,z_0=IC,is_clipped=clipped)
Gene_dat=hcat(Gene_Emp...)'

pp=Vector{Plots.Plot}()
for i in 1:6
    ineu1=n1[i]#rand(1:14)
    ineu2=n2[i]#rand(1:14)
    p1=Plots.plot(
        Plots.scatter(matrix_FP[group1,ineu1],matrix_FP[group1,ineu2],tickfontsize=10,legend=false,
            xlabel="Neuron $ineu1",ylabel="Neuron $ineu2",title="Trial $TS"),
    )
    Plots.scatter!(matrix_FP[group2,ineu1],matrix_FP[group2,ineu2],tickfontsize=12,legend=false)
    Plots.scatter!(matrix_FP[group3,ineu1],matrix_FP[group3,ineu2],tickfontsize=12,legend=false)
    Plots.plot!(Gene_dat[900000:1000000,ineu1],Gene_dat[900000:1000000,ineu2],tickfontsize=12,legend=false)
    
    push!(pp, p1)
end

sub=Plots.plot(pp[1],pp[2],pp[3],pp[4],pp[5],pp[6],layout=(2,3),size=(1024,720))

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
# Self-excited Attractors??

Time_Steps=2000000
num_Z=14
clipped=true

pp=Vector{Plots.Plot}()
for i in 1:9
    ineu1=3#rand(1:14)
    ineu2=1#rand(1:14)
    IC= FP_trials[string(TS)][i]
    Gene_Emp=get_latent_time_series(Time_Steps,A,W₁,W₂,h₁,h₂,num_Z,z_0=IC,is_clipped=clipped)
    Gene_dat=hcat(Gene_Emp...)'
    p1=Plots.plot(
        Plots.scatter(matrix_FP[group1,ineu1],matrix_FP[group1,ineu2],tickfontsize=10,legend=false,
            xlabel="Neuron $ineu1",ylabel="Neuron $ineu2",title="FP $i"),
    )
    Plots.scatter!(matrix_FP[group2,ineu1],matrix_FP[group2,ineu2],tickfontsize=12,legend=false)
    Plots.scatter!(matrix_FP[group3,ineu1],matrix_FP[group3,ineu2],tickfontsize=12,legend=false)
    Plots.plot!(Gene_dat[1:1000,ineu1],Gene_dat[1:1000,ineu2],tickfontsize=12,legend=false,c=:black)
    Plots.plot!(Gene_dat[1900000:2000000,ineu1],Gene_dat[1900000:2000000,ineu2],tickfontsize=12,legend=false,c=:red)
    push!(pp, p1)
end

sub=Plots.plot(pp[1],pp[2],pp[3],pp[4],pp[5],pp[6],pp[7],pp[8],pp[9],layout=(3,3),size=(1024,720))




##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################


Ini_trial=findall(x->x==1,diff(EmpInput[TS][:,1]))
End_trial=[i-1 for i in Ini_trial[2:end]]
push!(End_trial,length(EmpInput[TS][:,1]))


pp=Vector{Plots.Plot}()
for i in 1:6
    ineu1=n1[i]#rand(1:14)
    ineu2=n2[i]#rand(1:14)
    p1=Plots.plot(
        Plots.scatter(matrix_FP[group1,ineu1],matrix_FP[group1,ineu2],tickfontsize=10,legend=false,
            xlabel="Neuron $ineu1",ylabel="Neuron $ineu2",title="Trial $TS"),
    )
    Plots.scatter!(matrix_FP[group2,ineu1],matrix_FP[group2,ineu2],tickfontsize=12,legend=false)
    Plots.scatter!(matrix_FP[group3,ineu1],matrix_FP[group3,ineu2],tickfontsize=12,legend=false)
    for it in 2:2
        Wstop=EmpData[TS][Ini_trial[it]-49:Ini_trial[it],:]
        cue_pos = findall(x->x==1,EmpInput[TS][Ini_trial[it]:End_trial[it],1]).+Ini_trial[it]
        Cue=EmpData[TS][cue_pos,:]
        RITI=EmpData[TS][cue_pos[end]:End_trial[it],:]
        Plots.plot!(Wstop[:,ineu1],Wstop[:,ineu2],tickfontsize=12,legend=false,linecolor=:blue)
        Plots.plot!(Cue[:,ineu1],Cue[:,ineu2],tickfontsize=12,legend=false,linecolor=:green)
        Plots.plot!(RITI[:,ineu1],RITI[:,ineu2],tickfontsize=12,legend=false,linecolor=:red)
    end
    push!(pp, p1)
end

sub=Plots.plot(pp[1],pp[2],pp[3],pp[4],pp[5],pp[6],layout=(2,3),size=(1024,720))




pp=Vector{Plots.Plot}()
for i in 1:6
    ineu1=n1[i]#rand(1:14)
    ineu2=n2[i]#rand(1:14)
    p1=Plots.plot(
        Plots.scatter(matrix_FP[group1,ineu1],matrix_FP[group1,ineu2],tickfontsize=10,legend=false,
            xlabel="Neuron $ineu1",ylabel="Neuron $ineu2",title="Trial $TS"),
    )
    Plots.scatter!(matrix_FP[group2,ineu1],matrix_FP[group2,ineu2],tickfontsize=12,legend=false)
    Plots.scatter!(matrix_FP[group3,ineu1],matrix_FP[group3,ineu2],tickfontsize=12,legend=false)
    for it in 1:length(Ini_trial)
        save=sum(EmpInput[TS][Ini_trial[it]:End_trial[it],3].==1)
        gamble=sum(EmpInput[TS][Ini_trial[it]:End_trial[it],2].==4)
        nrew = sum(sum(EmpInput[TS][Ini_trial[it]:End_trial[it],[2,3]]).==0)
        if save>0
            Plots.plot!(EmpData[TS][Ini_trial[it]-49:End_trial[it],ineu1],EmpData[TS][Ini_trial[it]-49:End_trial[it],ineu2],
            tickfontsize=12,legend=false,linecolor=:orange)
        elseif gamble>0
            Plots.plot!(EmpData[TS][Ini_trial[it]-49:End_trial[it],ineu1],EmpData[TS][Ini_trial[it]-49:End_trial[it],ineu2],
            tickfontsize=12,legend=false,linecolor=:green)
        else
            Plots.plot!(EmpData[TS][Ini_trial[it]-49:End_trial[it],ineu1],EmpData[TS][Ini_trial[it]-49:End_trial[it],ineu2],
            tickfontsize=12,legend=false,linecolor=:blue)
        end
    end
    push!(pp, p1)
end

sub=Plots.plot(pp[1],pp[2],pp[3],pp[4],pp[5],pp[6],layout=(2,3),size=(1024,720))


##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
# Distribution of external stimulus
# Cue Stimulus
cue_s0=[]
cue_s1=[]
Dec_prob=[]
iti_l=[]
for i in 1:length(EmpInput)
    ini_c=findall(x->x==1,diff(EmpInput[i][:,1]))
    end_c=findall(x->x==-1,diff(EmpInput[i][:,1]))
    iti=diff(ini_c).-50
    push!(iti_l,iti)
    push!(cue_s0,ini_c)
    push!(cue_s1,end_c)
    # Decision probability
    dec_ratio = 1-length(findall(x->x==1,diff(EmpInput[i][:,3])))/length(ini_c)
    push!(Dec_prob,dec_ratio)
end
cue_dist=vcat(cue_s1...)-vcat(cue_s0...)
iti_dist=vcat(iti_l...)

# Gamble Probability Reward
block_change = [22,45]
gamb_prob = Array{Float32}(undef,length(EmpInput))
for i in 1:length(EmpInput)
    if i<=block_change[1]
        gamb_prob[i]=0.75
    elseif i>block_change[1] & i<=block_change[2]
        gamb_prob[i]=0.12
    else
        gamb_prob[i]=0.25
    end
end

# Safe Probability Reward
safe_prob=0.90


function virtual_trials(
    cue_dist::Vector,
    Dec_prob::Vector,
    gamb_prob::Vector,
    safe_prob::Float64,
    iti_dist::Vector,
    IC::Vector)
        
    # Creation of virtual trials
    num_virtual_trial=100
    virt_trial=zeros(1,3)
    virt_dec = zeros(num_virtual_trial)
    virt_rew = zeros(num_virtual_trial)
    for vt in 1:num_virtual_trial
        wheel = zeros(50,3)             # wheel stop 1s
        lcue=cue_dist[rand(1:length(cue_dist))]  
        cue = zeros(lcue,3)
        cue[:,1].=1
        # Gamble choice
        if rand()<=Dec_prob[TS]
            virt_dec[vt]=1
            # Probability of gamble reward
            if rand()<gamb_prob[TS]
                # Yes Reward
                virt_rew[vt]=1
                reward=zeros(25,3)
                reward[:,2].=4
            else
                # No Reward
                reward=zeros(25,3)
            end
        # Safe choice
        else
            if rand()<safe_prob
                # Yes Reward
                virt_rew[vt]=1
                reward=zeros(25,3)
                reward[:,2].=1
            else
                # No Reward
                reward=zeros(25,3)
            end
        end
    iti_time = zeros(iti_dist[rand(1:length(iti_dist))],3)
    ind_trial=vcat(wheel,cue,reward,iti_time)
    virt_trial=vcat(virt_trial,ind_trial)
    end

    # Initialization of the external inputs for the simulation
    Time_Steps=100000
    warm_up=zeros(Time_Steps,3)
    # Creation Stimulus
    Stimulus = vcat(warm_up,virt_trial)
    # Select initial condition
    num_Z=14
    clipped=true
    Simulation_step=size(Stimulus)[1]
    Gene_Virt=get_latent_input_series(Simulation_step,A,W₁,W₂,h₁,h₂,Stimulus,num_Z,z_0=IC,is_clipped=clipped)
    Gene_trial=hcat(Gene_Virt...)'
    return Gene_trial
end


Virt_Trial = virtual_trials(cue_dist,Dec_prob,gamb_prob,safe_prob,iti_dist,EmpData[TS][1,:])

# Select initial condition
pp=Vector{Plots.Plot}()
for i in 1:6
    ineu1=n1[i]#rand(1:14)
    ineu2=n2[i]#rand(1:14)
    p1=Plots.plot(
        Plots.scatter(matrix_FP[group1,ineu1],matrix_FP[group1,ineu2],tickfontsize=10,legend=false,
            xlabel="Neuron $ineu1",ylabel="Neuron $ineu2",title="Trial 1"),
    )
    Plots.scatter!(matrix_FP[group2,ineu1],matrix_FP[group2,ineu2],tickfontsize=12,legend=false)
    Plots.scatter!(matrix_FP[group3,ineu1],matrix_FP[group3,ineu2],tickfontsize=12,legend=false)
    Plots.plot!(Virt_Trial[1:100000,ineu1],Virt_Trial[1:100000,ineu2],tickfontsize=12,legend=false)
    Plots.plot!(Virt_Trial[100000:120000,ineu1],Virt_Trial[100000:120000,ineu2],tickfontsize=12,legend=false)
    push!(pp, p1)
end

sub=Plots.plot(pp[1],pp[2],pp[3],pp[4],pp[5],pp[6],layout=(2,3),size=(1024,720))



##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

# Exploring the Attractors

Gam_prob=Dec_prob
Reward_prob=gamb_prob
# Gamble Reward
Gam_prob.=1
Reward_prob.=1
Virt_GReward = virtual_trials(cue_dist,Gam_prob,Reward_prob,safe_prob,iti_dist,EmpData[TS][1,:])
# Gamble No Reward
Gam_prob.=1.0
Reward_prob.=0.0
Virt_GNOReward = virtual_trials(cue_dist,Gam_prob,Reward_prob,safe_prob,iti_dist,EmpData[TS][1,:])
# Safe Reward
Gam_prob.=0
Reward_safe=1.0
Virt_SReward = virtual_trials(cue_dist,Gam_prob,Reward_prob,Reward_safe,iti_dist,EmpData[TS][1,:])
# Safe No Reward
Gam_prob.=0
Reward_safe=0.0
Virt_SNOReward = virtual_trials(cue_dist,Gam_prob,Reward_prob,Reward_safe,iti_dist,EmpData[TS][1,:])


# Select initial condition
pp=Vector{Plots.Plot}()
for i in 1:6
    ineu1=n1[i]#rand(1:14)
    ineu2=n2[i]#rand(1:14)
    p1=Plots.plot(
        Plots.plot(Virt_GReward[1:100000,ineu1],Virt_GReward[1:100000,ineu2],tickfontsize=12,legend=false,
        xlabel="Neuron $ineu1",ylabel="Neuron $ineu2",title="Trial $TS"),
    )
    Plots.plot!(Virt_GReward[100000:110000,ineu1],Virt_GReward[100000:110000,ineu2],color=:green,
    tickfontsize=12,legend=false)
    Plots.plot!(Virt_SReward[100000:110000,ineu1],Virt_SReward[100000:110000,ineu2],color=:orange,
    tickfontsize=12,legend=false)
    Plots.scatter!(matrix_FP[group1,ineu1],matrix_FP[group1,ineu2],tickfontsize=12,legend=false)
    Plots.scatter!(matrix_FP[group2,ineu1],matrix_FP[group2,ineu2],tickfontsize=12,legend=false)
    Plots.scatter!(matrix_FP[group3,ineu1],matrix_FP[group3,ineu2],tickfontsize=12,legend=false)
    push!(pp, p1)
end
sub=Plots.plot(pp[1],pp[2],pp[3],pp[4],pp[5],pp[6],layout=(2,3),size=(1024,720))


pp=Vector{Plots.Plot}()
for i in 1:6
    ineu1=n1[i]#rand(1:14)
    ineu2=n2[i]#rand(1:14)
    p1=Plots.plot(
        Plots.plot(Virt_GNOReward[1:100000,ineu1],Virt_GNOReward[1:100000,ineu2],tickfontsize=12,legend=false,
        xlabel="Neuron $ineu1",ylabel="Neuron $ineu2",title="Trial $TS"),
    )
    Plots.plot!(Virt_GNOReward[1:100000,ineu1],Virt_GNOReward[1:100000,ineu2],tickfontsize=12,legend=false)
    Plots.plot!(Virt_GNOReward[100000:110000,ineu1],Virt_GNOReward[100000:110000,ineu2],color=:grey,
    tickfontsize=12,legend=false)
    Plots.plot!(Virt_GReward[1:100000,ineu1],Virt_GReward[1:100000,ineu2],tickfontsize=12,legend=false)
    Plots.plot!(Virt_GReward[100000:110000,ineu1],Virt_GReward[100000:110000,ineu2],color=:green,
    tickfontsize=12,legend=false)
    Plots.scatter!(matrix_FP[group1,ineu1],matrix_FP[group1,ineu2],tickfontsize=12,legend=false)
    Plots.scatter!(matrix_FP[group2,ineu1],matrix_FP[group2,ineu2],tickfontsize=12,legend=false)
    Plots.scatter!(matrix_FP[group3,ineu1],matrix_FP[group3,ineu2],tickfontsize=12,legend=false)
    push!(pp, p1)
end

sub=Plots.plot(pp[1],pp[2],pp[3],pp[4],pp[5],pp[6],layout=(2,3),size=(1024,720))


pp=Vector{Plots.Plot}()
for i in 1:6
    ineu1=n1[i]#rand(1:14)
    ineu2=n2[i]#rand(1:14)
    p1=Plots.plot(
        Plots.plot(Virt_GNOReward[1:100000,ineu1],Virt_GNOReward[1:100000,ineu2],tickfontsize=12,legend=false,
        xlabel="Neuron $ineu1",ylabel="Neuron $ineu2",title="Trial $TS"),
    )
    Plots.plot!(Virt_SNOReward[1:100000,ineu1],Virt_SNOReward[1:100000,ineu2],tickfontsize=12,legend=false)
    Plots.plot!(Virt_SNOReward[100000:110000,ineu1],Virt_SNOReward[100000:110000,ineu2],color=:grey,
    tickfontsize=12,legend=false)
    Plots.plot!(Virt_SReward[1:100000,ineu1],Virt_SReward[1:100000,ineu2],tickfontsize=12,legend=false)
    Plots.plot!(Virt_SReward[100000:110000,ineu1],Virt_SReward[100000:110000,ineu2],color=:orange,
    tickfontsize=12,legend=false)
    Plots.scatter!(matrix_FP[group1,ineu1],matrix_FP[group1,ineu2],tickfontsize=12,legend=false)
    Plots.scatter!(matrix_FP[group2,ineu1],matrix_FP[group2,ineu2],tickfontsize=12,legend=false)
    Plots.scatter!(matrix_FP[group3,ineu1],matrix_FP[group3,ineu2],tickfontsize=12,legend=false)
    push!(pp, p1)
end

sub=Plots.plot(pp[1],pp[2],pp[3],pp[4],pp[5],pp[6],layout=(2,3),size=(1024,720))


##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################


# Understanding FPs
FP1=reshape(vcat(FP_trials["1"]...),(size(FP_trials["1"])[1],14))
FP2=reshape(vcat(FP_trials["14"]...),(size(FP_trials["14"])[1],14))
FP3=reshape(vcat(FP_trials["20"]...),(size(FP_trials["20"])[1],14))
FP4=reshape(vcat(FP_trials["40"]...),(size(FP_trials["40"])[1],14))
FP_Session=vcat(FP1,FP2,FP3,FP4)


pca_model = fit(PCA, FP_Session; maxoutdim=2)
#Trial 1
#group1=[1,2,4]
#group2=[3]
#group3=[5,6,7]
#Trial 14
#group1=[1,2]
#group2=[3]
#group3=[4,5]
#Trial 20
group1=[i for i in range(1,7)]
group2=[i for i in range(8,12)]
group3=[i for i in range(13,17)]
group4=[i for i in range(18,34)]
Plots.scatter(projection(pca_model)[group1[1]:group1[end],1], projection(pca_model)[group1[1]:group1[end],2], labels="FP $group1",legend= :outerbottom,
tickfontsize=12, legendfontsize=12, size=(1080,900))
Plots.scatter!(projection(pca_model)[group2[1]:group2[end],1], projection(pca_model)[group2[1]:group2[end],2],labels="FP $group2")
Plots.scatter!(projection(pca_model)[group3[1]:group3[end],1], projection(pca_model)[group3[1]:group3[end],2],labels="FP $group3")
Plots.scatter!(projection(pca_model)[group4[1]:group4[end],1], projection(pca_model)[group4[1]:group4[end],2],labels="FP $group4")
Plots.xlabel!("PC1")
Plots.ylabel!("PC2")
Plots.title!("FPs reduced dimensions")


##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

# Trials Transitions Limiting behaviour atractors
# Select initial condition

Time_Steps=1000000
num_Z=14
clipped=true
Trial_id = [44,45,46,47,48,49,50,51,52]

pp=Vector{Plots.Plot}()
for i in 1:9
    ineu1=8#rand(1:14)
    ineu2=9#rand(1:14)
    id = Trial_id[i]
    IC= EmpData[Trial_id[i]][26,:]
    W₁ₜ = data[2][Trial_id[i],:,:]
    W₂ₜ = data[3][Trial_id[i],:,:]
    Aₜ = data[1]
    h₁ₜ = data[5]
    h₂ₜ = data[4]
    matrix_FP=hcat(FP_trials[string(id)]...)'
    Gene_Emp=get_latent_time_series(Time_Steps,Aₜ,W₁ₜ,W₂ₜ,h₁ₜ,h₂ₜ,num_Z,z_0=IC,is_clipped=clipped)   
    Gene_dat=hcat(Gene_Emp...)'
    p1=Plots.plot(
        Plots.plot(Gene_dat[990000:1000000,ineu1],Gene_dat[990000:1000000,ineu2],tickfontsize=12,legend=false,
            c=:red,xlabel="Neuron $ineu1",ylabel="Neuron $ineu2",title="Trial $id"),
    )
    Plots.scatter!(matrix_FP[:,ineu1],matrix_FP[:,ineu2],tickfontsize=12,legend=false,c=:green)  
    push!(pp, p1)
end

sub=Plots.plot(pp[1],pp[2],pp[3],pp[4],pp[5],pp[6],pp[7],pp[8],pp[9],layout=(3,3),size=(1024,720))

# 


Gam_prob=Dec_prob
Reward_prob=gamb_prob
Trial_id = [1,14,20,26,28,30,44,48,50]

pp=Vector{Plots.Plot}()
for i in 1:9
    ineu1=8#rand(1:14)
    ineu2=9#rand(1:14)
    id = Trial_id[i]
    W₁ = data[2][id,:,:]
    W₂ = data[3][id,:,:]
    A = data[1]
    h₁ = data[5]
    h₂ = data[4]
    id = Trial_id[i]
    # Gamble Reward
    matrix_FP=hcat(FP_trials[string(id)]...)'
    Gam_prob.=1
    Reward_prob.=1
    Virt_GReward = virtual_trials(cue_dist,Gam_prob,Reward_prob,safe_prob,iti_dist,EmpData[id][1,:])
    # Gamble No Reward
    Gam_prob.=1.0
    Reward_prob.=0.0
    Virt_GNOReward = virtual_trials(cue_dist,Gam_prob,Reward_prob,safe_prob,iti_dist,EmpData[id][1,:])
    pca_model = fit(PCA, hcat(FP_trials[string(id)]...)'; maxoutdim=2)
    p1=Plots.plot(
        Plots.plot(Virt_GNOReward[:,ineu1],Virt_GNOReward[:,ineu2],tickfontsize=12,legend=false,
        color=:grey,xlabel="Neuron $ineu1",ylabel="Neuron $ineu2",title="Trial $id"),
    )
    Plots.plot!(Virt_GReward[:,ineu1],Virt_GReward[:,ineu2],color=:green,
    tickfontsize=12,legend=false)
    Plots.scatter!(matrix_FP[:,ineu1],matrix_FP[:,ineu2],tickfontsize=12,legend=false,c=:red)
    push!(pp, p1)
end


sub=Plots.plot(pp[1],pp[2],pp[3],pp[4],pp[5],pp[6],pp[7],pp[8],pp[9],layout=(3,3),size=(1024,720))


Gam_prob=Dec_prob
Reward_prob=gamb_prob
Trial_id = [1,14,20,26,28,30,44,48,50]

pp=Vector{Plots.Plot}()
for i in 1:9
    ineu1=8#rand(1:14)
    ineu2=9#rand(1:14)
    id = Trial_id[i]
    W₁ = data[2][id,:,:]
    W₂ = data[3][id,:,:]
    A = data[1]
    h₁ = data[5]
    h₂ = data[4]
    id = Trial_id[i]
    # Gamble Reward
    matrix_FP=hcat(FP_trials[string(id)]...)'
    # Safe Reward
    Gam_prob.=0
    Reward_safe=1.0
    Virt_SReward = virtual_trials(cue_dist,Gam_prob,Reward_prob,Reward_safe,iti_dist,EmpData[TS][1,:])
    # Safe No Reward
    Gam_prob.=0
    Reward_safe=0.0
    Virt_SNOReward = virtual_trials(cue_dist,Gam_prob,Reward_prob,Reward_safe,iti_dist,EmpData[TS][1,:])
    pca_model = fit(PCA, hcat(FP_trials[string(id)]...)'; maxoutdim=2)
    p1=Plots.plot(
        Plots.plot(Virt_SNOReward[:,ineu1],Virt_SNOReward[:,ineu2],tickfontsize=12,legend=false,
        color=:grey,xlabel="Neuron $ineu1",ylabel="Neuron $ineu2",title="Trial $id"),
    )
    Plots.plot!(Virt_SReward[:,ineu1],Virt_SReward[:,ineu2],color=:orange,
    tickfontsize=12,legend=false)
    Plots.scatter!(matrix_FP[:,ineu1],matrix_FP[:,ineu2],tickfontsize=12,legend=false,c=:red)
    push!(pp, p1)
end


sub=Plots.plot(pp[1],pp[2],pp[3],pp[4],pp[5],pp[6],pp[7],pp[8],pp[9],layout=(3,3),size=(1024,720))

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################


pp=Vector{Plots.Plot}()
for i in 1:9
    ineu1=8#rand(1:14)
    ineu2=9#rand(1:14)
    id = Trial_id[i]
    W₁ = data[2][id,:,:]
    W₂ = data[3][id,:,:]
    A = data[1]
    h₁ = data[5]
    h₂ = data[4]
    id = Trial_id[i]
    # Gamble Reward
    matrix_FP=hcat(FP_trials[string(id)]...)'
    # Gamble Reward
    matrix_FP=hcat(FP_trials[string(id)]...)'
    Gam_prob.=1
    Reward_prob.=1
    Virt_GReward = virtual_trials(cue_dist,Gam_prob,Reward_prob,safe_prob,iti_dist,EmpData[id][1,:])
    # Safe Reward
    Gam_prob.=0
    Reward_safe=1.0
    Virt_SReward = virtual_trials(cue_dist,Gam_prob,Reward_prob,Reward_safe,iti_dist,EmpData[TS][1,:])
    pca_model = fit(PCA, hcat(FP_trials[string(id)]...)'; maxoutdim=2)
    p1=Plots.plot(
        Plots.plot(Virt_SReward[:,ineu1],Virt_SReward[:,ineu2],tickfontsize=12,legend=false,
        color=:orange,xlabel="Neuron $ineu1",ylabel="Neuron $ineu2",title="Trial $id"),
    )
    Plots.plot!(Virt_GReward[:,ineu1],Virt_GReward[:,ineu2],color=:green,
    tickfontsize=12,legend=false)
    Plots.scatter!(matrix_FP[:,ineu1],matrix_FP[:,ineu2],tickfontsize=12,legend=false,c=:red)
    push!(pp, p1)
end


sub=Plots.plot(pp[1],pp[2],pp[3],pp[4],pp[5],pp[6],pp[7],pp[8],pp[9],layout=(3,3),size=(1024,720))
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################



Diag_dir=zeros(14,2,17)

for ifp in 1:17
    un_imag=unique(abs.(imag(EV_trials["40"][ifp])))
    for i in eachindex(un_imag)
        pos_x=findall(x->x==un_imag[i],abs.(imag(EV_trials["40"][ifp])))
        if un_imag[i]==0
            Diag_dir[pos_x,2,ifp]=-1*pos_x
        else
            Diag_dir[pos_x,2,ifp]=pos_x
        end
        pos_y=findall(x->x>0,real(EV_trials["40"][ifp]))
        Diag_dir[pos_y,1,ifp]=pos_y
        pos_y=findall(x->x<0,real(EV_trials["40"][ifp]))
        Diag_dir[pos_y,1,ifp]=-1*pos_y
    end
end



Plots.scatter(Diag_dir[:,2,1],Diag_dir[:,1,1])
Plots.xlims!(-15,15)
Plots.ylims!(-15,15)

Plots.scatter(Diag_dir[:,2,2],Diag_dir[:,1,2])
Plots.xlims!(-15,15)
Plots.ylims!(-15,15)

Plots.scatter(Diag_dir[:,2,3],Diag_dir[:,1,3])
Plots.xlims!(-15,15)
Plots.ylims!(-15,15)

Plots.scatter(Diag_dir[:,2,4],Diag_dir[:,1,4])
Plots.xlims!(-15,15)
Plots.ylims!(-15,15)

Plots.scatter(Diag_dir[:,2,5],Diag_dir[:,1,5])
Plots.xlims!(-15,15)
Plots.ylims!(-15,15)

Plots.scatter(Diag_dir[:,2,6],Diag_dir[:,1,6])
Plots.xlims!(-15,15)
Plots.ylims!(-15,15)

Plots.scatter(Diag_dir[:,2,7],Diag_dir[:,1,7])
Plots.xlims!(-15,15)
Plots.ylims!(-15,15)