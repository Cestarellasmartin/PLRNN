using JLD2
using NPZ
using Pickle
using LinearAlgebra
using Plots
using MultivariateStats
using StatsPlots
using Statistics

include("src/utilities/helpers.jl")

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

# Real data
TS = 20
EmpData = Pickle.npyload("D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17_reduction/datasets/Training_data.npy") 
EmpInput = Pickle.npyload("D:/_work_cestarellas/Analysis/PLRNN/noautoencoder/neuralactivity/OFC/CE17_reduction/datasets/Training_inputs.npy")  

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
Trial_id = [1,14,20,26,28,30,44,48,50]

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
    #Plots.scatter!(matrix_FP[:,ineu1],matrix_FP[:,ineu2],tickfontsize=12,legend=false,c=:green)  
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