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

folder_path="D:\\_work_cestarellas\\Analysis\\PLRNN\\SCYFI\\data\\CE17_red"
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
                    rev=real(EO[iob])
                    if (sum(rev.>0)/size(rev)[1])==1
                        push!(Stab,1)
                    elseif (sum(rev.>0)/size(rev)[1])==0
                        push!(Stab,-1)
                    else
                        push!(Stab,0)
                    end
                end
            else
            # K-cycle > 1
                for iob in 1:num_ob
                    flag=maximum(diff(DO[iob])).>0.000001
                    if sum(flag)>0
                        push!(FP,DO[iob])
                        push!(EV,EO[iob])
                        push!(kC,ik)
                        rev=real(EO[iob])
                        if (sum(rev.>0)/size(rev)[1])==1
                            push!(Stab,1)
                        elseif (sum(rev.>0)/size(rev)[1])==0
                            push!(Stab,-1)
                        else
                            push!(Stab,0)
                        end
                    else
                        push!(FP,DO[iob][1])
                        push!(EV,EO[iob])
                        push!(kC,1)
                        rev=real(EO[iob])
                        if (sum(rev.>0)/size(rev)[1])==1
                            push!(Stab,1)
                        elseif (sum(rev.>0)/size(rev)[1])==0
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

groupedbar([FP_num KCY_num],
bar_position= :stack,
bar_width=0.5,
xlabel="Trials",ylabel="# of Different Objects",
title="FP and Cycles detected in the session",
label=["FP" "K-cycles"],
yticks=(1:15),
legend=:topleft)


groupedbar([ST_num UNST_num BIST_num],
bar_position= :stack,
bar_width=0.5,
xlabel="Trials",ylabel="# of Different Objects",
title="FP and Cycles detected in the session",
label=["Stable" "Unstable" "Bistable"],
yticks=(1:15),
legend=:topleft)

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
xticks!(1:7,cust_xtick)
yticks!(1:7,cust_xtick)
xlims!(0.5, num_fp+0.5)


sort_fp=sortperm(Δ₁[:,1])
cust_xtick=[string(x) for x in sort_fp]
heatmap(Δ₁[sort_fp,sort_fp], color=:viridis, aspect_ratio=:equal, 
xlabel="FPs", ylabel="FP", title="W₂*FP+h₂",
colorbar_title="Distance")
xticks!(1:7,cust_xtick)
yticks!(1:7,cust_xtick)
xlims!(0.5, num_fp+0.5)
# Working with K-cycles

# Work in progress...


##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

# Classification of FPs
# Select Trial
it=5
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
pie(labels, sizes, title="FP $it", legend=true)

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
# PCA dimensional reduction

pca_model = fit(PCA, hcat(FP_trials[string(TS)]...)'; maxoutdim=2)
# group1=[1,2,4]
# group2=[3]
# group3=[5,6,7]
group1=[2,5]
group2=[1,3]
group3=[4]
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
    Plots.plot!(EmpData[TS][:,ineu1],EmpData[TS][:,ineu2],tickfontsize=12,legend=false)
    
    push!(pp, p1)
end

sub=Plots.plot(pp[1],pp[2],pp[3],pp[4],pp[5],pp[6],layout=(2,3),size=(1024,720))


# Long Simulations

# Select initial condition
IC= EmpData[TS][26,:]
Time_Steps=100000
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
            xlabel="Neuron $ineu1",ylabel="Neuron $ineu2",title="Trial 1"),
    )
    Plots.scatter!(matrix_FP[group2,ineu1],matrix_FP[group2,ineu2],tickfontsize=12,legend=false)
    Plots.scatter!(matrix_FP[group3,ineu1],matrix_FP[group3,ineu2],tickfontsize=12,legend=false)
    Plots.plot!(Gene_dat[:,ineu1],Gene_dat[:,ineu2],tickfontsize=12,legend=false)
    
    push!(pp, p1)
end

sub=Plots.plot(pp[1],pp[2],pp[3],pp[4],pp[5],pp[6],layout=(2,3),size=(1024,720))


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













