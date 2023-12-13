using JLD2
using NPZ

# Loading file from SCYFI output
path_file="D:/_work_cestarellas/Analysis/PLRNN/SCYFI/data/"
cd(path_file)
Atrc=JLD2.load("FP_CE17_L6_00_Trial_28.jld2")
# Spliting FPs and their Eigen values
dyn_obj=Atrc["res"][1]
eigen_values=Atrc["res"][2]

# Determining the size of objects determined (in parallel version: high prob of repeated FPs)
num_obj=size(dyn_obj[1])[1]
# Determining the size of latent states 
num_states=size(dyn_obj[1][1][1])[1]

println("Number of Objects: ", num_obj)
println("Dim Latent States: ", num_states)

# Generating a matrix to cointain all FPs and other for Eigen Values
matrix_fp=zeros((num_states,num_obj))
matrix_ev=zeros(ComplexF64,(num_states,num_obj))

for i in 1:num_obj
    matrix_fp[:,i]=dyn_obj[1][i][1]
    matrix_ev[:,i]=eigen_values[1][i]
end

#Identify copies of FP and create a matrix with unique FP and Eigen Values
unique_FP=reduce(hcat,unique(eachcol(matrix_fp)))
unique_EV=reduce(hcat,unique(eachcol(matrix_ev)))

println("Number of Unique FPs :", size(unique_FP)[2])

# Saving Data as a dictionary for python
save_path="D:/_work_cestarellas/Analysis/PLRNN/SCYFI/Attractors/CE17/M0"
cd(save_path)
npzwrite("FP_CE17_00_28.npz", Dict("FP"=>unique_FP,"EV"=>unique_EV))
