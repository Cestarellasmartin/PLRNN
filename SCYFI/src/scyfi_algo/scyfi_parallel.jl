using .. Utilities
using Base.Threads

"""
A,W,h PLRNN parameters
heuristic algorithm of finding FP (Durstewitz 2017) extended to find all k cycles
We need to solve: (A+WD_k)*...*(A+WD_1) *z + [(A+WD_k)*...*(A+WD_2)+...+(A+WD)**1+1]*h = z
parallelized on CPU
"""
function scy_fi(
    A:: Array, W:: Array, h:: Array, order:: Integer, found_lower_orders:: Array,n_threads::Integer;
    outer_loop_iterations:: Union{Integer,Nothing}= nothing,
    inner_loop_iterations:: Union{Integer,Nothing} = nothing
     )
    dim = size(A)[1]
    cycles_found = Array[]
    eigvals =  Array[]
    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(order, outer_loop_iterations, inner_loop_iterations)
    # Do the multithreading over the initializations
    n_threads = Threads.nthreads()
    lk = ReentrantLock()
    Threads.@threads for thread_iterator = 1:n_threads
        i = -1
        while i < outer_loop_iterations # This loop can be viewed as (re-)initialization of the algo in some set of linear regions
            i += 1
            relu_matrix_list = construct_relu_matrix_list(dim, order)   # generate random set of linear regions to start from
            difference_relu_matrices = 1                                # flag that indiates wheter the initilized relu matrices are the same as the candidate matrices
            c = 0
            while c < inner_loop_iterations                             # This loop calculates cycle candidates, checks if they are virtual and if they are initializes 
                c += 1                                                  # the next calculation in the linear region of that virtual cycle
                z_candidate = get_cycle_point_candidate(A, W, relu_matrix_list, h, order)           # calculate cycle candidate
                if z_candidate !== nothing
                    trajectory = get_latent_time_series(order, A, W, h, dim, z_0=z_candidate)      
                    trajectory_relu_matrix_list = Array{Bool}(undef, dim, dim, order)
                    for j = 1:order
                        trajectory_relu_matrix_list[:,:,j] = Diagonal(trajectory[j].>0)                                      # get relu matrices of the candidate
                    end
                    for j = 1:order
                        difference_relu_matrices = sum(abs.(trajectory_relu_matrix_list[:,:,j].-relu_matrix_list[:,:,j]))   # check if cycle candidate lies in the same linear regions as the initialization
                        if difference_relu_matrices != 0
                            break
                        end
                        if !isempty(found_lower_orders)
                            if map(temp -> round.(temp, digits=2), trajectory[1]) ∈ map(temp -> round.(temp,digits=2),collect(Iterators.flatten(Iterators.flatten(found_lower_orders))))
                                difference_relu_matrices = 1
                                break
                            end
                        end
                    end
                    if difference_relu_matrices == 0  # if the linear regions match check if we already found that cycle
                        if map(temp1 -> round.(temp1, digits=2), trajectory[1]) ∉ map(temp -> round.(temp, digits=2), collect(Iterators.flatten(cycles_found)))
                            e = get_eigvals(A,W,relu_matrix_list,order)
                            lock(lk) do
                                push!(cycles_found,trajectory)
                                push!(eigvals,e)
                                i=0
                                c=0
                            end
                        end
                    end
                    if relu_matrix_list == trajectory_relu_matrix_list
                        relu_matrix_list = construct_relu_matrix_list(dim, order)
                    else
                        relu_matrix_list = trajectory_relu_matrix_list # if we did not find a real cycle use the regions of the virtual cycle to recalculate 
                    end
                else
                    relu_matrix_list = construct_relu_matrix_list(dim, order)
                end 
            end
        end
    end
    return cycles_found, eigvals
end


"""
heuristic algorithm of finding FP (Durstewitz 2017) extended to find all k cycles and for the shPLRNN
parallelized on CPU
shPLRNN
"""
function scy_fi(
    A::AbstractVector,
    W₁::AbstractMatrix,
    W₂::AbstractMatrix,
    h₁::AbstractVector,
    h₂::AbstractVector,
    order:: Integer,
    found_lower_orders:: Array,
    relu_pool:: Array,
    PLRNN::ShallowPLRNN,
    n_threads::Integer;
    outer_loop_iterations:: Union{Integer,Nothing} = nothing,
    inner_loop_iterations:: Union{Integer,Nothing} = nothing,
    get_pool_from_traj:: Bool = false,
    num_trajectories:: Integer, 
    len_trajectories:: Integer
    )
    
    latent_dim = size(A)[1]
    hidden_dim = size(h₂)[1]
    cycles_found = Array[]
    eigvals =  Array[]
    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(order, outer_loop_iterations, inner_loop_iterations)
    n_threads = Threads.nthreads()
    lk = ReentrantLock()

    Threads.@threads for thread_iterator = 1:n_threads
        i = -1
        while i < outer_loop_iterations # This loop can be viewed as (re-)initialization of the algo in some set of linear regions
            i += 1
            relu_matrix_list = construct_relu_matrix_list(relu_pool, order)     # generate random set of linear regions from pool to start from
            difference_relu_matrices = 1                                        # flag that indiates wheter the initilized relu matrices are the same as the candidate matrices
            c = 0
            while c < inner_loop_iterations
                c += 1
                z_candidate = get_cycle_point_candidate(A, W₁, W₂, h₁, h₂, relu_matrix_list, order)
                if z_candidate !== nothing
                    trajectory = get_latent_time_series(order, A, W₁, W₂, h₁, h₂, latent_dim, z_0=z_candidate)
                    trajectory_relu_matrix_list = Array{Bool}(undef, hidden_dim, hidden_dim, order)
                    for j = 1:order
                        trajectory_relu_matrix_list[:,:,j] = Diagonal((W₂*trajectory[j] + h₂).>0)                       # get relu matrices of the candidate
                    end
                    for j = 1:order
                        difference_relu_matrices = sum(abs.(trajectory_relu_matrix_list[:,:,j].-relu_matrix_list[:,:,j])) # check if cycle candidate lies in the same linear regions as the initialization
                        if difference_relu_matrices != 0
                            break
                        end
                        if !isempty(found_lower_orders)
                            if map(temp -> round.(temp, digits=2), trajectory[1]) ∈ map(temp -> round.(temp,digits=2),collect(Iterators.flatten(Iterators.flatten(found_lower_orders))))
                                difference_relu_matrices = 1
                                break
                            end
                        end
                    end
                    if difference_relu_matrices == 0    # if the linear regions match check if we already found that cycle
                        if map(temp1 -> round.(temp1, digits=2), trajectory[1]) ∉ map(temp -> round.(temp, digits=2), collect(Iterators.flatten(cycles_found)))
                            e = get_eigvals(A, W₁, W₂, relu_matrix_list, order)
                            lock(lk) do
                                push!(cycles_found,trajectory)
                                push!(eigvals,e)
                                i=0
                                c=0
                            end
                        end
                    end
                    if relu_matrix_list == trajectory_relu_matrix_list
                        relu_matrix_list = construct_relu_matrix_list(relu_pool, order)
                    else
                        relu_matrix_list = trajectory_relu_matrix_list  # if we did not find a real cycle use the regions of the virtual cycle to recalculate 
                    end
                else
                    relu_matrix_list = construct_relu_matrix_list(relu_pool, order)
                end 
            end
        end
    end
    return cycles_found, eigvals
end


"""
heuristic algorithm of finding FP extended to find all k cycles and for the clipped shPLRNN
parallelized on CPU
clipped shPLRNN
"""
function scy_fi(
    A::AbstractVector,
    W₁::AbstractMatrix,
    W₂::AbstractMatrix,
    h₁::AbstractVector,
    h₂::AbstractVector,
    order:: Integer,
    found_lower_orders:: Array,
    relu_pool:: Array,
    PLRNN::ClippedShallowPLRNN,
    n_threads::Integer;
    outer_loop_iterations:: Union{Integer,Nothing} = nothing,
    inner_loop_iterations:: Union{Integer,Nothing} = nothing,
    get_pool_from_traj:: Bool = false,
    num_trajectories:: Integer=10, 
    len_trajectories:: Integer=100
     )
    
    latent_dim = size(A)[1]
    hidden_dim = size(h₂)[1]
    cycles_found = Array[]
    eigvals =  Array[]
    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(order, outer_loop_iterations, inner_loop_iterations)
    n_threads = Threads.nthreads()
    lk = ReentrantLock()

    Threads.@threads for thread_iterator = 1:n_threads
        i = -1
        while i < outer_loop_iterations # This loop can be viewed as (re-)initialization of the algo in some set of linear regions
            i += 1
            relu_matrix_list_1,relu_matrix_list_2 = construct_relu_matrix_list(relu_pool, order,hidden_dim)     # generate random set of linear regions from pool to start from
           # relu_matrix_list_1 = construct_relu_matrix_list(relu_pool, order)     # generate random set of linear regions from pool to start from
            #relu_matrix_list_2 = relu_matrix_list_1 
    
            difference_relu_matrices = 1                                        # flag that indiates wheter the initilized relu matrices are the same as the candidate matrices
            c = 0
            while c < inner_loop_iterations
                c += 1
                z_candidate = get_cycle_point_candidate(A, W₁, W₂, h₁, h₂, relu_matrix_list_1, relu_matrix_list_2, order)
                if z_candidate !== nothing
                    trajectory = get_latent_time_series(order, A, W₁, W₂, h₁, h₂, latent_dim, z_0=z_candidate, is_clipped=true)
                    trajectory_relu_matrix_list_1 = Array{Bool}(undef, hidden_dim, hidden_dim, order)
                    trajectory_relu_matrix_list_2 = Array{Bool}(undef, hidden_dim, hidden_dim, order)
                    for j = 1:order
                        trajectory_relu_matrix_list_1[:,:,j] = Diagonal((W₂*trajectory[j] + h₂).>0)                       # get relu matrices of the candidate
                        trajectory_relu_matrix_list_2[:,:,j] = Diagonal((W₂*trajectory[j]).>0)                       # get relu matrices of the candidate
                    end
                    for j = 1:order
                        difference_relu_matrices = sum(abs.(trajectory_relu_matrix_list_1[:,:,j].-relu_matrix_list_1[:,:,j])) # check if cycle candidate lies in the same linear regions as the initialization
                        difference_relu_matrices += sum(abs.(trajectory_relu_matrix_list_2[:,:,j].-relu_matrix_list_2[:,:,j]))
                        if difference_relu_matrices != 0
                            break
                        end
                        if !isempty(found_lower_orders)
                            if map(temp -> round.(temp, digits=2), trajectory[1]) ∈ map(temp -> round.(temp,digits=2),collect(Iterators.flatten(Iterators.flatten(found_lower_orders))))
                                difference_relu_matrices = 1
                                break
                            end
                        end
                    end
                    if difference_relu_matrices == 0    # if the linear regions match check if we already found that cycle
                        if map(temp1 -> round.(temp1, digits=2), trajectory[1]) ∉ map(temp -> round.(temp, digits=2), collect(Iterators.flatten(cycles_found)))
                            e = get_eigvals(A, W₁, W₂, relu_matrix_list_1, relu_matrix_list_2, order)
                            lock(lk) do
                                push!(cycles_found,trajectory)
                                push!(eigvals,e)
                                i=0
                                c=0
                            end
                        end
                    end
                    if relu_matrix_list_1 == trajectory_relu_matrix_list_1 && relu_matrix_list_2 == trajectory_relu_matrix_list_2
                        relu_matrix_list_1,relu_matrix_list_2 = construct_relu_matrix_list(relu_pool, order,hidden_dim)
                        #relu_matrix_list_1 = construct_relu_matrix_list(relu_pool, order)     # generate random set of linear regions from pool to start from
                        #relu_matrix_list_2 = relu_matrix_list_1 
    
                    else
                        relu_matrix_list_1 = trajectory_relu_matrix_list_1  # if we did not find a real cycle use the regions of the virtual cycle to recalculate 
                        relu_matrix_list_2 = trajectory_relu_matrix_list_2
                    end
                else
                    relu_matrix_list_1,relu_matrix_list_2 = construct_relu_matrix_list(relu_pool, order,hidden_dim)
                    #relu_matrix_list_1 = construct_relu_matrix_list(relu_pool, order)     # generate random set of linear regions from pool to start from
                    #relu_matrix_list_2 = relu_matrix_list_1 
    
                end 
            end
        end
    end
    return cycles_found, eigvals
end



