"""
    MarabouBinarySearch(optimizer)

Iterative calls to Maraou that narrow in on the optimum value

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: hpolytope and hyperrectangle
3. Objective: Any linear objective

# Method
Repeated calls to Marabou. Each one eliminates a halfspace of possible
optimum values.

# Property
Sound and complete

"""
@with_kw struct MarabouBinarySearch
    optimizer = GLPK.Optimizer
    use_sbt = false
end
function optimize(solver::MarabouBinarySearch, problem::OutputOptimizationProblem, time_limit::Int = 1200)
    start_function_time = time()
    @debug "Optimizing with Marabou Binary Search"
    @assert problem.input isa Hyperrectangle or problem.input isa HPolytope

    # Find a feasible solution to pass it a lower bound
    model = Model(solver)
    input_vars = @variable(model, [i=1:size(problem.network.layers[1].weights, 2)])
    add_set_constraint!(model, problem.input, input_vars)
    optimize!(model)
    @assert (value.(input_vars) ∈ problem.input)
    @debug "Found feasible point"

    feasible_val = compute_objective(problem.network, value.(input_vars), problem.objective)
    if (!problem.max)
        feasible_val = feasible_val * -1.0
    end

    @debug "Feasible value: " feasible_val

    @debug "before making data files"

    # write out input constraint matrix A, input constraint vector b, and
    # objective weight vector c. Then, write nnet file
    data_file = "./src/utils/temp_files_for_transfer/temp_marabou.npz"
    network_file = "./src/utils/temp_files_for_transfer/temp_marabou_network.nnet"
    result_file = "./src/utils/temp_files_for_transfer/result.npz"

    # Augment the network to handle an arbitrary linear objective
    # if the last layer was ID() then this just combines the objective into that layer
    # TODO: Do we lose any efficiency if we expand without
    negative_objective = problem.max
    augmented_network = extend_network_with_objective(problem.network, problem.objective, !problem.max) # If the last layer is ID it won't add a layer
    @debug "after extend network with objective"


    A, b = tosimplehrep(problem.input)
    # Condense to sparse representation - three vectors (rows, cols, and their values)
    non_zero_indices = findall(!iszero, A)
    row_indices = [index[1] for index in non_zero_indices]
    col_indices = [index[2] for index in non_zero_indices]
    values = A[non_zero_indices]

    # Subtract 1 off the indices to match with python indexing
    npzwrite(data_file, Dict("A_rows" => row_indices .- 1, "A_cols" => col_indices .- 1, "A_values" => values, "b" => b, "feasible_value" => feasible_val))
    write_nnet(network_file, augmented_network)

    # Call MarabouPy.py with the path to the needed files
    # the command will maximize - so we will flip negative after if need be
    call_command_time = time()
    command = `python ./src/exact/MarabouBinarySearchPy.py  $data_file $network_file $result_file $time_limit`
    run(command)

    @debug "Time to get to call command: " (call_command_time - start_function_time)
    # Read back in the result
    result = np.load(result_file)
    status = -1
    if (get(result, :status) == "success")[1] # [1] b/c of how the equality comparison returns its result with the python object
        status = :success
    elseif (get(result, :status) == "timeout")[1]
        status = :timeout
    else
        status = :error
    end

    # Account for maximization vs minimization - we will have augmented the network
    # to get the maximum negative objective if we were minimizing so we need to flip it back
    obj_val = get(result, :objective_value)[1]
    if (!problem.max)
        obj_val = -1.0 * obj_val
    end

    return Result(status, get(result, :input), obj_val)
end
