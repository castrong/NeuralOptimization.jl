"""
    Marabou(optimizer)

A branch and bound search with frequent bound tightening

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: hpolytope and hyperrectangle
3. Objective: Any linear objective

# Method
Branch and bound with bound tightening at each step in the simplex

# Property
Sound and complete

"""
@with_kw struct Marabou
    use_sbt = false
end
function optimize(solver::Marabou, problem::OutputOptimizationProblem, time_limit::Int = 1200)
    @debug "Optimizing with Marabou"

    # write out input constraint matrix A, input constraint vector b, and
    # objective weight vector c. Then, write nnet file
    data_file = "./src/utils/temp_files_for_transfer/temp_marabou.npz"
    network_file = "./src/utils/temp_files_for_transfer/temp_marabou_network.nnet"
    result_file = "./src/utils/temp_files_for_transfer/result.npz"

    # If our last layer is ID we can replace the last layer and combine
    # it with the objective - shouldn't change performance but is
    # consistent with the network structure for Sherlock
    if (problem.network.layers[end].activation == Id())
        @debug "In Marabou incorporating into single output layer"
        augmented_network = extend_network_with_objective(problem.network, problem.objective) # If the last layer is ID it won't add a layer
        augmented_objective = LinearObjective([1.0], [1])
        augmented_problem = OutputOptimizationProblem(augmented_network, problem.input, augmented_objective, problem.max)
    else
        augmented_problem = problem
    end

    A, b = tosimplehrep(augmented_problem.input)
    weight_vector = linear_objective_to_weight_vector(augmented_problem.objective, length(augmented_problem.network.layers[end].bias))

    # account for maximization vs minimization
    if (!augmented_problem.max)
        weight_vector = -1.0 * weight_vector
    end

    npzwrite(data_file, Dict("A" => A, "b" => b, "weight_vector" => weight_vector))
    write_nnet(network_file, augmented_problem.network)

    # Call MarabouPy.py with the path to the needed files
    command = `python ./src/exact/MarabouOptimizationPy.py  $data_file $network_file $result_file $time_limit`
    run(command)

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

    obj_val = get(result, :objective_value)[1]
    # account for maximization vs minimization
    if (!augmented_problem.max)
        obj_val = -1.0 * obj_val
    end

    return Result(status, get(result, :input), obj_val)
end
