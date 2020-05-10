"""
    Sherlock(optimizer, ϵ::Float64)

Sherlock combines local and global search to estimate the range of the output node.

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
    dummy_var = 1
end
function optimize(solver::Marabou, problem::OutputOptimizationProblem, timeout=100)
    @debug "Optimizing with Marabou"

    # write out input constraint matrix A, input constraint vector b, and
    # objective weight vector c. Then, write nnet file
    data_file = "./utils/temp_files_for_transfer/temp_marabou.npz"
    network_file = "./utils/temp_files_for_transfer/temp_marabou_network.nnet"
    result_file = "./utils/temp_files_for_transfer/result.npz"
    num_outputs = length(problem.network.layers[end].bias)

    A, b = tosimplehrep(problem.input)
    weight_vector = LinearObjectiveToWeightVector(problem.objective, num_outputs)

    npzwrite(data_file, Dict("A" => A, "b" => b, "weight_vector" => weight_vector))
    write_nnet(network_file, problem.network)

    # Call MarabouPy.py with the path to the needed files
    command = `python ./exact/MarabouOptimizationPy.py  $data_file $network_file $result_file $timeout`
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

    return Result(status, get(result, :input), get(result, :objective_value)[1])
end
