"""
    LBFGS()

This uses Optim's LBFGS optimizer to approximately solve the problem

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: Hyperrectangle
3. Objective: Any linear objective

# Property
Approximate

"""
@with_kw struct LBFGS
    dummy_var = 1
end

function optimize(solver::LBFGS, problem::Problem, time_limit::Int = 1200)
    @debug string("Optimizing with: ", solver)
    # Assert the problem takes the necessary format
    # restriction from Optim, where we're getting our implementation
    # of LBFGS, focusing on box constrained problems
    @assert problem.input isa Hyperrectangle
    num_inputs = size(problem.network.layers[1].weights, 2)

    # Augment the network to handle an arbitrary linear objective
    # if the last layer was ID() then this just combines the objective into that layer
    augmented_network = extend_network_with_objective(problem.network, problem.objective) # If the last layer is ID it won't add a layer
    augmented_objective = LinearObjective([1.0], [1])
    augmented_problem = OutputOptimizationProblem(augmented_network, problem.input, augmented_objective, problem.max, problem.lower, problem.upper)

    # Since we're now guaranteed to have a single output,
    obj_variable = 1
    obj_coefficient = 1.0

    nnet = augmented_problem.network;

    function obj_function(x)
        compute_output(augmented_problem.network)
    end

    input_set = augmented_problem.input
    input_lower = max.(input_set.center - input_set.radius, augmented_problem.lower * ones(num_inputs));
    input_upper = min.(input_set.center + input_set.radius, augmented_problem.upper * ones(num_inputs));
    x_0 = (input_lower + input_upper) / 2.0

    # Maximize
    if (augmented_problem.max)
        # f, gradient, lower bound input, upper bound input, initial guess
        # use negative value and gradient because Optim will minimize it
        result = Optim.optimize(x->-obj_coefficient * compute_output(nnet, x)[obj_variable],
                                x->-obj_coefficient * get_gradient(nnet, x)[obj_variable, :],
                                input_lower, input_upper, x_0,
                                Optim.Fminbox(Optim.LBFGS()),
                                Optim.Options(time_limit = time_limit); inplace=false)
        opt_val = -minimum(result)

    # Minimize
    else
        result = Optim.optimize(x->obj_coefficient * compute_output(nnet, x)[obj_variable],
                                x->obj_coefficient * get_gradient(nnet, x)[obj_variable, :],
                                input_lower, input_upper, x_0,
                                Optim.Fminbox(Optim.LBFGS()),
                                Optim.Options(time_limit = time_limit); inplace=false)
        opt_val = minimum(result)
    end
    @debug "Result from LBFGS: " result

    return Result(:success, result.minimizer, opt_val)
end

function Base.show(io::IO, solver::LBFGS)
  print(io, "LBFGS")
end
