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
    ϵ = 1e-5
end

function optimize(solver::LBFGS, problem::OutputOptimizationProblem, time_limit::Int = 60000)
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
    augmented_problem = OutputOptimizationProblem(network=augmented_network, input=problem.input, objective=augmented_objective, max=problem.max, lower=problem.lower, upper=problem.upper)

    # Since we're now guaranteed to have a single output,
    obj_variable = 1
    obj_coefficient = 1.0

    nnet = augmented_problem.network;

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

function optimize(solver::LBFGS, problem::MinPerturbationProblem, time_limit::Int = 60000)
	num_outputs = length(problem.network.layers[end].bias)
	target = problem.target

	### Define functions that compute the objective and gradient given an
	### input x
	# c * norm(x - x_0) + (- log p(target))
	function obj_function(x, c)
		norm_val = norm((x - problem.center)[problem.dims], problem.norm_order)
		output = compute_output(problem.network, x)
		softmax_output = softmax(output)
		cross_entropy = -log(softmax_output[target])
		return c * norm_val + cross_entropy
	end
	function gradient_function(x, c)
		if (problem.norm_order == Inf)
			norm_grad = zeros(length(x))
			index_in_dims = argmax((abs.(x - problem.center))[problem.dims])
			norm_grad[problem.dims[index_in_dims]] = 1.0
		elseif(problem.objective == 1)
			norm_grad = sign.(x)
		else
			@assert false "Unsupported norm order for now"
		end
		output = compute_output(problem.network, x)
		softmax_output = softmax(output)

		network_grad = get_gradient(problem.network, x)
		dsoftmax_dout = zeros(num_outputs)
		for i = 1:num_outputs
			if i == target
				dsoftmax_dout[i] = softmax_output[target] * (1 - softmax_output[target])
			else
				dsoftmax_dout[i] = -softmax_output[target] * softmax_output[i]
			end
		end
		cross_entropy_grad = (1/softmax_output[target] * dsoftmax_dout' * network_grad)'
		return c * norm_grad + cross_entropy_grad
	end

	# Perform a binary search over c looking for the lowest value of c
	# that leads to an adversarial example
	x_0 = (low(problem.input) + high(problem.input)) / 2.0
	lower_bound_c = 0.0
	upper_bound_c = 10.0
	best_input = Vector{Float64}() # input associated with the lowest value of c
	found_any = false
	# We're looking for the smallest c (which weighs the perturbation)
	# which still leads to an adversarial example
	# upper_bound_c stores the lowest c that has gotten an adversarial example.
	# it upper bounds our optimal value of c.
	# lower_bound_c stores the highest value of c that has not gotten an adversarial example.
	# it lower bounds our optimal value of c.
	while (upper_bound_c - lower_bound_c >= solver.ϵ)
		c = (upper_bound_c + lower_bound_c) / 2.0
		println("Current c (LBFGS): ", c)
		result = Optim.optimize(x -> obj_function(x, c),
								x -> gradient_function(x, c),
								low(problem.input), high(problem.input), x_0,
								Optim.Fminbox(Optim.LBFGS()),
								Optim.Options(time_limit = time_limit); inplace=false)

		opt_input = Optim.minimizer(result) # current optimal input
		opt_output = compute_output(problem.network, opt_input)
		if all(opt_output[target] .>= opt_output)
			upper_bound_c = c
			found_any = true
			best_input = opt_input
		else
			# If you haven't found any yet, increase the upper bound so you try a higher value of c
			if !found_any
				upper_bound_c = upper_bound_c * 2
			else
				lower_bound_c = c
			end
		end
	end
	@assert (length(best_input) != 0) "Didn't find an adversarial example in LBFGS"
	return MinPerturbationResult(best_input, norm(best_input - problem.center, problem.norm_order))
end

function Base.show(io::IO, solver::LBFGS)
  print(io, "LBFGS")
end
