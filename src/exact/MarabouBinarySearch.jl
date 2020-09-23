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
    optimizer = GLPK.Optimizer # To find a feasible original point
    usesbt = false
	dividestrategy = "ReLUViolation"
end

function optimize(solver::MarabouBinarySearch, problem::OutputOptimizationProblem, time_limit::Int = 1200)
	@debug string("Optimizing with: ", solver)
    @assert problem.input isa Hyperrectangle or problem.input isa HPolytope
	init_marabou_binary_function()

    # Find a feasible solution to pass it a lower bound
    model = model_creator(solver)
    input_vars = @variable(model, [i=1:size(problem.network.layers[1].weights, 2)])
    add_set_constraint!(model, problem.input, input_vars, problem.lower, problem.upper)
    optimize!(model)
    @assert (value.(input_vars) ∈ problem.input)

    feasible_val = compute_objective(problem.network, value.(input_vars), problem.objective)
    if (!problem.max)
        feasible_val = feasible_val * -1.0
    end
	@debug "Feasible val is: " feasible_val

    # Augment the network to handle an arbitrary linear objective
    # if the last layer was ID() then this just combines the objective into that layer
    # TODO: Do we lose any efficiency if we expand without
    negative_objective = problem.max
    augmented_network = extend_network_with_objective(problem.network, problem.objective, !problem.max) # If the last layer is ID it won't add a layer

	A, b = tosimplehrep(problem.input)

	# Write the network then run the solver
    network_file = string(tempname(), ".nnet")
	write_nnet(network_file, augmented_network)
	(status, input_val, obj_val) = py"""marabou_binarysearch_python"""(A, b, feasible_val, network_file, solver.usesbt, solver.dividestrategy, problem.lower, problem.upper, time_limit)

	# Turn the string status into a symbol to return
    if (status == "success")
        status = :success
    elseif (status == "timeout")
        status = :timeout
    else
        status = :error
    end

	# Correct for the maximization
	if (!problem.max)
        obj_val = -1.0 * obj_val
    end

	return Result(status, input_val, obj_val)

    # Account for maximization vs minimization - we will have augmented the network
    # to get the maximum negative objective if we were minimizing so we need to flip it back
    obj_val = get(result, :objective_value)[1]
    if (!problem.max)
        obj_val = -1.0 * obj_val
    end

    return Result(status, get(result, :input), obj_val)
end

function optimize(solver::MarabouBinarySearch, problem::MinPerturbationProblem, time_limit::Int = 30)
	@assert problem.norm_order == Inf "Only Inf norm currently supported"
	init_python_functions()

	@assert problem.input isa Hyperrectangle "Only suppoorting hyperrectangle for now"

	# Input and output sets
	input_lower, input_upper = low(problem.input), high(problem.input)
	num_outputs = length(problem.network.layers[end].bias)
	# If it provides a target, convert that to an output set
	if (problem.target != nothing)
		sign = problem.target_dir == "max" ? 1.0 : -1.0
		# A matrix with each row corresponding to the target >= other index
		A_out = Matrix{Float64}(sign * I, num_outputs, num_outputs)
		A_out[:, problem.target] .= sign * -1.0
		A_out = A_out[1:end .!= problem.target, :] # remove the row which would try to do target >= target
		b_out = zeros(num_outputs - 1)
	else
		A_out, b_out = tosimplehrep(problem.output)
	end

	# Write the network then run the solver
	network_file = string(tempname(), ".nnet")
	write_nnet(network_file, problem.network)
	(status, input_val, obj_val) = py"""min_input_binary_search"""(problem.center, input_lower, input_upper, A_out, b_out, problem.dims .- 1, problem.norm_order, network_file, solver.usesbt, solver.dividestrategy, time_limit)
	return MinPerturbationResult(Symbol(status), input_val, obj_val)
end

function init_marabou_binary_function()
	py"""
	def encode_polytope(A, b, variables, network):
		# Add input constraints
		for row_index in range(A.shape[0]):
			constraint_equation = MarabouUtils.Equation(EquationType=MarabouCore.Equation.LE)
			constraint_equation.setScalar(b[row_index])
			# First check if this row corresponds to an upper or lower bound on a variable
			# it will be more efficient to store them in this way
			all_weights_mag_1 = True
			num_weights = 0
			index = 0
			mag_weight = 0

			for col_index in range(A.shape[1]):
				if A[row_index, col_index] != 0:
					index = col_index
					mag_weight = A[row_index, col_index]
					num_weights = num_weights + 1
					if (A[row_index, col_index] != 1 and A[row_index, col_index] != -1):
						all_weights_mag_1 = False

			if (all_weights_mag_1 and num_weights == 1):
				if (mag_weight == 1):
					if variables[index] in network.upperBounds.keys():
						network.setUpperBound(variables[index], min(b[row_index], network.upperBounds[variables[index]]))
					else:
						network.setUpperBound(variables[index], b[row_index])
				else:
					if variables[index] in network.lowerBounds.keys():
						network.setLowerBound(variables[index], max(-b[row_index], network.lowerBounds[variables[index]]))
					else:
						network.setLowerBound(variables[index], -b[row_index])
			# If not, then this row corresponds to some other linear equation - so we'll encode that
			else:
				for col_index in range(A.shape[1]):
					if (A[row_index, col_index] != 0):
						constraint_equation.addAddend(A[row_index, col_index], variables[col_index])

				network.addEquation(constraint_equation)
		return network

	def setup_network(network_file, A, b, use_sbt, lower, upper):
		network = Marabou.read_nnet(network_file, normalize=False)
		# TODO: FIGURE OUT HOW TO TURN ON AND OFF SBT
		#network.use_nlr = use_sbt
		inputVars = network.inputVars.flatten()
		numInputs = len(inputVars)

		# Set upper and lower on all input variables
		for var in network.inputVars.flatten():
		    network.setLowerBound(var, lower)
		    network.setUpperBound(var, upper)

		network = encode_polytope(A, b, inputVars, network)

		return network, inputVars, numInputs

	def setup_network_in_out(network_file, input_lower, input_upper, A_out, b_out):
		network = Marabou.read_nnet(network_file, normalize=False)
		# TODO: FIGURE OUT HOW TO TURN ON AND OFF SBT
		#network.use_nlr = use_sbt
		inputVars = network.inputVars.flatten()
		outputVars = network.outputVars.flatten()
		numInputs = len(inputVars)
		numOutputs = len(inputVars)

		# Impose a hyperrectangle on the input
		for (i, var) in enumerate(network.inputVars.flatten()):
		    network.setLowerBound(var, input_lower[i])
		    network.setUpperBound(var, input_upper[i])

		network = encode_polytope(A_out, b_out, outputVars, network)

		return network, inputVars, outputVars, numInputs, numOutputs

	def marabou_binarysearch_python(A, b, lower_bound, network_file, use_sbt, divide_strategy, lower, upper, timeout):
		# Load in the network
		network, inputVars, numInputs = setup_network(network_file, A, b, use_sbt, lower, upper)

		# Setup options
		options = MarabouCore.Options()
		options._optimize = False
		options._verbosity = 0
		options._timeoutInSeconds = timeout
		options._dnc = False
		# Parse the divide strategy from a string to its corresponding enum
		if (divide_strategy == "EarliestReLU"):
			options._divideStrategy = MarabouCore.DivideStrategy.EarliestReLU
		elif (divide_strategy == "ReLUViolation"):
			options._divideStrategy = MarabouCore.DivideStrategy.ReLUViolation
		elif (divide_strategy == "LargestInterval"):
			options._divideStrategy = MarabouCore.DivideStrategy.LargestInterval
		elif (divide_strategy == "Auto"):
			options._divideStrategy = MarabouCore.DivideStrategy.Auto

		start_time = time.time()

		# Loop until we are confident in our result within epsilon
		epsilon = 1e-4
		max_float = np.finfo(np.float32).max
		upper_bound = max_float
		best_input = []
		status = ""

		print("before loop")
		while (upper_bound - lower_bound) >= epsilon:
			print("start of loop")
			# Read in and set bounds every time
			output_var = network.outputVars.flatten()[0]
			next_val = 0
			# If still searching for an upper bound
			if (upper_bound == max_float):
				if (lower_bound < 0):
					next_val = -lower_bound
				elif (lower_bound == 0):
					next_val = epsilon
				else:
					next_val = lower_bound * 2
			# If we have a lower and upper, split the difference
			else:
				next_val = (lower_bound + upper_bound) / 2.0

			network.setLowerBound(output_var, next_val)

			# update your timeout
			time_remaining = timeout - int(time.time() - start_time)
			print("Time remaining: ", time_remaining)
			if (time_remaining <= 0):
				status = "timeout"
				break

			options._timeoutInSeconds = time_remaining
			vals, state = network.solve(filename="", options=options)

			if (state.hasTimedOut()):
				status = "timeout"
				break

			# UNSAT if it couldn't find a counter example
			if (len(vals) == 0):
				upper_bound = next_val
			else:
				lower_bound = next_val
				# return the most recent vals that were achievable
				best_input = vals

			print("lower, upper after this: ", (lower_bound, upper_bound))
			print("Optimality Gap: ", upper_bound - lower_bound)

		# Set status
		if status != "timeout":
			status = "success"

		#input_val = [best_input[i] for i in range(0, numInputs)]
		# If we found an input, then evaluate it and return that as the best objective
		best_objective = -1
		best_input_list = [-1]
		if (len(best_input)) > 0:
			best_input_list = [best_input[i] for i in range(0, numInputs)]
			network = Marabou.read_nnet(network_file, use_sbt)
			best_objective = network.evaluateWithMarabou([best_input_list])
			best_objective = best_objective.flatten()[0]
		else:
			best_objective = lower_bound

		return (status, best_input_list, best_objective) # flatten b/c network outputs as 1x1 array

	def min_input_binary_search(center, input_lower, input_upper, A_out, b_out, dims, norm_order, network_file, use_sbt, divide_strategy, timeout):
		network = Marabou.read_nnet(network_file, normalize=False)
		# TODO: FIGURE OUT HOW TO TURN ON AND OFF SBT
		#network.use_nlr = use_sbt
		network, input_vars, output_vars, num_inputs, num_outputs = setup_network_in_out(network_file, input_lower, input_upper, A_out, b_out)
		# Setup options
		options = MarabouCore.Options()
		options._optimize = False
		options._verbosity = 0
		options._timeoutInSeconds = timeout
		options._dnc = False
		# Parse the divide strategy from a string to its corresponding enum
		if (divide_strategy == "EarliestReLU"):
			options._divideStrategy = MarabouCore.DivideStrategy.EarliestReLU
		elif (divide_strategy == "ReLUViolation"):
			options._divideStrategy = MarabouCore.DivideStrategy.ReLUViolation
		elif (divide_strategy == "LargestInterval"):
			options._divideStrategy = MarabouCore.DivideStrategy.LargestInterval
		elif (divide_strategy == "Auto"):
			options._divideStrategy = MarabouCore.DivideStrategy.Auto

		start_time = time.time()

		# Find the upper bound on epsilon that we start with
		max_interval = 0.0
		for dim in dims:
			in_var = input_vars[dim]
			lower = network.lowerBounds[in_var]
			upper = network.upperBounds[in_var]
			max_interval = max(max_interval, upper - lower)

		tolerance = 1e-4
		lower_bound = 0.0
		upper_bound = max_interval / 2.0
		best_input = []
		status = ""
		# Loop until we are confident in our result within epsilon
		while (upper_bound - lower_bound) >= tolerance:
			# Check whether the area with radius
			# less than (upper_bound + lower_bound) / 2
			# has an adversarial example

			# Find the midpoint and create new upper and lower bounds based on it
			radius_to_test = (lower_bound + upper_bound) / 2.0
			print("testing radius: ", radius_to_test)
			new_input_lower = input_lower.copy()
			new_input_upper = input_upper.copy()
			for dim in dims:
				# Update based on radius, make sure it doesn't leave the original bounds
				# (this will make sure it respects the network's domain)
				new_input_lower[dim] = max(center[dim] - radius_to_test, input_lower[dim])
				new_input_upper[dim] = min(center[dim] + radius_to_test, input_upper[dim])
			# re-encode the network with these new input constraints (maybe a bit of overhead to re-encode fully)
			network, input_vars, output_vars, num_inputs, num_outputs = setup_network_in_out(network_file, new_input_lower, new_input_upper, A_out, b_out)

			# update your timeout
			time_remaining = timeout - int(time.time() - start_time)
			print("Time remaining: ", time_remaining)
			if (time_remaining <= 0):
				status = "timeout"
				break

			options._timeoutInSeconds = time_remaining
			vals, state = network.solve(filename="", options=options)
			if (state.hasTimedOut()):
				status = "timeout"
				break

			# UNSAT if it couldn't find a counter example
			if (len(vals) == 0):
				lower_bound = radius_to_test
			else:
				best_input = vals
				# Find the objective value with this satisfying example
				# and use that to set the upper bound instead of the radius to test
				obj_deltas = [abs(best_input[i] - center[i]) for i in dims]
				objective_value = max(obj_deltas)
				upper_bound = objective_value
				print("objective value: ", objective_value)
				print("Difference from taking radius: ", radius_to_test - objective_value)

			print("lower, upper after this: ", (lower_bound, upper_bound))
			print("Optimality Gap: ", upper_bound - lower_bound)
		# Set status
		final_input = [float("inf")]
		objective_value = float("inf")
		if status != "timeout":
			if len(best_input) == 0:
				status = "infeasible"
			else:
				status = "success"
				final_input = [best_input[i] for i in range(0, num_inputs)]
				deltas = [abs(final_input[i] - center[i]) for i in range(len(final_input))]
				obj_deltas = [abs(final_input[i] - center[i]) for i in dims]
				objective_value = max(obj_deltas)
				# Have to read the network again because of deallocation issues after the first call
				network = Marabou.read_nnet(network_file, normalize=False)
				marabou_optimizer_result = network.evaluateWithMarabou([final_input])
				print("Deltas from nominal: ", deltas)
				print("Objective Deltas: ", obj_deltas)
				print("Optimal output: ", marabou_optimizer_result)
				print("Optimal Objective: ", objective_value)

		return (status, final_input, objective_value)
	"""
end

function Base.show(io::IO, solver::MarabouBinarySearch)
	print(io, string("MarabouBinarySearch_", "sbt=", string(solver.usesbt), "_", "dividestrategy=", solver.dividestrategy))
end
