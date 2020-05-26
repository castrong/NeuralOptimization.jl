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
	@time init_marabou_binary_function()

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

function init_marabou_binary_function()
	py"""
	def setup_network(network_file, A, b, use_sbt, lower, upper):
		network = Marabou.read_nnet(network_file, use_sbt)
		inputVars = network.inputVars.flatten()
		numInputs = len(inputVars)

		# Set upper and lower on all input variables
		for var in network.inputVars.flatten():
		    network.setLowerBound(var, lower)
		    network.setUpperBound(var, upper)

		# # Add input constraints
		for row_index in range(A.shape[0]):
			input_constraint_equation = MarabouUtils.Equation(EquationType=MarabouCore.Equation.LE)
			input_constraint_equation.setScalar(b[row_index])

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
					network.setUpperBound(inputVars[index], min(b[row_index], upper))
				else:
					network.setLowerBound(inputVars[index], max(-b[row_index], lower))
			# If not, then this row corresponds to some other linear equation - so we'll encode that
			else:
				print("Constraint not a lower / upper bound")
				for col_index in range(A.shape[1]):
					if (A[row_index, col_index] != 0):
						print('Wasnt zero', (row_index, col_index))
						print('val:', (A[row_index, col_index]))
						input_constraint_equation.addAddend(A[row_index, col_index], inputVars[col_index])

				network.addEquation(input_constraint_equation)
				print("Adding equation with participating vars", input_constraint_equation.getParticipatingVariables())

		return network, inputVars, numInputs

	def marabou_binarysearch_python(A, b, lower_bound, network_file, use_sbt, divide_strategy, lower, upper, timeout):
		# Load in the network
		network, inputVars, numInputs = setup_network(network_file, A, b, use_sbt, lower, upper)

		# Setup options
		options = MarabouCore.Options()
		options._optimize = False
		options._verbosity = 1
		options._timeoutInSeconds = timeout
		options._dnc = False
		# Parse the divide strategy from a string to its corresponding enum
		if (divide_strategy == "EarliestReLU"):
			options._divideStrategy = MarabouCore.DivideStrategy.EarliestReLU
		elif (divide_strategy == "ReLUViolation"):
			options._divideStrategy = MarabouCore.DivideStrategy.ReLUViolation

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
	"""
end

function Base.show(io::IO, solver::MarabouBinarySearch)
	print(io, string("MarabouBinarySearch_", "sbt=", string(solver.usesbt), "_", "dividestrategy=", solver.dividestrategy))
end
