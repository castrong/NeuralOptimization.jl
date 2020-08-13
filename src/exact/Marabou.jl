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
    usesbt::Bool = false
	dividestrategy::String = "ReLUViolation"
	triangle_relaxation::Bool = false
end

function optimize(solver::Marabou, problem::OutputOptimizationProblem, time_limit::Int = 30)
	@debug string("Optimizing with: ", solver)
    @assert problem.input isa Hyperrectangle or problem.input isa HPolytope
	init_marabou_function()

    # If our last layer is ID we can replace the last layer and combine
    # it with the objective - shouldn't change performance but is
    # consistent with the network structure for Sherlock
    # if (problem.network.layers[end].activation == Id())
	# 	println("in marabou incorporating into single output layer")
    #     @debug "In Marabou incorporating into single output layer"
    #     augmented_network = extend_network_with_objective(problem.network, problem.objective) # If the last layer is ID it won't add a layer
	# 	augmented_objective = LinearObjective([1.0], [1])
    #     augmented_problem = OutputOptimizationProblem(augmented_network, problem.input, augmented_objective, problem.max, problem.lower, problem.upper, problem.lower_bounds, problem.upper_bounds)
    # else
        augmented_problem = problem
    # end

    A, b = tosimplehrep(augmented_problem.input)
	weight_vector = linear_objective_to_weight_vector(augmented_problem.objective, length(augmented_problem.network.layers[end].bias))
	# account for maximization vs minimization
	if (!augmented_problem.max)
        weight_vector = -1.0 * weight_vector
    end

	# Write the network then run the solver
	network_file = string(tempname(), ".nnet")
	println(network_file)
	write_nnet(network_file, augmented_problem.network)
	(status, input_val, obj_val) = py"""marabou_python"""(A, b, weight_vector, network_file, solver.usesbt, solver.dividestrategy, augmented_problem.lower, augmented_problem.upper, augmented_problem.lower_bounds, augmented_problem.upper_bounds, solver.triangle_relaxation, time_limit)

	# Turn the string status into a symbol to return
    if (status == "success")
        status = :success
    elseif (status == "timeout")
        status = :timeout
    else
        status = :error
    end

	# Correct for the maximization vs. minimization
	if (!problem.max)
        obj_val = -1.0 * obj_val
    end
	return Result(status, input_val, obj_val)
end

function init_marabou_function()
	py"""
	def marabou_python(A, b, weight_vector, network_file, use_sbt, divide_strategy, lower, upper, lower_bounds, upper_bounds, triangle_relaxation, timeout):
		# Load in the network
		network = Marabou.read_nnet(network_file, normalize=False)
		# TODO: FIGURE OUT HOW TO TURN ON AND OFF SBT
		#network.use_nlr = use_sbt
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

		# Add bounds on each node from bound list
		if (len(lower_bounds) > 0):
			print("In here")
			for layer_index in range(len(lower_bounds)):
				print("Layer index: ", layer_index)
				cur_lower_bounds = lower_bounds[layer_index]
				cur_upper_bounds = upper_bounds[layer_index]
				print("Upper: ", cur_upper_bounds)
				for bound_index in range(len(cur_lower_bounds)):
					cur_upper_bound = cur_upper_bounds[bound_index]
					cur_lower_bound = cur_lower_bounds[bound_index]
					# Set an upper and lower bound on the backward facing variable
					# f = relu(b)
					print("Indices: ", (layer_index, bound_index))
					# layer index + 1 since we don't have bounds on the input layer
					# we also wouldn't have a backward facing variable there
					backwardsFacingVariable = network.nodeTo_b(layer_index+1, bound_index)
					print("Bound: ", (cur_lower_bound, cur_upper_bound))
					network.setLowerBound(backwardsFacingVariable, cur_lower_bound)
					network.setUpperBound(backwardsFacingVariable, cur_upper_bound)

					# We can't do a triangle relaxation on the output layer since it won't have a forward facing variable
					if (triangle_relaxation and layer_index < len(lower_bounds) - 1):
						print("Triangle relaxation on layer: ", layer_index + 1)
						# Apply the triangle relaxation to nodes that are still split phase
						if (cur_upper_bound <= 0):

							print("Node fixed inactive")
						elif (cur_lower_bound > 0):
							print("Node fixed active")
						else:
							forwardsFacingVariable = network.nodeTo_f(layer_index+1, bound_index)
							triangleEquation = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)
							slope = cur_upper_bound / (cur_upper_bound - cur_lower_bound)

							# y <= (x - l) u/(u-l) --> x u/(u-l) - y >= l u/(u-l)
							triangleEquation.addAddend(slope, backwardsFacingVariable)
							triangleEquation.addAddend(-1.0, forwardsFacingVariable)
							triangleEquation.setScalar(cur_lower_bound * slope)
							network.addEquation(triangleEquation)


		# Set the options
		options = MarabouCore.Options()
		options._optimize = True
		options._verbosity = 0
		options._timeoutInSeconds = timeout
		# Parse the divide strategy from a string to its corresponding enum
		if (divide_strategy == "EarliestReLU"):
			options._divideStrategy = MarabouCore.DivideStrategy.EarliestReLU
		elif (divide_strategy == "ReLUViolation"):
			options._divideStrategy = MarabouCore.DivideStrategy.ReLUViolation


		# Add the optimization equation of the form
		# previous_addends - cost_fcn_var = 0 --> previous_addends = cost_fcn_var
		outputVars = network.outputVars.flatten()
		optEquation = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)

		# Introduce your optimization variable
		optVariable = network.getNewVariable()
		optEquation.addAddend(-1.0, optVariable) # -1 so that it equals all the other terms
		optEquation.setScalar(0.0)

		# Add in your other terms
		for index, coefficient in enumerate(weight_vector):
			if (coefficient != 0):
				print("Index ", index, ": Adding coefficient of ", coefficient)
				optEquation.addAddend(coefficient, outputVars[index])

		# Add the equation to the network, and mark the optimization variable
		network.addEquation(optEquation)
		network.setOptimizationVariable(optVariable)

		# Run the solver
		start_solve_time = time.time()
		vals, state = network.solve(filename="", options=options)
		print("Finished solving")

		status = ""
		input_val = [-1.0]
		objective_value = -1.0
		if (state.hasTimedOut()):
			status = "timeout"
		else:
			status = "success"
			input_val = [vals[i] for i in range(0, numInputs)]

			# Have to read the network again because of deallocation issues after the first call
			network = Marabou.read_nnet(network_file, use_sbt)
			marabouOptimizerResult = network.evaluateWithMarabou([input_val])
			objective_value = 0
			for index, coefficient in enumerate(weight_vector):
				objective_value += coefficient * marabouOptimizerResult[0][index] # odd indexing
			print("Objective value: ", objective_value)

		return (status, input_val, objective_value)
	"""
end

function Base.show(io::IO, solver::Marabou)
	print(io, string("Marabou_", "sbt=", string(solver.usesbt), "_", "dividestrategy=", solver.dividestrategy))
end
