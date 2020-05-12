import time
import sys
import numpy as np
import copy
from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy import MarabouUtils

'''
A script for running a MarabouOptimizer query in line with the inputs expected
for NeuralOptimization.jl. So we'll get passed (i) file with A, b, c, and an initial input we know is feasible. The input constraint will be
Ax <= b, , where x is the input to the network and y is the output from the network. 
The network will already be augmented such that there is a single output - c will just correspond to 
whether we are maximizing or minimizing that input. 
'''
use_sbt = False

data_file = sys.argv[1]
network_file = sys.argv[2]
result_file = sys.argv[3]
timeout = int(sys.argv[4])

data = np.load(data_file)
A_rows = data['A_rows']
A_cols = data['A_cols']
A_values = data['A_values']

b = data['b']
lower_bound = float(data['feasible_value'])

# Load in the network
network = Marabou.read_nnet(network_file, use_sbt)
inputVars = network.inputVars.flatten()
numInputs = len(inputVars)


# Fill in from the sparse representation TODO: This may be an inefficient way to do this
A = np.zeros((len(b), numInputs))
A[A_rows, A_cols] = A_values

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
			network.setUpperBound(inputVars[index], b[row_index])
		else:
			network.setLowerBound(inputVars[index], -b[row_index])

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


# Setup options
options = MarabouCore.Options()
options._optimize = False
options._verbosity = 0
options._timeoutInSeconds = timeout
options._dnc = False
epsilon = 1e-4

start_time = time.time()


# Loop until we are confident in our result within epsilon
max_float = np.finfo(np.float32).max
upper_bound = max_float
best_input = []
status = ""

while (upper_bound - lower_bound) >= epsilon:
	new_network = copy.deepcopy(network)
	output_var = new_network.outputVars.flatten()[0]
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
	
	print("!!!!!!!next val: ", next_val)
	new_network.setLowerBound(output_var, next_val)

	# update your timeout
	time_remaining = timeout - int(time.time() - start_time)
	print("Time remaining: ", time_remaining)
	if (time_remaining <= 0):
		status = "timeout"
		break

	options._timeoutInSeconds = time_remaining
	vals, state = new_network.solve(filename="", options=options)

	if (state.hasTimedOut()):
		status = "timeout"
		break

	# UNSAT if it couldn't find a counter example
	if (len(vals) == 0):
		upper_bound = next_val
		print("Updating upper to: ", upper_bound)

	else:
		lower_bound = next_val
		# return the most recent vals that were achievable
		best_input = vals
		print("Updating lower to: ", lower_bound)

	print("lower, upper after this: ", (lower_bound, upper_bound))

# Set status
if status != "timeout":
	status = "success"

#input_val = [best_input[i] for i in range(0, numInputs)]
# If we found an input, then evaluate it and return that as the best objective
best_objective = -1
best_input_list = [best_input[i] for i in range(0, numInputs)]

if len(best_input) > 0:
	network = Marabou.read_nnet(network_file, use_sbt)
	best_objective = network.evaluateWithMarabou([best_input_list])[0]
else:
	best_objective = lower_bound



# Save the output to a file
np.savez(result_file, status=status, input=best_input_list, objective_value=best_objective)







