import time
start_script_time = time.time()

import sys
import numpy as np
import copy
from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy import MarabouUtils
end_import_time = time.time()


'''
A script for running a MarabouOptimizer query in line with the inputs expected
for NeuralOptimization.jl. So we'll get passed (i) file with A, b, c. The input constraint will be
Ax <= b, and the objective will be transpose(c) * y, where x is the input to the network and y is the
output from the network.
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
weight_vector = data['weight_vector']

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

# Set the options
options = MarabouCore.Options()
options._optimize = True
options._verbosity = 0
options._timeoutInSeconds = timeout


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

# Save the output to a file
np.savez(result_file, status=status, input=input_val, objective_value=objective_value)


print("Time to finish import: ", end_import_time - start_script_time)
print("Time to get to actually solving: ", start_solve_time - start_script_time)


