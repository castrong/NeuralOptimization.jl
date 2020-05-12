import sys
import numpy as np
import copy
from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy import MarabouUtils
import time

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
A = data['A']
b = data['b']
x_0 = data['x_0']

weight_vector = data['weight_vector']

# Load in the network
network = Marabou.read_nnet(network_file, use_sbt)
inputVars = network.inputVars.flatten()
numInputs = len(inputVars)


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


network_copy = copy.deepcopy(network)
# Loop until we are confident in our result within epsilon
vals, state = new_network.solve(filename="", options=options)



