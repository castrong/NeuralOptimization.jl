ENV["JULIA_DEBUG"] = Main # turns on logging (@debug, @info, @warn) for "included" files

# Read in an example network
include("./NeuralOptimization.jl")
using NeuralVerification

nnet_file = "./Networks/AutoTaxi/AutoTaxi_32Relus_200Epochs_OneOutput.nnet"

optimizer = NeuralOptimization.LBFGS()

# Create the problem: network, input constraints, output constraints,
# objective function, max vs. min, and the problem type (:linear_objective or :min_perturbation_linf)
network = NeuralOptimization.read_nnet(nnet_file)
num_inputs = size(network.layers[1].weights, 2)
input = NeuralOptimization.Hyperrectangle(low=0.2 * ones(num_inputs), high=0.8 * ones(num_inputs))
output = nothing

end_layer_index = length(network.layers) + 1 # add 1 to account for the input layer
objective = NeuralOptimization.LinearObjective([1.0], [(end_layer_index, 1)]) # objective is to just maximize the output
max = true
problem_type = :linear_objective

problem = NeuralOptimization.Problem(network, input, output, objective, max, problem_type)
result = NeuralOptimization.optimize(optimizer, problem)
println("Status: ", result.status, " Optimal Value: ", result.objective_value)

# Check that this input gives the optimal value
println("Optimal val from input: ", NeuralOptimization.compute_objective(network, result.input, objective))
