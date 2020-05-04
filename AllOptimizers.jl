ENV["JULIA_DEBUG"] = Main # turns on logging (@debug, @info, @warn) for "included" files

# Read in an example network
include("./NeuralOptimization.jl")
using NeuralVerification

nnet_file = "./Networks/AutoTaxi/AutoTaxi_32Relus_200Epochs_OneOutput.nnet"

LBFGS_optimizer = NeuralOptimization.LBFGS()
VanillaMIP_optimizer = NeuralOptimization.VanillaMIP(time_limit=10)


# Create the problem: network, input constraints, output constraints,
# objective function, max vs. min, and the problem type (:linear_objective or :min_perturbation_linf)
network = NeuralOptimization.read_nnet(nnet_file)
num_inputs = size(network.layers[1].weights, 2)

input = NeuralOptimization.Hyperrectangle(low=0.4 * ones(num_inputs), high=0.6 * ones(num_inputs))
objective = NeuralOptimization.LinearObjective([1.0], [1]) # objective is to just maximize the output
max = true

problem = NeuralOptimization.OutputOptimizationProblem(network, input, objective, max)

# Run each optimizer
result_LBFGS = NeuralOptimization.optimize(LBFGS_optimizer, problem)
result_VanillaMIP = NeuralOptimization.optimize(VanillaMIP_optimizer, problem)

# Print results
println("LBFGS Status: ", result_LBFGS.status, " Optimal Value: ", result_LBFGS.objective_value)
println("Optimal val from input: ", NeuralOptimization.compute_objective(network, result_LBFGS.input, objective))

println("Vanilla MIP Status: ", result_VanillaMIP.status, " Optimal Value: ", result_VanillaMIP.objective_value)
println("Optimal val from input: ", NeuralOptimization.compute_objective(network, result_VanillaMIP.input, objective))
