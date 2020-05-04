ENV["JULIA_DEBUG"] = Main # turns on logging (@debug, @info, @warn) for "included" files
using Gurobi
using GLPK
using TimerOutputs
using NPZ
import Pkg; Pkg.add("Colors")
using Colors, Plots

# Read in an example network
include("./NeuralOptimization.jl")
using NeuralVerification

# Network and input files
nnet_file = "./Networks/AutoTaxi/AutoTaxi_128Relus_200Epochs_OneOutput.nnet"
example_input = "./Datasets/AutoTaxi/AutoTaxi_2323.npy"
center_input = transpose(npzread(example_input))

# Create the optimizers
LBFGS_optimizer = NeuralOptimization.LBFGS()
VanillaMIP_optimizer = NeuralOptimization.VanillaMIP(optimizer=Gurobi.Optimizer, m=1e3, time_limit=10)
Sherlock_optimizer = NeuralOptimization.Sherlock(optimizer=Gurobi.Optimizer)

# Create the problem: network, input constraints, output constraints,
# objective function, max vs. min, and the problem type (:linear_objective or :min_perturbation_linf)
network = NeuralOptimization.read_nnet(nnet_file)
num_inputs = size(network.layers[1].weights, 2)

input = NeuralOptimization.Hyperrectangle(vec(center_input)[:], 0.05 * ones(num_inputs)) # center and radius
objective = NeuralOptimization.LinearObjective([1.0], [1]) # objective is to just maximize the output
max = true

problem = NeuralOptimization.OutputOptimizationProblem(network, input, objective, max)

# Run each optimizer
LBFGS_time = @elapsed result_LBFGS = NeuralOptimization.optimize(LBFGS_optimizer, problem)
VanillaMIP_time = @elapsed result_VanillaMIP = NeuralOptimization.optimize(VanillaMIP_optimizer, problem)
Sherlock_time = @elapsed result_Sherlock = NeuralOptimization.optimize(Sherlock_optimizer, problem)


# Print results
println("LBFGS time: ", LBFGS_time, ", Status: ", result_LBFGS.status, " Optimal Value: ", result_LBFGS.objective_value)
println("Optimal val from input: ", NeuralOptimization.compute_objective(network, result_LBFGS.input, objective))

println("VanillaMIP time: ", VanillaMIP_time, ", Status: ", result_VanillaMIP.status, " Optimal Value: ", result_VanillaMIP.objective_value)
println("Optimal val from input: ", NeuralOptimization.compute_objective(network, result_VanillaMIP.input, objective))

println("Sherlock time: ", Sherlock_time, " Status: ", result_Sherlock.status, ", Optimal Value: ", result_Sherlock.objective_value)
println("Optimal val from input: ", NeuralOptimization.compute_objective(network, result_Sherlock.input, objective))
