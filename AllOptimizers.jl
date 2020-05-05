ENV["JULIA_DEBUG"] = Main # turns on logging (@debug, @info, @warn) for "included" files
using Gurobi
using GLPK
using TimerOutputs
using NPZ
import Pkg; Pkg.add("Colors")
using Colors, Plots

# Read in an example network
include("./NeuralOptimization.jl")

# Network and input files
nnet_file = "./Networks/AutoTaxi/AutoTaxi_32Relus_200Epochs_OneOutput.nnet" #"./Networks/MNIST/mnist10x10.nnet"
network = NeuralOptimization.read_nnet(nnet_file)

example_input = "./Datasets/AutoTaxi/AutoTaxi_2323.npy" #"./Datasets/MNIST/MNISTlabel_0_index_0_.npy"
center_input = npzread(example_input) # Transpose for AutoTaxi

# Create the optimizers
LBFGS_optimizer = NeuralOptimization.LBFGS()
FGSM_optimizer = NeuralOptimization.FGSM()
VanillaMIP_optimizer = NeuralOptimization.VanillaMIP(optimizer=Gurobi.Optimizer, m=1e3, time_limit=10)
Sherlock_optimizer = NeuralOptimization.Sherlock(optimizer=Gurobi.Optimizer)
#optimizers = [LBFGS_optimizer, FGSM_optimizer, VanillaMIP_optimizer, Sherlock_optimizer]
optimizers = [FGSM_optimizer]

# Create the problem: network, input constraints, output constraints,
# objective function, max vs. min, and the problem type (:linear_objective or :min_perturbation_linf)
num_inputs = size(network.layers[1].weights, 2)

input = NeuralOptimization.Hyperrectangle(vec(center_input)[:], 0.001 * ones(num_inputs)) # center and radius
objective = NeuralOptimization.LinearObjective([1.0], [1]) # objective is to just maximize the first output
max = true

problem = NeuralOptimization.OutputOptimizationProblem(network, input, objective, max)

# Run each optimizer and save results
results = []
times = []
for optimizer in optimizers
    time = @elapsed result = NeuralOptimization.optimize(optimizer, problem)
    push!(results, result)
    push!(times, time)
end
# Print out the results
for (optimizer, result, time) in zip(optimizers, results, times)
    println("Result for: ", optimizer)
    println("   Status: ", result.status)
    println("   Objective: ", result.objective_value)
    println("   Time: ", time)
end
