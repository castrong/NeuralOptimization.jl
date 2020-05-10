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
nnet_file = "./Networks/MNIST/mnist10x10.nnet"
network = NeuralOptimization.read_nnet(nnet_file)

example_input = "./Datasets/MNIST/MNISTlabel_0_index_0_.npy"
center_input = transpose(npzread(example_input)) # Transpose for AutoTaxi
#plot(Gray.(reshape(center_input, 28, 28)))
# Visualize: plot(Gray.(reshape(center_input, __, __)))
input_radius = 0.0003
time_limit = 900

# Create the optimizers

# Approximate
LBFGS_optimizer = NeuralOptimization.LBFGS()
FGSM_optimizer = NeuralOptimization.FGSM()

# NSVerify and Sherlock
VanillaMIP_Gurobi_optimizer = NeuralOptimization.VanillaMIP(optimizer=Gurobi.Optimizer, m=1e3, time_limit=time_limit)
VanillaMIP_GLPK_optimizer = NeuralOptimization.VanillaMIP(optimizer=GLPK.Optimizer, m=1e3, time_limit=time_limit)
Sherlock_Gurobi_optimizer = NeuralOptimization.Sherlock(optimizer=Gurobi.Optimizer, time_limit=time_limit)
Sherlock_GLPK_optimizer = NeuralOptimization.Sherlock(optimizer=GLPK.Optimizer, time_limit=time_limit)

# Marabou
Marabou_optimizer = NeuralOptimization.Marabou()
optimizers = [Marabou_optimizer]

# Create the problem: network, input constraints, output constraints, max vs. min
num_inputs = size(network.layers[1].weights, 2)

input = NeuralOptimization.Hyperrectangle(vec(center_input)[:], input_radius * ones(num_inputs)) # center and radius
objective = NeuralOptimization.LinearObjective([1.0], [3]) # objective is to just maximize the first output
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
