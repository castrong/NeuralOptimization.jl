ENV["JULIA_DEBUG"] = Main # turns on logging (@debug, @info, @warn) for "included" files
using Gurobi
using GLPK
using TimerOutputs
using NPZ
import Pkg; Pkg.add("Colors")
using Colors, Plots


# Read in an example network
include("../src/NeuralOptimization.jl")

# Network and input files
nnet_file = "./Networks/MNIST/mnist10x10.nnet"
network = NeuralOptimization.read_nnet(nnet_file)
example_input = "./Datasets/MNIST/MNISTlabel_0_index_0_.npy"
center_input = npzread(example_input) # Transpose for AutoTaxi - transpose(npzread(example_input))
#plot(Gray.(reshape(center_input, 28, 28)))
# Visualize: plot(Gray.(reshape(center_input, __, __)))
input_radius = 0.0001
time_limit = 10

# Create the optimizers

# Approximate
LBFGS_optimizer = NeuralOptimization.LBFGS()
FGSM_optimizer = NeuralOptimization.FGSM()
PGD_optimizer = NeuralOptimization.PGD()


# NSVerify and Sherlock
VanillaMIP_Gurobi_optimizer = NeuralOptimization.VanillaMIP(optimizer=Gurobi.Optimizer, m=1e3)
VanillaMIP_GLPK_optimizer = NeuralOptimization.VanillaMIP(optimizer=GLPK.Optimizer, m=1e3)
Sherlock_Gurobi_optimizer = NeuralOptimization.Sherlock(optimizer=Gurobi.Optimizer)
Sherlock_GLPK_optimizer = NeuralOptimization.Sherlock(optimizer=GLPK.Optimizer)

# Marabou
Marabou_optimizer = NeuralOptimization.Marabou()

# List all your optimizers you'd like to run
optimizers = [LBFGS_optimizer, PGD_optimizer, FGSM_optimizer, VanillaMIP_Gurobi_optimizer, VanillaMIP_GLPK_optimizer, Sherlock_Gurobi_optimizer, Sherlock_GLPK_optimizer, Marabou_optimizer]

# Create the problem: network, input constraints, output constraints, max vs. min
num_inputs = size(network.layers[1].weights, 2)

input = NeuralOptimization.Hyperrectangle(vec(center_input)[:], input_radius * ones(num_inputs)) # center and radius
objective = NeuralOptimization.LinearObjective([1.0], [1]) # objective is to just maximize the first output
max = true

problem = NeuralOptimization.OutputOptimizationProblem(network, input, objective, max)

# Run each optimizer and save results
results = []
times = []
for optimizer in optimizers
    time = @elapsed result = NeuralOptimization.optimize(optimizer, problem, time_limit)
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
