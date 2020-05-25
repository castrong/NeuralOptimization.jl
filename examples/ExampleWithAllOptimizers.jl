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
nnet_file = "./Networks/AutoTaxi/AutoTaxi_32Relus_200Epochs_OneOutput.nnet"
network = NeuralOptimization.read_nnet(nnet_file)

example_input = "./Datasets/AutoTaxi/AutoTaxi_12345.npy"
center_input = transpose(npzread(example_input)) # Transpose for AutoTaxi - transpose(npzread(example_input))
#plot(Gray.(reshape(center_input, 28, 28)))
# Visualize: plot(Gray.(reshape(center_input, __, __)))
input_radius = 0.0001
time_limit = 60

# Create the optimizers

# Approximate
LBFGS_optimizer = NeuralOptimization.LBFGS()
FGSM_optimizer = NeuralOptimization.FGSM()
PGD_optimizer = NeuralOptimization.PGD()


# NSVerify and Sherlock
VanillaMIP_Gurobi_optimizer_8threads = NeuralOptimization.VanillaMIP(optimizer=Gurobi.Optimizer, m=1e3, threads=8)
VanillaMIP_Gurobi_optimizer_1thread = NeuralOptimization.VanillaMIP(optimizer=Gurobi.Optimizer, m=1e3, threads=1)
VanillaMIP_GLPK_optimizer = NeuralOptimization.VanillaMIP(optimizer=GLPK.Optimizer, m=1e3)
Sherlock_Gurobi_optimizer_8threads = NeuralOptimization.Sherlock(optimizer=Gurobi.Optimizer, m=1e3, threads=8)
Sherlock_Gurobi_optimizer_1thread = NeuralOptimization.Sherlock(optimizer=Gurobi.Optimizer, m=1e3, threads=1)
Sherlock_GLPK_optimizer = NeuralOptimization.Sherlock(optimizer=GLPK.Optimizer)

# Marabou
Marabou_optimizer_ReLUViolation = NeuralOptimization.Marabou(use_sbt=false, divide_strategy = "ReLUViolation")
Marabou_optimizer_sbt_ReLUViolation = NeuralOptimization.Marabou(use_sbt=true, divide_strategy = "ReLUViolation")
Marabou_optimizer_earliestReLU = NeuralOptimization.Marabou(use_sbt=false, divide_strategy = "EarliestReLU")
Marabou_optimizer_sbt_earliestReLU = NeuralOptimization.Marabou(use_sbt=true, divide_strategy = "EarliestReLU")
MarabouBinary_optimizer = NeuralOptimization.MarabouBinarySearch(divide_strategy = "EarliestReLU")
MarabouBinary_optimizer_sbt = NeuralOptimization.MarabouBinarySearch(use_sbt=true, divide_strategy = "EarliestReLU")

# List all your optimizers you'd like to run
optimizers = [VanillaMIP_GLPK_optimizer]
# optimizers = [LBFGS_optimizer,
#               PGD_optimizer,
#               FGSM_optimizer,
               Marabou_optimizer_ReLUViolation,
#               Marabou_optimizer_sbt_ReLUViolation,
#               Marabou_optimizer_earliestReLU,
#               Marabou_optimizer_sbt_earliestReLU,
#               MarabouBinary_optimizer,
#               MarabouBinary_optimizer_sbt,
#               VanillaMIP_Gurobi_optimizer_8threads,
#               VanillaMIP_Gurobi_optimizer_1thread,
#               VanillaMIP_GLPK_optimizer,
#               Sherlock_Gurobi_optimizer_8threads,
#               Sherlock_Gurobi_optimizer_1thread,
#               Sherlock_GLPK_optimizer,
#               ]
# Create the problem: network, input constraints, output constraints, max vs. min
num_inputs = size(network.layers[1].weights, 2)

input = NeuralOptimization.Hyperrectangle(vec(center_input)[:], input_radius * ones(num_inputs)) # center and radius
objective = NeuralOptimization.LinearObjective([1.0], [1]) # objective is to just maximize the first output
maximize = true

problem = NeuralOptimization.OutputOptimizationProblem(network, input, objective, maximize, -Inf, Inf)

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
