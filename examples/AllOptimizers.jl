using Pkg
#Pkg.activate("/Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl")


ENV["JULIA_DEBUG"] = Main # turns on logging (@debug, @info, @warn) for "included" files
using Gurobi
using GLPK
using TimerOutputs
using NPZ
using JuMP
import Pkg; Pkg.add("Colors")
#using Colors, Plots
using StatProfilerHTML

# Read in an example network
include("../src/NeuralOptimization.jl")
include("../src/utils/CreateAllOptimizers.jl")

# Network and input files
nnet_file = "./Networks/MNIST/mnist10x20.nnet"
network = NeuralOptimization.read_nnet(nnet_file)

input_file = "./Datasets/MNIST/MNISTlabel_0_index_0_.npy"
center_input = npzread(input_file)

# Visualize: plot(Gray.(reshape(center_input, __, __)))
input_radius = 0.001
time_limit = 300

# List all your optimizers you'd like to run
optimizers = [Marabou_optimizer_sbt_earliestReLU, LBFGS_optimizer]
# optimizers = [
             # LBFGS_optimizer,
             # PGD_optimizer,
             # FGSM_optimizer,
             # Marabou_optimizer_ReLUViolation,
             # Marabou_optimizer_sbt_ReLUViolation,
             # Marabou_optimizer_earliestReLU,
             # Marabou_optimizer_sbt_earliestReLU,
             # MarabouBinary_optimizer_ReLUViolation,
             # MarabouBinary_optimizer_sbt_ReLUViolation,
             # MarabouBinary_optimizer_earliestReLU,
             # MarabouBinary_optimizer_sbt_earliestReLU,
             # VanillaMIP_Gurobi_optimizer_8threads,
             # VanillaMIP_Gurobi_optimizer_1thread,
             # VanillaMIP_GLPK_optimizer,
             # Sherlock_Gurobi_optimizer_8threads,
             # Sherlock_Gurobi_optimizer_1thread,
             # Sherlock_GLPK_optimizer,
             #]

start_time = time()
println("Making simple Gurobi model")
# m = JuMP.Model(with_optimizer(Gurobi.Optimizer))
# @variable(m, x)
# @constraint(m, x >= 5)
# @constraint(m , x <= 10)
# @objective(m, Max, x)
# optimize!(m)
println("Finished making simple Gurobi model: ", time() - start_time)

# Make simple problem
simple_nnet = NeuralOptimization.read_nnet("./Networks/small_nnet.nnet")
simple_objective = NeuralOptimization.LinearObjective([1.0], [1])
simple_input = NeuralOptimization.Hyperrectangle([1.0], [1.0])
simple_problem = NeuralOptimization.OutputOptimizationProblem(network=simple_nnet, input=simple_input, objective=simple_objective, max=true, lower=-Inf, upper=Inf)
time_temp = @elapsed result = NeuralOptimization.optimize(optimizers[1], simple_problem, 20)
println("Finished simple optimization, took: ", time_temp)


# Run each optimizer and save results
results = []
times = []
center_input = npzread(input_file)
# Create the problem: network, input constraints, output constraints, max vs. min
num_inputs = size(network.layers[1].weights, 2)
input = NeuralOptimization.Hyperrectangle(vec(center_input)[:], input_radius * ones(num_inputs)) # center and radius
objective = NeuralOptimization.LinearObjective([1.0], [1]) # objective is to just maximize the first output
maximize = true
problem = NeuralOptimization.OutputOptimizationProblem(network=network, input=input, objective=objective, max=maximize, lower=0.0, upper=1.0)

for optimizer in optimizers
    #NeuralOptimization.optimize(optimizer, problem, 1)
    time = @elapsed result = NeuralOptimization.optimize(optimizer, problem, time_limit)
    push!(results, result)
    push!(times, time)
end

# Print out the results
for (result, time, optimizer) in zip(results, times, optimizers)
    println("Result for: ", optimizer)
    println("   Status: ", result.status)
    println("   Objective: ", result.objective_value)
    println("   Time: ", time)
end

println("Minimum: ", minimum(results[1].input))
println("Maximum: ", maximum(results[1].input))
