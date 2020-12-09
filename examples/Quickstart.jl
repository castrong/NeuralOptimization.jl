ENV["JULIA_DEBUG"] = Main # turns on logging (@debug, @info, @warn) for "included" files
include("@__DIR__/../../src/NeuralOptimization.jl")
using NPZ

# Read in your nnet 
nnet_file = "./Networks/AutoTaxi/AutoTaxi_32Relus_200Epochs_OneOutput.nnet"
network = NeuralOptimization.read_nnet(nnet_file)

# Create your input region. In this case we create a Hyperrectangle
# with radius 0.02. The algorithm will handle limiting the input domain
# when the problem is created, so we don't need to cutoff the input Hyperrectangle
# between 0 and 1. 
center_input = npzread("./Datasets/AutoTaxi/AutoTaxi_2323_transposed.npy")[:]
radius = 0.02
input = NeuralOptimization.Hyperrectangle(low=center_input .- radius, high=center_input .+ radius)

# Create the objective. The first argument are the coefficients
# and the second argument are the indices of the output these coefficients 
# correspond to.
objective = NeuralOptimization.LinearObjective([1.0], [1]) # objective is to just maximize the output
max = true # maximize the objective 

# Create the problem: network, input constraints, objective,
# max vs. min, lower bound on the inputs, upper bound on the inputs
problem = NeuralOptimization.OutputOptimizationProblem(network, input, objective, max, 0.0, 1.0, [[]], [[]])

# Solve with 4 different optimizers
lbfgs = NeuralOptimization.LBFGS()
pgd = NeuralOptimization.PGD()
fgsm = NeuralOptimization.FGSM()
nsverify = NeuralOptimization.VanillaMIP()

result_lbfgs = NeuralOptimization.optimize(lbfgs, problem)
result_pgd = NeuralOptimization.optimize(pgd, problem)
result_fgsm = NeuralOptimization.optimize(fgsm, problem)
result_nsverify = NeuralOptimization.optimize(nsverify, problem)
println("LBFG: Status: ", result_lbfgs.status, " Optimal Value: ", result_lbfgs.objective_value)
println("PGD: Status: ", result_pgd.status, " Optimal Value: ", result_pgd.objective_value)
println("FGSM: Status: ", result_fgsm.status, " Optimal Value: ", result_fgsm.objective_value)
println("NSVerify: Status: ", result_nsverify.status, " Optimal Value: ", result_nsverify.objective_value)

# Check an input 
println("Optimal val from nsverify input: ", NeuralOptimization.compute_objective(network, result_nsverify.input, objective))