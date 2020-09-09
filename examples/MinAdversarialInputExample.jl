ENV["JULIA_DEBUG"] = Main # turns on logging (@debug, @info, @warn) for "included" files
include("../src/NeuralOptimization.jl")
using NPZ
using LazySets

# Network and input files
nnet_file = "./Networks/ACASXu/ACASXU_experimental_v2a_2_2.nnet"
network = NeuralOptimization.read_nnet(nnet_file)
input_set = Hyperrectangle(low=[0.6, -0.5, -0.5, 0.45, -0.5],
					   high=[0.6798577687, 0.5, 0.5, 0.5, -0.45])
center_input = input_set.center
problem = NeuralOptimization.MinPerturbationProblem(network=network,
								 center=center_input,
								 target=target,
								 input=input_set,
								 norm_order=Inf,
								 dims = [1])

# nnet_file = "./Networks/MNIST/mnist10x20.nnet"
# network = NeuralOptimization.read_nnet(nnet_file)
# example_input = "./Datasets/MNIST/MNISTlabel_0_index_0_.npy"
# center_input = vec(npzread(example_input))
# radius = 0.5
# input_set = Hyperrectangle(low=max.(center_input .- radius, 0.0), high=min.(center_input .+ radius, 1.0))
#
# target = 3
# label = argmax(NeuralOptimization.compute_output(network, center_input))
# println("Label: ", label, "  Target: ", target)
#
# problem = NeuralOptimization.MinPerturbationProblem(network=network,
# 								 center=center_input,
# 								 target=target,
# 								 input=input_set,
# 								 norm_order=Inf,
# 								 dims = [])

#NeuralOptimization.optimize(NeuralOptimization.LBFGS(), problem)
NeuralOptimization.optimize(NeuralOptimization.Marabou(), problem, 60)


# # A problem based on finding the minimum perturbation to the input
# # on some norm in order to satisfy some constraints on the output
# @with_kw struct MinPerturbationProblem{N<: Number} <: Problem
# 	network::Network
# 	center_input::Vector{N}
# 	input::Hyperrectangle # Used to add bounds on the input region that we'd like it to hold to
# 	output::HPolytope
# 	objective::Symbol = :linf # can be :linf
# 	dims::Vector{Int} # Dims that we want to consider as part of the optimization
# end
