#! /usr/bin/env julia

#=

    Run a query from a line with the following format:

    optimizer_string path/to/network/network.nnet path/to/property/property.txt path/to/result/result_file.txt

    For example:
    FGSM /Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/Networks/ACASXu/ACASXU_experimental_v2a_1_1.nnet /Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/OptimizationProperties/ACASXu/acas_property_optimization_1.txt /Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Results/FGSM.ACASXU_experimental_v2a_1_1.acas_property_optimization_1.txt

    To run a quick test:
    module test
           ARGS = ["--environment_path", "/Users/castrong/Desktop/Research/NeuralOptimization.jl/", "--optimizer", "VanillaMIP_optimizer=Gurobi.Optimizer_threads=1_m=1.0e6", "--network_file", "/Users/castrong/Desktop/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Networks/ACASXU_experimental_v2a_1_1.nnet", "--property_file", "/Users/castrong/Desktop/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Properties/acas_property_optimization_4.txt", "--result_file", "/Users/castrong/Desktop/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Results/FGSM.ACASXU_experimental_v2a_1_2.acas_property_optimization_4.txt"]
           include("/Users/castrong/Desktop/Research/NeuralOptimization.jl/src/utils/BenchmarkUtils/RunQueryFromProperty.jl")
    end

=#


using Pkg
using ArgParse
arg_settings = ArgParseSettings()
@add_arg_table! arg_settings begin
    "--environment_path"
        help = "Base path to your files. We will activate this package environment"
        arg_type = String
        default = "/Users/castrong/Desktop/Research/NeuralOptimization.jl"
	"--optimizer"
		help = "String describing the optimizer"
		arg_type = String
		default = "FGSM"
		required = true
    "--network_file"
        help = "Network file name"
        arg_type = String
        required = true
    "--property_file"
        help = "Property file name"
        arg_type = String
        required = true
	"--result_file"
		help = "Result file name"
		arg_type = String
		required = true
end

# Parse your arguments
parsed_args = parse_args(ARGS, arg_settings)
println(parsed_args)
environment_path = parsed_args["environment_path"]
optimizer_string = parsed_args["optimizer"]
network_file = parsed_args["network_file"]
property_file = parsed_args["property_file"]
result_file = parsed_args["result_file"]

# Activate the environment and include NeuralOptiimization.jl
Pkg.activate(environment_path)
include(joinpath(environment_path, "src/NeuralOptimization.jl"))

# A problem needs a network, input set, objective and whether to maximize or minimize.
# it also takes in the lower and upper bounds on the network input variables which describe
# the domain of the network.

println("network file: ", network_file)
network = NeuralOptimization.read_nnet(network_file)
num_inputs = size(network.layers[1].weights, 2)

if occursin("mnist", network_file)
	println("MNIST network")
	lower = 0.0
	upper = 1.0
elseif occursin("AutoTaxi", network_file)
	println("AutoTaxi network")
	lower = 0.0
	upper = 1.0
elseif occursin("ACASXU", network_file)
	println("ACAS network")
	lower = -Inf
	upper = Inf
else
	@assert false "Network category unrecognized"
end

input_set, objective, maximize_objective = NeuralOptimization.read_property_file(property_file, num_inputs; lower=lower, upper=upper)
println(input_set)
println(objective)
println(maximize_objective)
problem = NeuralOptimization.OutputOptimizationProblem(network, input_set, objective, maximize_objective, lower, upper)
optimizer = NeuralOptimization.parse_optimizer(optimizer_string)
println(optimizer)

NeuralOptimization.optimize(optimizer, problem, 10)
