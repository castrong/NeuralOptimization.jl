#! /usr/bin/env julia

#=

    Run a query from a line with the following format:

    optimizer_string path/to/network/network.nnet path/to/property/property.txt path/to/result/result_file.txt

    For example:
    FGSM /Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/Networks/ACASXu/ACASXU_experimental_v2a_1_1.nnet /Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/OptimizationProperties/ACASXu/acas_property_optimization_1.txt /Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Results/FGSM.ACASXU_experimental_v2a_1_1.acas_property_optimization_1.txt

    To run a quick test:
    module test
           ARGS = ["--environment_path", "/Users/castrong/Desktop/Research/NeuralOptimization.jl/", "--optimizer", "Marabou_sbt=true_dividestrategy=EarliestReLU", "--network_file", "/Users/castrong/Desktop/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Networks/mnist10x20.nnet", "--property_file", "/Users/castrong/Desktop/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Properties/mnist_property_mininput_MNISTlabel_2_index_1__3.txt", "--result_file", "/Users/castrong/Desktop/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Results/Marabou_sbt=true_dividestrategy=EarliestReLU.mnist10x20.mnist_property_mininput_MNISTlabel_2_index_1__3.txt"]
           include("/Users/castrong/Desktop/Research/NeuralOptimization.jl/src/utils/BenchmarkUtils/RunQueryFromProperty.jl")
    end

	module test
           ARGS = ["--environment_path", "/Users/castrong/Desktop/Research/NeuralOptimization.jl/", "--optimizer", "Marabou_sbt=true_dividestrategy=EarliestReLU", "--network_file", "/Users/castrong/Downloads/full_benchmarks/Networks/ACASXU_experimental_v2a_1_5.nnet", "--property_file", "/Users/castrong/Downloads/full_benchmarks/Properties/acas_property_optimization_4.txt", "--result_file", "/Users/castrong/Desktop/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Results/debug_result.txt"]
           include("/Users/castrong/Desktop/Research/NeuralOptimization.jl/src/utils/BenchmarkUtils/RunQueryFromProperty.jl")
    end

=#


using Pkg
using ArgParse
using CPUTime

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
	"--timeout"
		help = "Internal soft timeout for the solve"
		arg_type = Int64
		default = 36000
end

# Parse your arguments
parsed_args = parse_args(ARGS, arg_settings)
println("Parsed args: ", parsed_args)
environment_path = parsed_args["environment_path"]
optimizer_string = parsed_args["optimizer"]
network_file = parsed_args["network_file"]
property_file = parsed_args["property_file"]
result_file = parsed_args["result_file"]
timeout = parsed_args["timeout"]

# If the result file is already there do nothing
if !isfile(result_file)

	# Activate the environment and include NeuralOptiimization.jl
	Pkg.activate(environment_path)
	include(joinpath(environment_path, "src/NeuralOptimization.jl"))

	# Parse your optimizer
	optimizer = NeuralOptimization.parse_optimizer(optimizer_string)

	# Run a simple problem to avoid startup time being counted
	simple_nnet = NeuralOptimization.read_nnet(joinpath(environment_path, "./Networks/small_nnet.nnet"))
	simple_objective = NeuralOptimization.LinearObjective([1.0], [1])
	simple_input = NeuralOptimization.Hyperrectangle([1.0], [1.0])
	simple_problem = NeuralOptimization.OutputOptimizationProblem(network=simple_nnet, input=simple_input, objective=simple_objective, max=true, lower=-Inf,upper=Inf)
	simple_input_problem = NeuralOptimization.MinPerturbationProblem(network=simple_nnet, input=simple_input, center = [0.5], target=1, dims=[1], output = NeuralOptimization.HPolytope([NeuralOptimization.HalfSpace([1.0], 5.0)]), norm_order=Inf)
	time_simple_output = @elapsed result_output = NeuralOptimization.optimize(optimizer, simple_problem, 20)
	time_simple_input = @elapsed result_input = NeuralOptimization.optimize(optimizer, simple_input_problem, 20)
	println("Simple output problem ran in: ", time_simple_output)
	println("Simple input problem ran in: ", time_simple_input)
	println("Simple min problem output: ", result_input)

	# A problem needs a network, input set, objective and whether to maximize or minimize.
	# it also takes in the lower and upper bounds on the network input variables which describe
	# the domain of the network.
	println("network file: ", network_file)
	network = NeuralOptimization.read_nnet(network_file)
	num_inputs = size(network.layers[1].weights, 2)

	if occursin("mnist", basename(network_file))
		println("MNIST network")
		lower = 0.0
		upper = 1.0
	elseif occursin("AutoTaxi", basename(network_file))
		println("AutoTaxi network")
		lower = 0.0
		upper = 1.0
	elseif occursin("ACASXU", basename(network_file))
		println("ACAS network")
		lower = -Inf
		upper = Inf
	else
		@assert false "Network category unrecognized"
	end

	# Read your problem using the property file
	# then run and time the optimizer
	problem = NeuralOptimization.property_file_to_problem(property_file, network, lower, upper)
	CPUtic()
	result = NeuralOptimization.optimize(optimizer, problem, timeout)
	elapsed_time = CPUtoc()

	# Print some things for help debugging
	println("Result status: ", result.status)
	println("Optimizer: ", optimizer)
	println("Result objective value: ", result.objective_value)
	println("Elapsed time: ", elapsed_time)

	optimal_input = result.input
	optimal_output = []
	# We can only compute the output if our problem finished successfully
	if (result.status == :success)
	      println("Computing optimal output from optimal input")
	      optimal_output = NeuralOptimization.compute_output(network, vec(result.input)[:])
	end

	open(result_file, "w") do f
	    # Writeout our results - for the optimal output we remove the brackets on the list
	    print(f, string(result.status), ",") # status
	    print(f, string(result.objective_value), ",") # objective value
		println(f, string(elapsed_time)) # elapsed time, end this line

		# If it was successful then write out the optimal input and output (ignores partial results for now)
		if (result.status == :success)
			println(f, string(optimal_input)[2:end-1]) # Write optimal input on its own line
			println(f, string(optimal_output)[2:end-1])# Write optimal output on its own line
		end
	end

# The file already exists
else
	println("Result File: ", result_file)
	println("Result file already exists, skipping execution!")
end
