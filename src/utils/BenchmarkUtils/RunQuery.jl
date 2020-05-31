include("../../NeuralOptimization.jl")
using NPZ
using LazySets
#=
    Read in a description of the query from the command line arguments
    Format:
    optimizer_name,  optimizer, class, network_file, input_file, delta, objective_variables, objective_coefficients, maximize, query_output_filename

=#
comma_replacement = "[-]" # has to match with the comma replacement in BenchmarkFileWriters.jl!!!

args = string.(split(ARGS[1], ",")) # string. to convert from substring to string


optimizer_string = args[2]
chunks = split(optimizer_string, "_")

# take in a single argument which holds your arguments comma separated

# Optimizer name and optimizer itself
optimizer_name = args[1]
optimizer = NeuralOptimization.parse_optimizer(optimizer_string)

# Class of problem, network file, input file
class = args[3]
network_file = args[4]
input_file = args[5]

# Radii of our hyperrectangle, objective function
delta_list = parse.(Float64, split(args[6][2:end-1], comma_replacement))
objective_variables = parse.(Int, split(args[7][2:end-1], comma_replacement)) # clip off the [] in the list, then split based on commas, then parse to an int
objective_coefficients = parse.(Float64, split(args[8][2:end-1], comma_replacement))
objective = NeuralOptimization.LinearObjective(objective_coefficients, objective_variables)
timeout = 10 # hard coded for now

# Whether to maximize or minimize and our output filename
maximize = args[9] == "maximize" ? true : false
output_file = args[10]
# Make the path to your output file if it doesnt exist
mkpath(dirname(output_file))

#=
    Setup and run the query, and write the results to an output file
=#

# Read in the network
network = NeuralOptimization.read_nnet(network_file)
num_inputs = size(network.layers[1].weights, 2)

# Create the problem: network, input constraint, objective, maximize or minimize
center_input = npzread(input_file)
input = NeuralOptimization.Hyperrectangle(vec(center_input)[:], delta_list) # center and radius
problem = NeuralOptimization.OutputOptimizationProblem(network, input, objective, maximize, -Inf, Inf)

# Run simple problem to avoid startup time being counted
simple_nnet = NeuralOptimization.read_nnet("./Networks/small_nnet.nnet")
simple_objective = NeuralOptimization.LinearObjective([1.0], [1])
simple_input = NeuralOptimization.Hyperrectangle([1.0], [1.0])
simple_problem = NeuralOptimization.OutputOptimizationProblem(simple_nnet, simple_input, simple_objective, true, -Inf, Inf)
time_temp = @elapsed result = NeuralOptimization.optimize(optimizer, simple_problem, 20)

# Parse the optimizer
elapsed_time = @elapsed result = NeuralOptimization.optimize(optimizer, problem, timeout)
println("Result status: ", result.status)
println("Optimizer: ", optimizer)
println("Name: ", optimizer_name)
println("Result objective value: ", result.objective_value)
println("Elapsed time: ", elapsed_time)

optimal_input = result.input
optimal_output = -1
# We can only compute the output if our problem finished successfully
if (result.status == :success)
      println("Computing optimal output from optimal input")
      optimal_output = NeuralOptimization.compute_output(network, vec(result.input)[:])
end


output_file = string(output_file, ".csv") # add on the .csv
open(output_file, "w") do f
    # Writeout our results - for the optimal output we remove the brackets on the list 
    write(f,
          string(result.status), ",",
          string(result.objective_value), ",",
          string(elapsed_time), ",",
          string(optimal_output)[2:end-1], "\n")
   close(f)
end
