include("../src/NeuralOptimization.jl")
include("../src/utils/CreateAllOptimizers.jl")

using NPZ

# Network, input, and bound files
nnet_file = "./Networks/MNIST/mnist10x20.nnet"
network = NeuralOptimization.read_nnet(nnet_file)
input_radius = 0.016
use_bounds = true

input_file = "./Datasets/MNIST/MNISTlabel_0_index_0_.npy"
center_input = npzread(input_file)

bound_file = "/Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/Bounds/MNIST/bounds_mnist10x20_label_0_index_0_delta_0.016.txt"
bound_strings = readlines(bound_file)
# In the format lower, upper for each layer - won't include the input layer
bound_lists = [parse.(Float64, split(bound_str, ",")) for bound_str in bound_strings]
lower_bounds, upper_bounds = NeuralOptimization.bounds_to_lower_upper(bound_lists)

# Choose your objective and optimizer
objective = NeuralOptimization.LinearObjective([1.0, -1.0], [4, 1]) # objective is to just maximize the first output
maximize = true
println("Bound on objective (naive): ", NeuralOptimization.bounds_to_objective_bounds(objective, bound_lists[end-1], bound_lists[end]))
#optimizer = Marabou_optimizer_sbt_earliestReLU
#optimizer = Marabou_optimizer_sbt_ReLUViolation
optimizer = NeuralOptimization.Marabou(usesbt=false, dividestrategy = "ReLUViolation", triangle_relaxation=true)


# Setup the problem
num_inputs = size(network.layers[1].weights, 2)
input = NeuralOptimization.Hyperrectangle(vec(center_input)[:], input_radius * ones(num_inputs)) # center and radius

time_limit = 7200
# Send in bounds or an empty list
if (use_bounds)
    problem = NeuralOptimization.OutputOptimizationProblem(network=network, input=input, objective=objective, max=maximize, lower=0.0, upper=1.0, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
else
    problem = NeuralOptimization.OutputOptimizationProblem(network=network, input=input, objective=objective, max=maximize, lower=0.0, upper=1.0)
end
elapsed_time = @elapsed result = NeuralOptimization.optimize(optimizer, problem, time_limit)

println(result.status)
println("Optimal Objective: ", result.objective_value)
println("Took: ", elapsed_time)
