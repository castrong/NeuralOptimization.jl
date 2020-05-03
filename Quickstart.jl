# Read in an example network
include("./NeuralOptimization.jl")

nnet_file = "./Networks/AutoTaxi/AutoTaxi_32Relus_200Epochs_OneOutput.nnet"

optimizer = NeuralOptimization.LBFGS()

# Create the problem: network, input, output,
network = NeuralOptimization.read_nnet(nnet_file)
input = Hyperrectangle(low=[-1, -1], high=[1, 1])
output = nothing

end_layer_index = length(network.layers) + 1 # add 1 to account for the input layer
objective = LinearObjective([1.0], [(end_layer_index, 1)])


# struct Problem{P, Q}
#     network::Network
#     input::P
#     output::Q
#
# 	objective::linear_objective
# 	max::Bool
# 	# Can be :min_perturbation_linf, :linear_objective
# 	problem_type::Symbol
# end
