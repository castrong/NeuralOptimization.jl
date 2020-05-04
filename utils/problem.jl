
# Linear objective on the output variables
struct LinearObjective{N<: Number}
	coefficients::Vector{N}
	# Defined as the indices of the output layer corresponding to the coefficients
	variables::Vector{Int}
end

"""
    Problem
Supertype of all problem types.
"""
abstract type Problem end

"""
    OutputOptimizationProblem{P, Q}(network::Network, input::P, output::Q, objective)

Problem definition for neural optimization.

The optimization problem consists of: find the point in the input set
s.t. the output is in the output set and the input maximizes (or minimizes
depending on the value of the bool max) the objective function
"""
struct OutputOptimizationProblem{P} <: Problem
    network::Network
    input::P
	objective::LinearObjective
	max::Bool
end

# A problem based on finding the minimum perturbation to the input
# on some norm in order to satisfy some constraints on the output
struct MinPerturbationProblem{P, N<: Number} <: Problem
	network::Network
	center_input::Vector{N}
	output::HPolytope
	objective::Symbol # can be :linf, :l1
end



struct Result{N<: Number}
	# Status can be :success, :timeout, :error
    status::Symbol
	input::Vector{N}
	objective_value::Float64
end
