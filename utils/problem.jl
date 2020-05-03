
# Our objective takes the form of a linear equation
struct LinearObjective{N<: Number}
	coefficients::Vector{N}
	# Defined as a vector of (layer number, index in layer)
	variables::Vector{Tuple{Int, Int}}
end
"""
    Problem{P, Q}(network::Network, input::P, output::Q, objective::O)

Problem definition for neural optimization.

The optimization problem consists of: find the point in the input set
s.t. the output is in the output set and the input maximizes (or minimizes
depending on the value of the bool max) the objective function
"""
struct Problem{P, Q}
    network::Network
    input::P
    output::Q
	objective::LinearObjective
	max::Bool
	# Can be :min_perturbation_linf, :linear_objective
	problem_type::Symbol
end

struct Result{N<: Number}
	# Status can be :success, :timeout, :error
    status::Symbol
	input::Vector{N}
	output::Float64
end
