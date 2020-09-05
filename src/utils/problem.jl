"""
    PolytopeComplement

The complement to a given set. Note that in general, a `PolytopeComplement` is not necessarily a convex set.
Also note that `PolytopeComplement`s are open by definition.

### Examples
```julia
julia> H = Hyperrectangle([0,0], [1,1])
Hyperrectangle{Int64}([0, 0], [1, 1])

julia> PC = complement(H)
PolytopeComplement of:
  Hyperrectangle{Int64}([0, 0], [1, 1])

julia> center(H) ∈ PC
false

julia> high(H).+[1,1] ∈ PC
true
```
"""
struct PolytopeComplement{S<:LazySet}
    P::S
end

Base.show(io::IO, PC::PolytopeComplement) = (println(io, "PolytopeComplement of:"), println(io, "  ", PC.P))
LazySets.issubset(s, PC::PolytopeComplement) = LazySets.is_intersection_empty(s, PC.P)
LazySets.is_intersection_empty(s, PC::PolytopeComplement) = LazySets.issubset(s, PC.P)
LazySets.tohrep(PC::PolytopeComplement) = PolytopeComplement(convert(HPolytope, PC.P))
Base.in(pt, PC::PolytopeComplement) = pt ∉ PC.P
complement(PC::PolytopeComplement)  = PC.P
complement(P::LazySet) = PolytopeComplement(P)
# etc.

# Linear objective on the output variables
struct LinearObjective{N<: Number}
	# Defined as the indices of the output layer corresponding to the coefficients
	coefficients::Vector{N}
	variables::Vector{Int}
end

# How to display linear objectives
function Base.show(io::IO, objective::LinearObjective)
	index = 1
    for (coefficient, variable) in zip(objective.coefficients, objective.variables)
		print(io, coefficient, "*y", variable)
		if (index != length(objective.coefficients))
			print(io, "+")
		end
		index = index + 1
	end
end

# A way to convert your LinearObjective into a weight vector.
# ex: If coefficients = [1.0, -1.0, 1.0] and variables = [1, 4, 6] with n = 6
# then the weight vector is [1.0, 0, 0, -1.0, 0, 1.0]
function linear_objective_to_weight_vector(objective::LinearObjective, n::Int)
    weight_vector = zeros(n)
    weight_vector[objective.variables] = objective.coefficients;
    return weight_vector
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
@with_kw struct OutputOptimizationProblem{P} <: Problem
    network::Network
    input::P
	objective::LinearObjective
	max::Bool
	# Upper and lower on the input variables, the domain of the network
	lower::Float64
	upper::Float64
	# Upper and lower bounds on all variables
	lower_bounds::Array{Array{Float64,1},1} = [[]]
	upper_bounds::Array{Array{Float64,1},1} = [[]]
end

# A problem based on finding the minimum perturbation to the input
# on some norm in order to satisfy some constraints on the output
@with_kw struct MinPerturbationProblem{N<: Number} <: Problem
	network::Network
	center::Vector{N}
	target::Int = Inf
	dims::Vector{Int} # Dims that we want to consider as part of the optimization
	input::Hyperrectangle # Used to add bounds on the input region that we'd like it to hold to
	output::HPolytope = HPolytope()
	norm_order::Float64
end



struct Result{N<: Number}
	# Status can be :success, :timeout, :error
    status::Symbol
	input::Vector{N}
	objective_value::Float64
end

struct MinPerturbationResult{N<: Number}
	# status can be :success, :timeout, :none_found
	status::Symbol
	input::Vector{N}
	perturbation::Float64 # value of the l-1 or l-inf norm for the optimal
end
