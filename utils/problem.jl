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
	coefficients::Vector{N}
	# Defined as the indices of the output layer corresponding to the coefficients
	variables::Vector{Int}
end

# A way to convert your LinearObjective into a weight vector.
# ex: If coefficients = [1.0, -1.0, 1.0] and variables = [1, 4, 6] with n = 6
# then the weight vector is [1.0, 0, 0, -1.0, 0, 1.0]
function LinearObjectiveToWeightVector(objective::LinearObjective, n::Int)
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
