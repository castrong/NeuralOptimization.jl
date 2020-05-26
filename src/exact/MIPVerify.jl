"""
    Marabou(optimizer)

A branch and bound search with frequent bound tightening

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: hpolytope and hyperrectangle
3. Objective: Any linear objective

# Method
Branch and bound with bound tightening at each step in the simplex

# Property
Sound and complete

"""
@with_kw mutable struct MIPVerify
    
end

function optimize(solver::Marabou, problem::OutputOptimizationProblem, time_limit::Int = 30)
	@debug string("Optimizing with: ", solver)
    @assert problem.input isa Hyperrectangle or problem.input isa HPolytope

end

function Base.show(io::IO, solver::Marabou)
	print(io, string("Marabou_", "sbt=", string(solver.usesbt), "_", "dividestrategy=", solver.dividestrategy))
end
