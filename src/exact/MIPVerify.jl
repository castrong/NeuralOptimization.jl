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
    dummy_var = false
end

function optimize(solver::MIPVerify, problem::OutputOptimizationProblem, time_limit::Int = 30)
	@debug "DON'T CALL THIS, MIPVERIFY DUMMY CODE"
end

function Base.show(io::IO, solver::MIPVerify)
	print(io, "MIPVerify")
end
