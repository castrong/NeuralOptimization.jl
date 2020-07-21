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
	optimizer = GLPK.Optimizer
    threads = 1
	strategy = "lp"
	preprocess_timeout_per_node = 5
end

function optimize(solver::MIPVerify, problem::OutputOptimizationProblem, time_limit::Int = 30)
	@debug "DON'T CALL THIS, MIPVERIFY DUMMY CODE"
end

function Base.show(io::IO, solver::MIPVerify)
	optimizer_string = "otheroptimizer"
    if solver.optimizer == GLPK.Optimizer
        optimizer_string = "GLPK.Optimizer"
    elseif solver.optimizer == Gurobi.Optimizer
        optimizer_string = "Gurobi.Optimizer"
    end
    print(io, string("MIPVerify_", "optimizer=", optimizer_string, "_", "threads=", string(solver.threads), "_strategy=", solver.strategy, "_preprocesstimeoutpernode=", solver.preprocess_timeout_per_node))
end
