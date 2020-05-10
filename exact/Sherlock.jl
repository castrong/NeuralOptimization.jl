"""
    Sherlock(optimizer, ϵ::Float64)

Sherlock combines local and global search to estimate the range of the output node.

# Problem requirement
1. Network: any depth, ReLU activation, single output
2. Input: hpolytope and hyperrectangle

# Method
Local search: solve a linear program to find local optima on a line segment of the piece-wise linear network.
Global search: solve a feasibilty problem using MILP encoding (default calling NSVerify).
- `optimizer` default `GLPKSolverMIP()`
- `ϵ` is the margin for global search, default `0.1`.

# Property
Sound but not complete.

# Reference
[S. Dutta, S. Jha, S. Sanakaranarayanan, and A. Tiwari,
"Output Range Analysis for Deep Neural Networks,"
*ArXiv Preprint ArXiv:1709.09130*, 2017.](https://arxiv.org/abs/1709.09130)

[https://github.com/souradeep-111/sherlock](https://github.com/souradeep-111/sherlock)
"""
@with_kw struct Sherlock
    m = 1e3
    optimizer = GLPK.Optimizer
    Threads::Int = 1
    time_limit::Int = 10
    ϵ::Float64 = 1e-4
end

function optimize(solver::Sherlock, problem::OutputOptimizationProblem)
    @debug "Optimizing with Sherlock"

    @assert problem.input isa Hyperrectangle # Bc haven't implemented sample with polytopes
    @assert length(problem.objective.coefficients) == 1 # we'll return the max or min of a single output variable
    objective = problem.objective

    (x_u, u) = output_bound(solver, problem, :max)
    (x_l, l) = output_bound(solver, problem, :min)
    println("bounds: [", l, ", ", u, "]")
    bound = Hyperrectangle(low = [l], high = [u])
    reach = Hyperrectangle(low = [l - solver.ϵ], high = [u + solver.ϵ])

    if (problem.max)
        opt_val = u[objective.variables[1]] * objective.coefficients[1];
        return Result(:success, x_u, opt_val)
    else
        opt_val = l[objective.variables[1]] * objective.coefficients[1];
        return Result(:success, x_l, opt_val)
    end
end

function output_bound(solver::Sherlock, problem::OutputOptimizationProblem, type::Symbol)
    opt = solver.optimizer
    x = sample(problem.input)
    while true
        (x, bound) = local_search(solver, problem, x, opt, type)
        bound_ϵ = bound + ifelse(type == :max, solver.ϵ, -solver.ϵ)
        (x_new, bound_new, feasible) = global_search(solver, problem, bound_ϵ, opt, type)
        feasible || return (x, bound)
        (x, bound) = (x_new, bound_new)
    end
end

# TODO: Implement Sample for  Hpolytopes - that's the only reason
# its restricted to a hyperrectangle right now
# Choose a single vertex
function sample(set::Hyperrectangle)
    return high(set)
end

function local_search(solver::Sherlock, problem::OutputOptimizationProblem, x::Vector{Float64}, optimizer, type::Symbol)
    nnet = problem.network
    act_pattern = get_activation(nnet, x)
    gradient = get_gradient(nnet, x)
    model = Model(with_optimizer(optimizer))
    neurons = init_neurons(model, nnet)
    add_set_constraint!(model, problem.input, first(neurons))
    encode_network!(model, nnet, neurons, act_pattern, StandardLP())
    o = gradient * neurons[1]
    index = ifelse(type == :max, 1, -1)
    @objective(model, Max, index * o[1])

    set_time_limit_sec(model, solver.time_limit)
    optimize!(model)
    x_new = value.(neurons[1])
    bound_new = compute_output(nnet, x_new)
    return (x_new, bound_new[1])
end

function global_search(solver::Sherlock, problem::OutputOptimizationProblem, bound::Float64, optimizer, type::Symbol)
    index = ifelse(type == :max, 1.0, -1.0)
    h = HalfSpace([index], index * bound)
    output_set = HPolytope([h])
    result_status, result_x  = ns_verify(solver::Sherlock, problem.network, problem.input, output_set)
    if result_status == :violated
        bound = compute_output(problem.network, result_x)
        return (result_x, bound[1], true)
    else
        return ([], 0.0, false)
    end
end

function ns_verify(solver::Sherlock, network, input_set, output_set)
    model = Model(solver)
    neurons = init_neurons(model, network)
    deltas = init_deltas(model, network)
    println("Neuron size: ", size(neurons))
    println("")
    add_set_constraint!(model, input_set, first(neurons))
    add_complementary_set_constraint!(model, output_set, last(neurons))
    encode_network!(model, network, neurons, deltas, MixedIntegerLP(solver.m))
    feasibility_problem!(model)
    
    set_time_limit_sec(model, solver.time_limit)
    optimize!(model)
    if termination_status(model) == OPTIMAL
        return :violated, value.(first(neurons))
    end
    if termination_status(model) == INFEASIBLE
        return :holds, -1
    end
    return CounterExampleResult(:unknown)
end
