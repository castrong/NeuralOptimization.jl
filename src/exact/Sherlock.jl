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
    m = 1.0e6
    optimizer = GLPK.Optimizer
    output_flag = 1 # output flag for JuMP model initialization
    threads = 1 # threads to use in the solver
    ϵ::Float64 = 1e-4
end

function optimize(solver::Sherlock, problem::OutputOptimizationProblem, time_limit::Int = 1200)
    @debug string("Optimizing with: ", solver)
    @assert problem.input isa Hyperrectangle # Bc haven't implemented sample with polytopes
    start_time = time()

    # Augment the network to handle an arbitrary linear objective
    # if the last layer was ID() then this just combines the objective into that layer
    augmented_network = extend_network_with_objective(problem.network, problem.objective) # If the last layer is ID it won't add a layer
    augmented_objective = LinearObjective([1.0], [1])
    augmented_problem = OutputOptimizationProblem(augmented_network, problem.input, augmented_objective, problem.max)

    if (problem.max)
        (x_u, u) = output_bound(solver, augmented_problem, :max, start_time, time_limit)
        # If timed out, return a corresponding status - otherwise, return the result
        if (x_u == TIME_LIMIT)
            return Result(:timeout, [-1.0], -1.0)
        else
            opt_val = u[1]; # always 1 since augment network
            return Result(:success, x_u, opt_val)
        end
    else
        (x_l, l) = output_bound(solver, augmented_problem, :min, start_time, time_limit)
        # If timed out, return a corresponding status - otherwise, return the result
        if (x_l == TIME_LIMIT)
            return Result(:timeout, [-1.0], -1.0)
        else
            opt_val = l[1]; # always 1 since augment network
            return Result(:success, x_l, opt_val)
        end
    end
end

function output_bound(solver::Sherlock, problem::OutputOptimizationProblem, type::Symbol, start_time::Float64, time_limit::Int)
    opt = solver.optimizer
    x = sample(problem.input)
    while true
        (x, bound) = local_search(solver, problem, x, type, start_time, time_limit)
        if (x == TIME_LIMIT)
            return (TIME_LIMIT, TIME_LIMIT)
        end
        bound_ϵ = bound + ifelse(type == :max, solver.ϵ, -solver.ϵ)
        (x_new, bound_new, feasible) = global_search(solver, problem, bound_ϵ, type, start_time, time_limit)
        if (feasible == TIME_LIMIT)
            return (TIME_LIMIT, TIME_LIMIT)
        end
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

function local_search(solver::Sherlock, problem::OutputOptimizationProblem, x::Vector{Float64}, type::Symbol, start_time::Float64, time_limit::Int)
    nnet = problem.network
    act_pattern = get_activation(nnet, x)
    gradient = get_gradient(nnet, x)
    model = Model(solver)
    neurons = init_neurons(model, nnet)
    add_set_constraint!(model, problem.input, first(neurons))
    encode_network!(model, nnet, neurons, act_pattern, StandardLP())
    o = gradient * neurons[1]
    index = ifelse(type == :max, 1, -1)
    @objective(model, Max, index * o[1])

    set_time_limit_sec(model, trunc(Int, time_limit - (time() - start_time)))
    optimize!(model)
    if (termination_status(model) == TIME_LIMIT)
        return TIME_LIMIT, TIME_LIMIT
    end

    x_new = value.(neurons[1])
    bound_new = compute_output(nnet, x_new)
    return (x_new, bound_new[1])
end

function global_search(solver::Sherlock, problem::OutputOptimizationProblem, bound::Float64, type::Symbol, start_time::Float64, time_limit::Int)
    index = ifelse(type == :max, 1.0, -1.0)
    h = HalfSpace([index], index * bound)
    output_set = HPolytope([h])
    result_status, result_x  = ns_verify(solver::Sherlock, problem.network, problem.input, output_set, start_time, time_limit)
    if result_status == :violated
        bound = compute_output(problem.network, result_x)
        return (result_x, bound[1], true)
    elseif result_status == :holds
        return ([], 0.0, false)
    elseif result_status == TIME_LIMIT
        return (TIME_LIMIT, TIME_LIMIT, TIME_LIMIT)
    end
end

function ns_verify(solver::Sherlock, network, input_set, output_set, start_time::Float64, time_limit::Int)
    model = Model(solver)
    neurons = init_neurons(model, network)
    deltas = init_deltas(model, network)
    add_set_constraint!(model, input_set, first(neurons))
    add_complementary_set_constraint!(model, output_set, last(neurons))
    encode_network!(model, network, neurons, deltas, MixedIntegerLP(solver.m))
    feasibility_problem!(model)

    # Exit early if we've reached the time limit
    time_remaining = time_limit - (time() - start_time)
    if (time_remaining <= 0)
        return TIME_LIMIT, TIME_LIMIT
    end

    set_time_limit_sec(model, trunc(Int32, time_remaining))
    optimize!(model)
    if termination_status(model) == OPTIMAL
        return :violated, value.(first(neurons))

    elseif termination_status(model) == INFEASIBLE
        return :holds, -1

    elseif (termination_status(model) == TIME_LIMIT)
        return TIME_LIMIT, TIME_LIMIT
    end
    return CounterExampleResult(:unknown)
end

function Base.show(io::IO, solver::Sherlock)
    optimizer_string = "otheroptimizer"
    if solver.optimizer == GLPK.Optimizer
        optimizer_string = "GLPK.Optimizer"
    elseif solver.optimizer == Gurobi.Optimizer
        optimizer_string = "Gurobi.Optimizer"
    end
    print(io, string("Sherlock_", "optimizer=", optimizer_string, "_", "threads=", string(solver.threads), "_m=", string(solver.m)))
end
