"""
    VanillaMIP(optimizer, m, time_limit)

This formulates the problem directly as an MIP then uses the prescribed solver to solve the MIP.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: Polytope
3. Objective: Any linear objective

# Property
Sound and complete.

"""
@with_kw struct VanillaMIP
    optimizer = GLPK.Optimizer
    output_flag = 1 # output flag for JuMP model initialization
    threads = 1 # threads to use in the solver
    m::Float64 = 1.0e4  # The big M in the linearization
end

function to_string(solver::LBFGS)

    return
end

function optimize(solver::VanillaMIP, problem::OutputOptimizationProblem, time_limit::Int = 1200)
    @debug "Optimizing with VanillaMIP"
    model = Model(solver)
    neurons = init_neurons(model, problem.network)
    deltas = init_deltas(model, problem.network)
    objective = problem.objective

    encode_network!(model, problem.network, neurons, deltas, MixedIntegerLP(solver.m))
    add_set_constraint!(model, problem.input, first(neurons))

    # Add an objective to maximize our output
    weight_vector = linear_objective_to_weight_vector(objective, length(last(neurons)))
    if problem.max
        @objective(model, Max, transpose(weight_vector) * last(neurons))
    else
        @objective(model, Min, transpose(weight_vector) * last(neurons))
    end

    # Set a time limit
    set_time_limit_sec(model, time_limit)

    optimize!(model)
    @debug termination_status(model)

    if (termination_status(model) == OPTIMAL)
        @debug "VanillaMIP Returned Optimal"
        return Result(:success, value.(neurons[1]), objective_value(model))
    elseif (termination_status(model) == TIME_LIMIT)
        @debug "VanillaMIP Timed Out"
        return Result(:timeout, [-1.0], -1.0)
    else
        @debug "VanillaMIP Errored"
        return Result(:error, [-1.0], -1.0)
    end
end

function Base.show(io::IO, solver::VanillaMIP)
    optimizer_string = "otheroptimizer"
    if solver.optimizer == GLPK.Optimizer
        optimizer_string = "GLPK"
    elseif solver.optimizer == Gurobi.Optimizer
        optimizer_string = "Gurobi"
    end
    print(io, string("VanillaMIP_", optimizer_string, "_", string(solver.threads), "threads_", string(solver.m), "m"))
end
