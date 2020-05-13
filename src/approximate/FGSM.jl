"""
    FGSM()

This uses Adversarial.jl's implementation of FGSM to approximately solve the problem

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: Hypercube (small change to FGSM in Adversarial.jl needed to extend to hyperrectangle)
3. Objective: Any linear objective

# Property
Approximate

"""

@with_kw struct FGSM
    dummy_var = 1
end

function optimize(solver::FGSM, problem::Problem, time_limit::Int = 1200)
    @debug "Optimizing with FGSM"

    # only works with hypercube for now
    @assert problem.input isa Hyperrectangle
    @assert all(r->r==problem.input.radius[1], problem.input.radius)
    radius = problem.input.radius[1]
    x_0 = problem.input.center
    true_label = -1 # this won't matter since our loss function doesn't depend on it

    num_outputs = length(problem.network.layers[end].bias)

    flux_model = Flux.Chain(problem.network)
    weight_vector = linear_objective_to_weight_vector(problem.objective, num_outputs)

    # Odd reshaping to get this version of Flux to work with LinearAlgebra (Transpose had errors)
    # this is our objective - FGSM tries to maximize the cost so we're framing our objective as the cost
    cost_function = (x, y) -> (reshape(weight_vector, 1, num_outputs) * reshape(flux_model(x), num_outputs, 1))[1]
    if (problem.max)
        x_adv = Adversarial.FGSM(flux_model, (x, y)->cost_function(x, y), x_0, true_label; ϵ = radius, clamp_range=(0,1))
        obj_val = cost_function(x_adv, -1)
    else
        x_adv = Adversarial.FGSM(flux_model, (x, y)->-cost_function(x, y), x_0, true_label; ϵ = radius, clamp_range=(0,1))
        obj_val = cost_function(x_adv, -1)
    end
    return Result(:success, x_adv, obj_val)
end

function Base.show(io::IO, solver::FGSM)
  print(io, "FGSM")
end
