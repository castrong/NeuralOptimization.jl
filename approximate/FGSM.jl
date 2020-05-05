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

function optimize(solver::FGSM, problem::Problem)
    @debug "Optimizing with FGSM"

    # only works with hypercube for now
    @assert problem.input isa Hyperrectangle
    @assert all(r->r==problem.input.radius[1], problem.input.radius)
    radius = problem.input.radius[1]
    x_0 = problem.input.center
    true_label = -1 # this won't matter since our loss function doesn't depend on it

    num_outputs = length(problem.network.layers[end].bias)

    flux_model = Flux.Chain(problem.network)
    weight_vector = LinearObjectiveToWeightVector(problem.objective, num_outputs)
    if (problem.max)
        x_adv = Adversarial.FGSM(flux_model, (x, y)->transpose(weight_vector) * flux_model(x), x_0, true_label; ϵ = radius, clamp_range=(0,1))
    else
        x_adv = Adversarial.FGSM(flux_model, (x, y)->-transpose(weight_vector) * flux_model(x), x_0, true_label; ϵ = radius, clamp_range=(0,1))
    end
    return Result(:success, x_adv, flux_model(x_adv))
end
