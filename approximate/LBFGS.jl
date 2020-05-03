# Input constraint is:
# Output constraint is:

@with_kw struct LBFGS
    test_num = 100
end

function optimize(solver::LBFGS, problem::Problem)
    # Assert the problem takes the necessary format
    @assert problem.input isa LazySet.Hyperrectangle
    @assert problem.output == nothing

    nnet = problem.network;

    # Incorporate your objective function into an obje
    function obj_function(x)
        compute_output(problem.network, )
    end

    input_lower = h.center - h.radius;
    input_upper = h.center + h.radius;
    x_0 = h.center

    if (problem.max)
        # f, gradient, lower bound input, upper bound input, initial guess
        # use negative value and gradient because Optim will minimize it
        result = Optim.optimize(x->-NeuralVerification.compute_output(nnet, x)[1],
                                x->-NeuralVerification.get_gradient(nnet, x),
                                input_lower, input_upper, x_0,
                                Optim.Fminbox(Optim.LBFGS()); inplace=false)
        opt_val = -minimum(result)
    else
        result = Optim.optimize(x->NeuralVerification.compute_output(nnet_nv, x)[1],
                                x->NeuralVerification.get_gradient(nnet_nv, x),
                                input_lower, input_upper, x_0,
                                Optim.Fminbox(Optim.LBFGS()); inplace=false)
        opt_val = minimum(result)
    end

    return Result(:success, minimizer(opt_val), opt_val)
end
