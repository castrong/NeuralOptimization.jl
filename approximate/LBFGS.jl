# Input constraint is:
# Output constraint is:

@with_kw struct LBFGS
    test_num = 100
end

function optimize(solver::LBFGS, problem::Problem)
    @debug "Optimizing with LBFGS"
    # Assert the problem takes the necessary format
    # restriction from Optim, where we're getting our implementation
    # of LBFGS, focusing on box constrained problems
    @assert problem.input isa Hyperrectangle
    @assert problem.output == nothing

    # Since we don't yet have a gradient for a general objective
    # assert we're taking the gradient of an output - this wouldn't be hard to add though!
    # Just adding in the gradient for each of the subvariables
    # TODO: Add support for gradient with linear objective
    @assert length(problem.objective.coefficients) == 1
    @assert problem.objective.variables[1][1] == length(problem.network.layers) + 1 # the objective is an output variable
    obj_variable = problem.objective.variables[1]
    obj_coefficient = problem.objective.coefficients[1];

    nnet = problem.network;

    # Incorporate your objective function into an obje
    function obj_function(x)
        compute_output(problem.network, )
    end

    input_set = problem.input
    input_lower = input_set.center - input_set.radius;
    input_upper = input_set.center + input_set.radius;
    x_0 = input_set.center

    # Maximize
    if (problem.max)
        # f, gradient, lower bound input, upper bound input, initial guess
        # use negative value and gradient because Optim will minimize it
        result = Optim.optimize(x->-obj_coefficient * compute_output(nnet, x)[obj_variable[2]],
                                x->-obj_coefficient * get_gradient(nnet, x)[obj_variable[2], :],
                                input_lower, input_upper, x_0,
                                Optim.Fminbox(Optim.LBFGS()); inplace=false)
        opt_val = -minimum(result)
    # Minimize
    else
        result = Optim.optimize(x->obj_coefficient * compute_output(nnet, x)[obj_variable[2]],
                                x->obj_coefficient * get_gradient(nnet, x)[obj_variable[2], :],
                                input_lower, input_upper, x_0,
                                Optim.Fminbox(Optim.LBFGS()); inplace=false)
        opt_val = minimum(result)
    end
    @debug "Result from LBFGS: " result

    return Result(:success, result.minimizer, opt_val)
end
