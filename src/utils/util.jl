"""
    read_nnet(fname::String; last_layer_activation = Id())
Read in neural net from a `.nnet` file and return Network struct.
The `.nnet` format is borrowed from [NNet](https://github.com/sisl/NNet).
The format assumes all hidden layers have ReLU activation.
Keyword argument `last_layer_activation` sets the activation of the last
layer, and defaults to `Id()`, (i.e. a linear output layer).
"""
function read_nnet(fname::String; last_layer_activation = Id())
    f = open(fname)
    line = readline(f)
    while occursin("//", line) #skip comments
        line = readline(f)
    end
    # number of layers
    nlayers = parse(Int64, split(line, ",")[1])
    # read in layer sizes
    layer_sizes = parse.(Int64, split(readline(f), ",")[1:nlayers+1])
    # read past additonal information
    for i in 1:5
        line = readline(f)
    end
    # i=1 corresponds to the input dimension, so it's ignored
    layers = Layer[read_layer(dim, f) for dim in layer_sizes[2:end-1]]
    push!(layers, read_layer(last(layer_sizes), f, last_layer_activation))

    return Network(layers)
end


"""
    write_nnet(fname::String, network::Network)
    write out a neural network in the .nnet format. Based on
    code here: https://github.com/sisl/NNet/blob/master/utils/writeNNet.py.
"""
function write_nnet(fname::String, network::Network)
    println("Writting to: ", fname)
    println("Size last layer: ", size(network.layers[end].weights))
    open(fname, "w") do f
        #####################
        # First, we write the header lines:
        # The first line written is just a line of text
        # The second line gives the four values:
        #     Number of fully connected layers in the network
        #     Number of inputs to the network
        #     Number of outputs from the network
        #     Maximum size of any hidden layer
        # The third line gives the sizes of each layer, including the input and output layers
        # The fourth line gives an outdated flag, so this can be ignored
        # The fifth line specifies the minimum values each input can take
        # The sixth line specifies the maximum values each input can take
        #     Inputs passed to the network are truncated to be between this range
        # The seventh line gives the mean value of each input and of all outputs
        # The eighth line gives the range of each input and of all outputs
        #     These two lines are used to map raw inputs to the 0 mean, unit range of the inputs and outputs
        #     used during training
        # The ninth line begins the network weights and biases
        ####################
        layers = network.layers

        # line 1
        write(f, "// Neural Network File Format by Kyle Julian, Stanford 2016\n") # line 1

        # line 2 (i)
        write(f, string(length(network.layers), ",")) # number of layer connections (# of weight bias pairs)

        num_inputs = size(network.layers[1].weights, 2)
        write(f, string(num_inputs, ",")) # number of inputs

        num_outputs = length(network.layers[end].bias)
        write(f, string(num_outputs, ",")) # number of outputs

        hidden_layer_sizes = [size(layer.weights, 1) for layer in network.layers][1:end-1] # chop off the output layer
        # mimicking https://github.com/sisl/NNet/blob/master/utils/writeNNet.py,
        # set the max hidden layer size to just be the input size
        write(f, string(num_inputs), ",\n") # max size of any hidden layer

        # line 3
        layer_sizes = [num_inputs, hidden_layer_sizes..., num_outputs]
        write(f, string(join(layer_sizes, ','), ","))
        write(f, "\n")

        # line 4
        write(f, "0,\n") # outdated flag, ignore

        # line 5, 6
        write(f, string(join(fill(-floatmax(Float16), num_inputs), ","), ",\n"))
        write(f, string(join(fill(floatmax(Float16), num_inputs), ","), ",\n"))

        # line 7, 8 - mean, range of 0, 1 means it won't rescale the input at all
        # is asserting our inputs are already scaled to be mean 0, range 1
        write(f, string(join(fill(0.0, num_inputs), ","), ",\n"))
        write(f, string(join(fill(1.0, num_inputs), ","), ",\n"))

        ##################
        # Write weights and biases of neural network
        # First, the weights from the input layer to the first hidden layer are written
        # Then, the biases of the first hidden layer are written
        # The pattern is repeated by next writing the weights from the first hidden layer to the second hidden layer,
        # followed by the biases of the second hidden layer.
        ##################
        for layer in layers
            # each output of a layer gets a line for its corresponding weights
            # and corresponding bias

            # Write the current weight
            weights = layer.weights
            for i = 1:size(weights, 1)
                for j = 1:size(weights, 2)
                    write(f, @sprintf("%.10e,", weights[i, j])) #five digits written. More can be used, but that requires more space.
                end
                write(f, "\n")
            end

            # Write the current bias
            bias = layer.bias
            for i = 1:length(bias)
                write(f, @sprintf("%.10e,", bias[i])) #five digits written. More can be used, but that requires more space.
                write(f, "\n")
            end

        end

        close(f)
    end


end

"""
    read_property_file(filename::String)

Read a property file and return: (i) an input set, and (ii) an objective, and
(iii) a boolean that is true if the objective should be maximized.
Each line in the property file

For now we assume a hyper-rectangle input set.
"""
function read_property_file(filename::String, num_inputs::Int64; lower::Float64=0.0, upper::Float64=1.0)

    # Keep track of the input lower and upper bounds that you accumulate
    lower_bounds = lower .* ones(num_inputs)
    upper_bounds = upper .* ones(num_inputs)
    # Variables and coefficients for objective
    variables::Vector{Int64} = []
    coefficients::Vector{Float64} = []
    maximize_objective = true

    lines = readlines(filename)
    for line in lines
        line = replace(line, " "=>"") # Remove spaces
        if occursin("Maximize", line) || occursin("Minimize", line)
            println("Objective line: ", line)
            maximize_objective = line[1:8] == "Maximize" ? true : false
            expr_string = line[9:end]
            done = false

            while !done
                plus_index = findfirst('+', expr_string)
                done = plus_index == nothing # You're finished if you've reached the last term (no + left)

                # If you're on the last term adjust the index appropriately
                plus_index = plus_index == nothing ? length(expr_string)+1 : plus_index # handle if + is not found, the last term

                # Isolate the current term and parse it
                cur_term = expr_string[1:plus_index-1]
                loc_y = findfirst('y', cur_term)
                @assert loc_y != nothing "didn't find a y in this term"
                coefficient_string = cur_term[1:loc_y-1]
                if (coefficient_string == "-")
                    coefficient = -1.0
                elseif (coefficient_string == "")
                    coefficient = 1.0
                else
                    coefficient = parse(Float64, coefficient_string)
                end
                variable = parse(Int64, cur_term[loc_y + 1:end]) + 1 # +1 in index since property file starts indexing from 0

                # Add the coefficient and variable to the list
                push!(coefficients, coefficient)
                push!(variables, variable)

                # Update your expr_string to cut off the first term
                expr_string = expr_string[plus_index+1:end]
            end
        elseif occursin("x", line)
            # Handle each type of comparator
            if (occursin("<=", line))
                comparator_index = findfirst("<=", line)
                x_index = findfirst('x', line)
                variable_index = parse(Int64, line[x_index+1:comparator_index[1]-1])  # go from after x to before comparator
                scalar = parse(Float64, line[comparator_index[2]+1:end])
                upper_bounds[variable_index + 1] = min(upper, scalar) # +1 in index since property file starts indexing from 0
            elseif (occursin(">=", line))
                comparator_index = findfirst(">=", line)
                x_index = findfirst('x', line)
                variable_index = parse(Int64, line[x_index+1:comparator_index[1]-1])  # go from after x to before comparator
                scalar = parse(Float64, line[comparator_index[2]+1:end])
                lower_bounds[variable_index + 1] = max(lower, scalar) # +1 in index since property file starts indexing from 0
            elseif (occursin("==", line)) # is it == or =?
                comparator_index = findfirst("==", line)
                x_index = findfirst('x', line)
                variable_index = parse(Int64, line[x_index+1:comparator_index[1]-1]) # go from after x to before comparator
                scalar = parse(Float64, line[comparator_index[2]+1:end])
                lower_bounds[variable_index + 1] = max(lower, scalar) # +1 in index since property file starts indexing from 0
                upper_bounds[variable_index + 1] = min(upper, scalar)
            else
                @assert false string("Unrecognized comparator: ", line)
            end
        else
            @assert false string("Unrecognized line in property file: ", line)
        end
    end

    # Return the hyperrectangle, the objective, and whether to maximize or minimize
    return NeuralOptimization.Hyperrectangle(low=lower_bounds, high=upper_bounds), NeuralOptimization.LinearObjective(coefficients, variables), maximize_objective
end

"""
    read_layer(output_dim::Int, f::IOStream, [act = ReLU()])

Read in layer from nnet file and return a `Layer` containing its weights/biases.
Optional argument `act` sets the activation function for the layer.
"""
function read_layer(output_dim::Int64, f::IOStream, act = ReLU())

    rowparse(splitrow) = parse.(Float64, splitrow[findall(!isempty, splitrow)])
     # first read in weights
     W_str_vec = [rowparse(split(readline(f), ",")) for i in 1:output_dim]
     weights = vcat(W_str_vec'...)
     # now read in bias
     bias_string = [split(readline(f), ",")[1] for j in 1:output_dim]
     bias = rowparse(bias_string)
     # activation function is set to ReLU as default
     return Layer(weights, bias, act)
end

"""
    compute_output(nnet::Network, input::Vector{Float64})

Propagate a given vector through a nnet and compute the output.
"""
function compute_output(nnet::Network, input)
    curr_value = input
    for layer in nnet.layers # layers does not include input layer (which has no weights/biases)
        curr_value = layer.activation(affine_map(layer, curr_value))
    end
    return curr_value # would another name be better?
end


function linear_objective_to_weight_vector(objective::LinearObjective, n::Int)
    weight_vector = zeros(n)
    weight_vector[objective.variables] = objective.coefficients;
    return weight_vector
end

# Given upper and lower bounds on the output variables
# give an upper and lower bound on the objective
function bounds_to_objective_bounds(objective::LinearObjective, output_lower, output_upper)
    objective_lower = 0
    objective_upper = 0
    for i = 1:length(objective.coefficients)
        coeff = objective.coefficients[i]
        var = objective.variables[i]
        # With negative coefficients we must switch which the upper and lower will really correspond to
        # in terms of their contribution to the objective
        if (coeff < 0)
            objective_lower = objective_lower + output_upper[var] * coeff
            objective_upper = objective_upper + output_lower[var] * coeff
        else
            objective_lower = objective_lower + output_lower[var] * coeff
            objective_upper = objective_upper + output_upper[var] * coeff
        end
    end
    return objective_lower, objective_upper
end

# List of bounds alternates between upper and lower - separate into two lists
function bounds_to_lower_upper(bounds)
    @assert iseven(length(bounds))
    return bounds[1:2:end], bounds[2:2:end]
end

function parse_optimizer(optimizer_string)
    println("Optimizer string: ", optimizer_string)
    chunks = split(optimizer_string, "_")
    optimizer_type = chunks[1]
    chunks = split(optimizer_string, "_") # optimizer type, followed by arguments separated by _
    if (optimizer_type == "Marabou")
        sbt_string = split(chunks[2], "=")[2]
        sbt = parse(Bool, sbt_string)
        dividestrategy = split(chunks[3], "=")[2]
        return NeuralOptimization.Marabou(usesbt=sbt, dividestrategy=dividestrategy)
    elseif (optimizer_type == "MarabouBinarySearch")
        sbt_string = split(chunks[2], "=")[2]
        sbt = parse(Bool, sbt_string)
        dividestrategy = split(chunks[3], "=")[2]
        return NeuralOptimization.MarabouBinarySearch(usesbt=sbt, dividestrategy=dividestrategy)
    elseif (optimizer_type == "Sherlock")
        backend_optimizer_string = split(chunks[2], "=")[2]
        threads_string = split(chunks[3], "=")[2]
        m_string = split(chunks[4], "=")[2]
        @assert backend_optimizer_string == "Gurobi.Optimizer" || backend_optimizer_string == "GLPK.Optimizer"
        if backend_optimizer_string == "Gurobi.Optimizer"
            backend = Gurobi.Optimizer
            threads = parse(Int, threads_string)
            m = parse(Float32, m_string)
            return NeuralOptimization.Sherlock(optimizer=backend, threads=threads, m=m)
        else
            backend = GLPK.Optimizer
            m = parse(Float32, m_string)
            return NeuralOptimization.Sherlock(optimizer=backend, m=m)
        end
    elseif (optimizer_type == "VanillaMIP")
        backend_optimizer_string = split(chunks[2], "=")[2]
        threads_string = split(chunks[3], "=")[2]
        m_string = split(chunks[4], "=")[2]
        @assert backend_optimizer_string == "Gurobi.Optimizer" || backend_optimizer_string == "GLPK.Optimizer"
        if backend_optimizer_string == "Gurobi.Optimizer"
            backend = Gurobi.Optimizer
            threads = parse(Int, threads_string)
            m = parse(Float32, m_string)
            return NeuralOptimization.VanillaMIP(optimizer=backend, threads=threads, m=m)
        else
            backend = GLPK.Optimizer
            m = parse(Float32, m_string)
            return NeuralOptimization.VanillaMIP(optimizer=backend, m=m)
        end
    elseif (optimizer_type == "LBFGS")
        return NeuralOptimization.LBFGS()
    elseif (optimizer_type == "FGSM")
        return NeuralOptimization.FGSM()
    elseif (optimizer_type == "PGD")
        return NeuralOptimization.PGD()
    end
end

# Convert between the two types - for now just support id and ReLU activations
function network_to_mipverify_network(network, label="default_label")
    mipverify_layers = []
    for layer in network.layers
        weights = copy(transpose(layer.weights)) # copy to get rid of transpose type
        bias = layer.bias
        push!(mipverify_layers, MIPVerify.Linear(weights, bias))
        if (layer.activation == ReLU())
            @debug "Adding ReLU layer to MIPVerify representation"
            push!(mipverify_layers, MIPVerify.ReLU())
        elseif (layer.activation == Id())
            @debug "ID layer for MIPVerify is assumed (no explicit representation)"
        else
            @debug "Only ID and ReLU activations supported right now"
            throw(ArgumentError("Only ID and ReLU activations supported right now"))
        end
    end
    return MIPVerify.Sequential(mipverify_layers, label)
end

"""
extend_network_with_objective(network::Network, objective::LinearObjective, negative_objective::Bool)

If the last layer is an Id() layer, then changes the layer to account for the objective.
It becomes a single output whose value will be equal to that objective.
If the last layer is not an Id() layer, then adds a layer to the end of a network which makes
the single output of this augmented network
equal to the objective function evaluated on the original output layer

negative_objective can specify that you'd actually like the output to be the negative of the objective

Returns the new network
"""
# if the last layer is ID can we replace it with just a new weight and bias
# e.g. if it was y = Ax + b, it can become c' (Ax + b) = c' Ax + c'b where c' is our weight vector
function extend_network_with_objective(network::Network, objective::LinearObjective, negative_objective::Bool=false)
    nnet = deepcopy(network)
    weight_vector = linear_objective_to_weight_vector(objective, length(nnet.layers[end].bias))
    last_layer = nnet.layers[end]
    obj_scaling = negative_objective ? -1.0 : 1.0 # switches between positive or negative objective

    # If the last layer is Id() we can replace the last layer with a new one
    if (last_layer.activation == Id())
        new_weights = Array(transpose(weight_vector) * last_layer.weights) * obj_scaling
        new_bias = Array([transpose(weight_vector) * last_layer.bias]) * obj_scaling

        @assert size(new_weights, 1) == 1
        @assert length(new_bias) == 1
        nnet.layers[end] = Layer(new_weights, new_bias, Id())
        return nnet
    # Otherwise we add an extra layer on
    else
        new_layer = Layer(weight_vector * obj_scaling, [0], ID())
        push!(nnet.layers, new_layer)
        return nnet
    end
end

function compute_objective(nnet::Network, input, objective::LinearObjective)
    curr_value = input
    for layer in nnet.layers # layers does not include input layer (which has no weights/biases)
        curr_value = layer.activation(affine_map(layer, curr_value))
    end

    # Fill in a weight vector from the objective, then dot it with the output layer
    weight_vector = linear_objective_to_weight_vector(objective, length(curr_value))
    return transpose(weight_vector) * curr_value # would another name be better?
end

"""
    get_activation(L, x::Vector)
Finds the activation pattern of a vector `x` subject to the activation function given by the layer `L`.
Returns a Vector{Bool} where `true` denotes the node is "active". In the sense of ReLU, this would be `x[i] >= 0`.
"""
get_activation(L::Layer{ReLU}, x::Vector) = x .>= 0.0
get_activation(L::Layer{Id}, args...) = trues(n_nodes(L))

"""
    get_activation(nnet::Network, x::Vector)

Given a network, find the activation pattern of all neurons at a given point x.
Returns Vector{Vector{Bool}}. Each Vector{Bool} refers to the activation pattern of a particular layer.
"""
function get_activation(nnet::Network, x::Vector{Float64})
    act_pattern = Vector{Vector{Bool}}(undef, length(nnet.layers))
    curr_value = x
    for (i, layer) in enumerate(nnet.layers)
        curr_value = affine_map(layer, curr_value)
        act_pattern[i] = get_activation(layer, curr_value)
        curr_value = layer.activation(curr_value)
    end
    return act_pattern
end

"""
    get_activation(nnet::Network, input::Hyperrectangle)

Given a network, find the activation pattern of all neurons for a given input set.
Assume ReLU.
return Vector{Vector{Int64}}.
- 1: activated
- 0: undetermined
- -1: not activated
"""
function get_activation(nnet::Network, input::Hyperrectangle)
    bounds = get_bounds(nnet, input)
    return get_activation(nnet, bounds)
end

"""
    get_activation(nnet::Network, bounds::Vector{Hyperrectangle})

Given a network, find the activation pattern of all neurons given the node-wise bounds.
Assume ReLU.
return Vector{Vector{Int64}}.
- 1: activated
- 0: undetermined
- -1: not activated
"""
function get_activation(nnet::Network, bounds::Vector{Hyperrectangle})
    act_pattern = Vector{Vector{Int}}(undef, length(nnet.layers))
    for (i, layer) in enumerate(nnet.layers)
        act_pattern[i] = get_activation(layer, bounds[i])
    end
    return act_pattern
end

function get_activation(L::Layer{ReLU}, bounds::Hyperrectangle)
    before_act_bound = approximate_affine_map(L, bounds)
    lower = low(before_act_bound)
    upper = high(before_act_bound)
    act_pattern = zeros(n_nodes(L))
    for j in 1:n_nodes(L) # For evey node
        if lower[j] > 0.0
            act_pattern[j] = 1
        elseif upper[j] < 0.0
            act_pattern[j] = -1
        end
    end
    return act_pattern
end

"""
    get_gradient(nnet::Network, x::Vector)

Given a network, find the gradient at the input x
"""
function get_gradient(nnet::Network, x::Vector)
    z = x
    gradient = Matrix(1.0I, length(x), length(x))
    for (i, layer) in enumerate(nnet.layers)
        z_hat = affine_map(layer, z)
        σ_gradient = act_gradient(layer.activation, z_hat)
        gradient = Diagonal(σ_gradient) * layer.weights * gradient
        z = layer.activation(z_hat)
    end
    return gradient
end

"""
    act_gradient(act::ReLU, z_hat::Vector{N}) where N

Computing the gradient of an activation function at point z_hat.
Currently only support ReLU and Id.
"""
act_gradient(act::ReLU, z_hat::Vector) = z_hat .>= 0.0
act_gradient(act::Id,   z_hat::Vector) = trues(length(z_hat))

"""
    get_gradient(nnet::Network, input::AbstractPolytope)

Get lower and upper bounds on network gradient for a given input set.
Return:
- `LG::Vector{Matrix}`: lower bounds
- `UG::Vector{Matrix}`: upper bounds
"""
function get_gradient(nnet::Network, input::AbstractPolytope)
    LΛ, UΛ = act_gradient_bounds(nnet, input)
    return get_gradient(nnet, LΛ, UΛ)
end

"""
    act_gradient_bounds(nnet::Network, input::AbstractPolytope)

Computing the bounds on the gradient of all activation functions given an input set.
Currently only support ReLU.
Return:
- `LΛ::Vector{Matrix}`: lower bounds
- `UΛ::Vector{Matrix}`: upper bounds
"""
function act_gradient_bounds(nnet::Network, input::AbstractPolytope)
    bounds = get_bounds(nnet, input)
    LΛ = Vector{Matrix}(undef, 0)
    UΛ = Vector{Matrix}(undef, 0)
    for (i, layer) in enumerate(nnet.layers)
        before_act_bound = approximate_affine_map(layer, bounds[i])
        lower = low(before_act_bound)
        upper = high(before_act_bound)
        l = act_gradient(layer.activation, lower)
        u = act_gradient(layer.activation, upper)
        push!(LΛ, Diagonal(l))
        push!(UΛ, Diagonal(u))
    end
    return (LΛ, UΛ)
end

"""
    get_gradient(nnet::Network, LΛ::Vector{Matrix}, UΛ::Vector{Matrix})

Get lower and upper bounds on network gradient for given gradient bounds on activations
Inputs:
- `LΛ::Vector{Matrix}`: lower bounds on activation gradients
- `UΛ::Vector{Matrix}`: upper bounds on activation gradients
Return:
- `LG::Vector{Matrix}`: lower bounds
- `UG::Vector{Matrix}`: upper bounds
"""
function get_gradient(nnet::Network, LΛ::Vector{Matrix}, UΛ::Vector{Matrix})
    n_input = size(nnet.layers[1].weights, 2)
    LG = Matrix(1.0I, n_input, n_input)
    UG = Matrix(1.0I, n_input, n_input)
    for (i, layer) in enumerate(nnet.layers)
        LG_hat, UG_hat = interval_map(layer.weights, LG, UG)
        LG = LΛ[i] * max.(LG_hat, 0) + UΛ[i] * min.(LG_hat, 0)
        UG = LΛ[i] * min.(UG_hat, 0) + UΛ[i] * max.(UG_hat, 0)
    end
    return (LG, UG)
end

"""
    get_gradient(nnet::Network, LΛ::Vector{Vector{N}}, UΛ::Vector{Vector{N}}) where N

Get lower and upper bounds on network gradient for given gradient bounds on activations
Inputs:
- `LΛ::Vector{Vector{N}}`: lower bounds on activation gradients
- `UΛ::Vector{Vector{N}}`: upper bounds on activation gradients
Return:
- `(LG, UG)` lower and upper bounds
"""
function get_gradient(nnet::Network, LΛ::Vector{Vector{N}}, UΛ::Vector{Vector{N}}) where N
    n_input = size(nnet.layers[1].weights, 2)
    LG = Matrix(1.0I, n_input, n_input)
    UG = Matrix(1.0I, n_input, n_input)
    for (i, layer) in enumerate(nnet.layers)
        LG_hat, UG_hat = interval_map(layer.weights, LG, UG)
        LG = Diagonal(LΛ[i]) * max.(LG_hat, 0) + Diagonal(UΛ[i]) * min.(LG_hat, 0)
        UG = Diagonal(LΛ[i]) * min.(UG_hat, 0) + Diagonal(UΛ[i]) * max.(UG_hat, 0)
    end
    return (LG, UG)
end

"""
    interval_map(W::Matrix, l, u)

Simple linear mapping on intervals
Inputs:
- `W::Matrix{N}`: linear mapping
- `l::Vector{N}`: lower bound
- `u::Vector{N}`: upper bound
Outputs:
- `(lbound, ubound)` (after the mapping)
"""
function interval_map(W::Matrix{N}, l::AbstractVecOrMat, u::AbstractVecOrMat) where N
    l_new = max.(W, zero(N)) * l + min.(W, zero(N)) * u
    u_new = max.(W, zero(N)) * u + min.(W, zero(N)) * l
    return (l_new, u_new)
end

"""
    get_bounds(problem::Problem)
    get_bounds(nnet::Network, input::Hyperrectangle)

This function calls maxSens to compute node-wise bounds given a input set.

Return:
- `Vector{Hyperrectangle}`: bounds for all nodes **after** activation. `bounds[1]` is the input set.
"""
function get_bounds(nnet::Network, input::Hyperrectangle, act::Bool = true) # NOTE there is another function by the same name in convDual. Should reconsider dispatch
    if act
        solver = MaxSens(0.0, true)
        bounds = Vector{Hyperrectangle}(undef, length(nnet.layers) + 1)
        bounds[1] = input
        for (i, layer) in enumerate(nnet.layers)
            bounds[i+1] = forward_layer(solver, layer, bounds[i])
        end
        return bounds
    else
       error("before activation bounds not supported yet.")
    end
end
get_bounds(problem::Problem) = get_bounds(problem.network, problem.input)

"""
    affine_map(layer, input::AbstractPolytope)

Affine transformation of a set using the weights and bias of a layer.

Inputs:
- `layer`: Layer
- `input`: input set (Hyperrectangle, HPolytope)
Return:
- `output`: set after transformation.


    affine_map(layer, input)

Inputs:
- `layer`: Layer
- `input`: Vector
Return:
- `output`: Vector after mapping
"""
affine_map(layer::Layer, input) = layer.weights*input + layer.bias
function affine_map(layer::Layer, input::AbstractPolytope)
    W, b = layer.weights, layer.bias
    return translate(b, linear_map(W, input))
end

"""
   approximate_affine_map(layer, input::Hyperrectangle)

Returns a Hyperrectangle overapproximation of the affine map of the input.
"""
function approximate_affine_map(layer::Layer, input::Hyperrectangle)
    c = affine_map(layer, input.center)
    r = abs.(layer.weights) * input.radius
    return Hyperrectangle(c, r)
end

function translate(v::Vector, H::HPolytope)
    # translate each halfpsace according to:
    # a⋅(x-v) ≤ b  ⟶  a⋅x ≤ b+a⋅v
    C, d = tosimplehrep(H)
    return HPolytope(C, d+C*v)
end
# translate(v::Vector, H::Hyperrectangle)   = Hyperrectangle(H.center .+ v, H.radius)
translate(v::Vector, V::AbstractPolytope) = tohrep(VPolytope([x+v for x in vertices_list(V)]))

"""
    split_interval(dom, i)

Split a set into two at the given index.

Inputs:
- `dom::Hyperrectangle`: the set to be split
- `i`: the index to split at
Return:
- `(left, right)::Tuple{Hyperrectangle, Hyperrectangle}`: two sets after split
"""
function split_interval(dom::Hyperrectangle, i::Int64)
    input_lower, input_upper = low(dom), high(dom)

    input_upper[i] = dom.center[i]
    input_split_left = Hyperrectangle(low = input_lower, high = input_upper)

    input_lower[i] = dom.center[i]
    input_upper[i] = dom.center[i] + dom.radius[i]
    input_split_right = Hyperrectangle(low = input_lower, high = input_upper)
    return (input_split_left, input_split_right)
end
