#=
Make benchmark_maker.jl to take in a config file that specifies the benchmarks to create
and outputs a Benchmarks.txt file.

We will take in a base directory, and will create:

BaseDirectory
    Networks (folder with network files)
    Properties (folder with property files)
    Benchmarks.txt
    [Queries.txt will be added by ExperimentMaker]

Each line in Benchmarks.txt consists of:

network.nnet property.txt

How to run in REPL:

module test
       ARGS = ["--config_file", "/Users/castrong/Desktop/Research/NeuralOptimization.jl/ConfigFiles/BenchmarkConfigs/test_benchmark_maker.yaml"]
       include("/Users/castrong/Desktop/Research/NeuralOptimization.jl/src/utils/BenchmarkUtils/BenchmarkMaker.jl")
end
=#

using Pkg
# Interface:
# BenchmarkUtils/BenchmarkMaker.jl --config_file path/to/file/my_file.yaml
using ArgParse
using YAML
arg_settings = ArgParseSettings()
@add_arg_table! arg_settings begin
    "--config_file"
        help = "Config .yaml file which describes the benchmarks to create"
        arg_type = String
end
# Parse your arguments
parsed_args = parse_args(ARGS, arg_settings)
yaml_file = parsed_args["config_file"]
println(yaml_file)

config = YAML.load(open(yaml_file))

root_dir = config["global"]["root_dir"]
Pkg.activate(root_dir)

using Random
using NPZ

# Define and create the Networks and Properties folder
output_path = config["global"]["output_path"]
networks_output_path = joinpath(output_path, "Networks")
properties_output_path = joinpath(output_path, "Properties")
benchmarks_file = joinpath(output_path, "Benchmarks.txt")
mkpath.([output_path, networks_output_path, properties_output_path])

function write_property_file_from_image(input_image_file::String, epsilon::Float64, coefficients::Array{Float64, 1}, variables::Array{Int64, 1}, maximize::Bool, output_file::String)
    input_image = npzread(input_image_file)
    open(output_file, "w") do f
        # Write the lower and upper bounds on each pixel
        for (index, x_0) in enumerate(input_image)
            # Property file specification indexes from 0, so substract one off the index.
            println(f, "x", index - 1, " >= ", x_0 - epsilon)
            println(f, "x", index - 1, " <= ", x_0 + epsilon)
        end

        # Print the objective
        print(f, maximize ? "Maximize " : "Minimize ")
        for (coeff_index, (coeff, variable)) in enumerate(zip(coefficients, variables))
            print(f, coeff, "y", variable-1) # Index - 1 since property file specification indexes from 0
            # Don't print the "+" after the last coefficient
            if (coeff_index < length(coefficients))
                print(f, " + ")
            end
        end
    end
end

function write_property_file_minadv(input_image_file::String, lower::Float64, upper::Float64, target::Int, num_outputs::Int, output_file::String)
    input_image = npzread(input_image_file)
    open(output_file, "w") do f
        # Write the lower and upper bounds on each pixel
        for (index, ~) in enumerate(input_image)
            # Property file specification indexes from 0, so substract one off the index.
            println(f, "x", index - 1, " >= ", lower)
            println(f, "x", index - 1, " <= ", upper)
        end

        # Write the output constraints
        for out_index in 1:num_outputs
            if (out_index != target)
                println(f, "y", target - 1, " >= ", "y", out_index - 1)
            end
        end

        # Write the center - the input_image_file
        println(f, "Center ", string(input_image)[2:end-1]) # 2 --> end-1 to avoid the brackets

        # Write out the target
        println(f, "Target ", target - 1, ", ", "max")

        # Write the dimensions to optimize over
        println(f, "Minimum input perturbation all")
    end
end


#=
    Check for and process ACAS

=#
if haskey(config, "acas")
    acas_config = config["acas"]
    network_dir = acas_config["network_dir"]
    property_dir = acas_config["property_dir"]

    if haskey(config["acas"], "output")
        # Load in information for creating acas query
        acas_output_config = config["acas"]["output"]
        properties = split(acas_output_config["properties"], " ")
        number_of_networks = parse.(Int, split(acas_output_config["number_of_networks"], " "))
        println("Properties: ", properties)

        for (property, cur_num_networks) in zip(properties, number_of_networks)
            # Find the property to copy
            property_name = string("acas_property_optimization_", property, ".txt")
            property_file = joinpath(root_dir, property_dir, property_name)

            # Copy to our property file
            property_output_file = joinpath(properties_output_path, property_name)
            cp(property_file, property_output_file, force=true) # will write several times if multiple properties

            cur_count = 0
            # Copy over the network files
            for i = 1:5
                for j = 1:9
                    network_name = string("ACASXU_experimental_v2a_", i, "_", j, ".nnet")
                    network_file = joinpath(root_dir, network_dir, network_name)
                    network_output_file = joinpath(networks_output_path, network_name)
                    cp(network_file, network_output_file, force=true) # will write several times if multiple properties

                    # Add a line to your benchmark file
                    open(benchmarks_file, "a") do f
                        println(f, network_output_file, " ", property_output_file)
                    end

                    cur_count = cur_count + 1
                    if (cur_count >= cur_num_networks)
                        break
                    end
                end
                if (cur_count >= cur_num_networks)
                    break
                end
            end
        end
    end
    if (haskey(config["acas"], "min_input"))
        acas_input_config = config["acas"]["min_input"]
        properties = split(acas_input_config["properties"], " ")
        number_of_networks = parse.(Int, split(acas_input_config["number_of_networks"], " "))
        dims = split(acas_input_config["dims"], " ")
        targets = parse.(Int, split(acas_input_config["target"], " "))
        target_dirs = split(acas_input_config["target_dir"], " ")

        for (property, cur_num_networks) in zip(properties, number_of_networks)
            # Find the property to copy
            property_name = string("acas_property_optimization_", property, ".txt")
            property_file = joinpath(root_dir, property_dir, property_name)

            for (dim_list, target, target_dir) in zip(dims, targets, target_dirs)
                property_lines = readlines(property_file)
                # Add in output constraints
                for out_index = 1:5
                    if out_index != target
                        push!(property_lines, string("y", target - 1, target_dir=="max" ? " >= " : " <= ", "y", out_index - 1))
                    end
                end

                # Write out the target
                push!(property_lines, string("Target ", target - 1, ", ", target_dir))

                # remove the objective
                filter!(line -> !(occursin("Maximize", line) || occursin("Minimize", line)), property_lines)
                # Add the minimum input perturbation dimensions
                if (dim_list == "all")
                    dim_str = "all"
                else
                    dim_str = join(string.(parse.(Int64, split(dim_list, ",")) .- 1), ",") # adjust to index from 0
                end
                push!(property_lines, "Minimum Input Perturbation "*dim_str)

                property_output_file = joinpath(properties_output_path, string("acas_property_", property, "_mininput_", dim_list, "_target_", target, "_maxtarget_", target_dir, ".txt"))
                # Write to file
                open(property_output_file, "w") do f
                    write(f, join(property_lines, "\n"))
                end

                cur_count = 0
                # Copy over the network files
                for i = 1:5
                    for j = 1:9
                        network_name = string("ACASXU_experimental_v2a_", i, "_", j, ".nnet")
                        network_file = joinpath(root_dir, network_dir, network_name)
                        network_output_file = joinpath(networks_output_path, network_name)
                        cp(network_file, network_output_file, force=true) # will write several times if multiple properties

                        # Add a line to your benchmark file
                        open(benchmarks_file, "a") do f
                            println(f, network_output_file, " ", property_output_file)
                        end

                        cur_count = cur_count + 1
                        if (cur_count >= cur_num_networks)
                            break
                        end
                    end
                    if (cur_count >= cur_num_networks)
                        break
                    end
                end
            end
        end
    end
end


#=
    Check for and process MNIST
=#

label_to_target = Dict([0 => 6, 1 => 7, 2 => 3, 3 => 2, 4 => 9, 5 => 8, 6 => 0, 7 => 1, 8 => 5, 9 => 4])

if haskey(config, "mnist")
    mnist_config = config["mnist"]
    network_dir = mnist_config["network_dir"]
    input_image_dir = mnist_config["input_image_dir"]
    start_file_name = mnist_config["start_file_name"]

    if (haskey(config["mnist"], "output"))
        mnist_output_config = config["mnist"]["output"]
        architectures = split(mnist_output_config["architectures"], " ")
        number_of_images = parse.(Int, split(mnist_output_config["number_of_images"], " "))
        epsilons = parse.(Float64, split(mnist_output_config["epsilons"], " "))

        # Find the possible inputs
        files = filter(f->startswith(f, start_file_name), readdir(input_image_dir))
        # Shuffle files to choose random set
        shuffle!(files)

        for (architecture, num_images) in zip(architectures, number_of_images)
            # Copy over the network file
            network_name = string("mnist", architecture, ".nnet")
            network_file = joinpath(root_dir, network_dir, network_name)
            network_output_file = joinpath(networks_output_path, network_name)
            cp(network_file, network_output_file, force=true)

            for image_index = 1:num_images
                cur_image_file = files[image_index]
                cur_image_file_noext, ext = splitext(cur_image_file)
                cur_label = parse(Int64, split(cur_image_file_noext, "_")[2])
                cur_target = label_to_target[cur_label]

                for epsilon in epsilons
                    # Create property file from the image
                    property_name = string("mnist_property_", cur_image_file_noext, "_", epsilon, ".txt")
                    property_file = joinpath(properties_output_path, property_name)
                    # Pass in target and label, switch to indexing from 1 instead of 0
                    # (the filename indexes from 0)
                    write_property_file_from_image(joinpath(input_image_dir, cur_image_file), epsilon, [1.0, -1.0], [cur_target+1, cur_label+1], true, property_file)

                    # Add a line to your benchmark file
                    open(benchmarks_file, "a") do f
                        println(f, network_output_file, " ", property_file)
                    end
                end
            end
        end
    end
    if(haskey(config["mnist"], "min_input"))
        println("Making min input for mnist")
        mnist_input_config = config["mnist"]["min_input"]
        architectures = split(mnist_output_config["architectures"], " ")
        number_of_images = parse.(Int, split(mnist_output_config["number_of_images"], " "))

        # Find the possible inputs
        files = filter(f->startswith(f, start_file_name), readdir(input_image_dir))
        # Shuffle files to choose random set
        shuffle!(files)

        for (architecture, num_images) in zip(architectures, number_of_images)
            # Copy over the network file
            network_name = string("mnist", architecture, ".nnet")
            network_file = joinpath(root_dir, network_dir, network_name)
            network_output_file = joinpath(networks_output_path, network_name)
            cp(network_file, network_output_file, force=true)

            for image_index = 1:num_images
                cur_image_file = files[image_index]
                cur_image_file_noext, ext = splitext(cur_image_file)
                cur_label = parse(Int64, split(cur_image_file_noext, "_")[2])
                cur_target = label_to_target[cur_label]
                property_name = string("mnist_property_mininput_", cur_image_file_noext, "_", cur_target, ".txt")
                property_file = joinpath(properties_output_path, property_name)
                write_property_file_minadv(joinpath(input_image_dir, cur_image_file), 0.0, 1.0, cur_target+1, 10, property_file)

                # Add a line to oyour benchmark file
                open(benchmarks_file, "a") do f
                    println(f, network_output_file, " ", property_file)
                end
            end
        end
    end
end

#=
    Check for and process Taxinet
=#

if haskey(config, "taxinet")
    taxinet_config = config["taxinet"]
    network_dir = taxinet_config["network_dir"]
    architectures = split(taxinet_config["architectures"], " ")
    input_image_dir = taxinet_config["input_image_dir"]
    start_file_name = taxinet_config["start_file_name"]
    number_of_images = parse.(Int, split(taxinet_config["number_of_images"], " "))
    epsilons = parse.(Float64, split(taxinet_config["epsilons"], " "))
    maximize_output = taxinet_config["maximize_output"]
    minimize_output = taxinet_config["minimize_output"]

    # Find the possible inputs
    files = filter(f->startswith(f, start_file_name), readdir(input_image_dir))
    # Shuffle files to choose random set
    shuffle!(files)

    for (architecture, num_images) in zip(architectures, number_of_images)
        # Copy over the network file
        network_name = string("AutoTaxi_", architecture, "Relus_200Epochs_OneOutput.nnet")
        network_file = joinpath(root_dir, network_dir, network_name)
        network_output_file = joinpath(networks_output_path, network_name)
        cp(network_file, network_output_file, force=true)

        for image_index = 1:num_images
            cur_image_file = files[image_index]
            cur_image_file_noext, ext = splitext(cur_image_file)

            for epsilon in epsilons
                # Create property file from the image
                property_name_max = string("autotaxi_property_", cur_image_file_noext, "_", epsilon, "_max", ".txt")
                property_file_max = joinpath(properties_output_path, property_name_max)
                property_name_min = string("autotaxi_property_", cur_image_file_noext, "_", epsilon, "_min", ".txt")
                property_file_min = joinpath(properties_output_path, property_name_min)

                if (maximize_output)
                    write_property_file_from_image(joinpath(input_image_dir, cur_image_file), epsilon, [1.0], [1], true, property_file_max)

                    # Add a line to your benchmark file
                    open(benchmarks_file, "a") do f
                        println(f, network_output_file, " ", property_file_max)
                    end
                end
                if (minimize_output)
                    # Encode minimizing as maximizing the negative output
                    write_property_file_from_image(joinpath(input_image_dir, cur_image_file), epsilon, [-1.0], [1], true, property_file_min)

                    # Add a line to your benchmark file
                    open(benchmarks_file, "a") do f
                        println(f, network_file, " ", property_file_min)
                    end
                end
            end
        end
    end
end

# Copy config file to the output folder at the end once it has all run
config_path, config_name = splitdir(yaml_file)
cp(yaml_file, joinpath(output_path, config_name))
