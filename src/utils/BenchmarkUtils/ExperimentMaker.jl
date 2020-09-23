#=
ExperimentMaker.jl takes in a config file that specifies a benchmark file
as well as the solvers that we'd like to run it with

We will take in a base directory which is assumed to look like:

BaseDirectory
    Networks (folder with network files)
    Properties (folder with property files)
    Benchmarks.txt
    [Queries.txt added by this script]

And we will add Queries.txt to this directory. This will
take each benchmark and append each solver to it.

Each line in Query.txt consists of:

solver_description network.nnet property.txt output_file.txt

How to run in REPL:

module test
       ARGS = ["--config_file", "/Users/castrong/Desktop/Research/NeuralOptimization.jl/ConfigFiles/BenchmarkConfigs/test_experiment_maker.yaml"]
       include("/Users/castrong/Desktop/Research/NeuralOptimization.jl/src/utils/BenchmarkUtils/ExperimentMaker.jl")
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
        help = "Config .yaml file which describes the solvers to use and the benchmark file to build off of"
        arg_type = String
end
# Parse your arguments
parsed_args = parse_args(ARGS, arg_settings)
yaml_file = parsed_args["config_file"]
println(yaml_file)

config = YAML.load(open(yaml_file))

root_dir = config["global"]["root_dir"]
Pkg.activate(root_dir)

include("$(@__DIR__)/../../NeuralOptimization.jl")
using GLPK
using Gurobi

# Define the results folder
output_path = config["global"]["output_path"]
results_path = joinpath(output_path, "Results")
benchmark_file = joinpath(output_path, "Benchmarks.txt") # assume a benchmark file will be located there
query_file = joinpath(output_path, string("benchmark_set_", config["global"]["query_file_name"], ".txt")) # prepend with benchmark set for the cluster's convention
@assert isfile(benchmark_file)
mkpath(results_path)

# Copy config file to the output folder
config_path, config_name = splitdir(yaml_file)
cp(yaml_file, joinpath(output_path, config_name))

solvers = []
if haskey(config, "fgsm")
    push!(solvers, NeuralOptimization.FGSM())
end

if haskey(config, "lbfgs")
    push!(solvers, NeuralOptimization.LBFGS())
end

if haskey(config, "pgd")
    push!(solvers, NeuralOptimization.PGD())
end

if haskey(config, "marabou")
    usesbts = parse.(Bool, split(string(config["marabou"]["usesbt"]), ","))
    dividestrategies = split(config["marabou"]["dividestrategy"], ",")
    perReLUTimeouts = parse.(Float64, split(config["marabou"]["perReLUTimeout"], ","))
    for (usesbt, dividestrategy, perReLUTimeout) in zip(usesbts, dividestrategies, perReLUTimeouts)
        push!(solvers, NeuralOptimization.Marabou(usesbt=usesbt, dividestrategy=dividestrategy, perReLUTimeout=perReLUTimeout))
    end
end

if haskey(config, "marabou_binary_search")
    usesbts = parse.(Bool, split(string(config["marabou_binary_search"]["usesbt"]), ","))
    dividestrategies = split(config["marabou_binary_search"]["dividestrategy"], ",")
    perReLUTimeouts = parse.(Float64, split(config["marabou_binary_search"]["perReLUTimeout"], ","))
    for (usesbt, dividestrategy, perReLUTimeout) in zip(usesbts, dividestrategies, perReLUTimeouts)
        push!(solvers, NeuralOptimization.MarabouBinarySearch(usesbt=usesbt, dividestrategy=dividestrategy, perReLUTimeout=perReLUTimeout))
    end
end

if haskey(config, "mipverify")

    optimizer_strings = split(config["mipverify"]["optimizer"], ",")
    threadss = parse.(Int64, split(string(config["mipverify"]["threads"]), ","))
    strategies = split(string(config["mipverify"]["strategy"]), ",")
    preprocess_timeout_per_nodes = parse.(Float64, split(config["mipverify"]["preprocess_timeout_per_node"], ","))

    for (optimizer_string, threads, strategy, preprocess_timeout) in zip(optimizer_strings, threadss, strategies, preprocess_timeout_per_nodes)
        if (optimizer_string == "glpk")
            optimizer = GLPK.Optimizer
        elseif (optimizer_string == "gurobi")
            optimizer = Gurobi.Optimizer
        else
            @assert false "Unrecognized optimizer for Sherlock"
        end
        push!(solvers, NeuralOptimization.MIPVerify(optimizer=optimizer, threads=threads, strategy=strategy, preprocess_timeout_per_node=preprocess_timeout))
    end
end

if haskey(config, "sherlock")
    ms = parse.(Float64, split(string(config["sherlock"]["m"]), ","))
    optimizer_strings = split(config["sherlock"]["optimizer"], ",")

    output_flags = parse.(Int64, split(string(config["sherlock"]["output_flag"]), ","))
    threadss = parse.(Int64, split(string(config["sherlock"]["threads"]), ","))

    for (m, optimizer_string, output_flag, threads) in zip(ms, optimizer_strings, output_flags, threadss)
        if (optimizer_string == "glpk")
            optimizer = GLPK.Optimizer
        elseif (optimizer_string == "gurobi")
            optimizer = Gurobi.Optimizer
        else
            @assert false "Unrecognized optimizer for Sherlock"
        end
        push!(solvers, NeuralOptimization.Sherlock(m=m, optimizer=optimizer, output_flag=output_flag, threads=threads))
    end
end

if haskey(config, "vanillamip")
    ms = parse.(Float64, split(string(config["vanillamip"]["m"]), ","))
    optimizer_strings = split(config["vanillamip"]["optimizer"], ",")

    output_flags = parse.(Int64, split(string(config["vanillamip"]["output_flag"]), ","))
    threadss = parse.(Int64, split(string(config["vanillamip"]["threads"]), ","))

    for (m, optimizer_string, output_flag, threads) in zip(ms, optimizer_strings, output_flags, threadss)
        if (optimizer_string == "glpk")
            optimizer = GLPK.Optimizer
        elseif (optimizer_string == "gurobi")
            optimizer = Gurobi.Optimizer
        else
            @assert false "Unrecognized optimizer for Sherlock"
        end
        push!(solvers, NeuralOptimization.VanillaMIP(m=m, optimizer=optimizer, output_flag=output_flag, threads=threads))
    end
end

println(solvers)

# For each solver run each of the benchmarks in
benchmark_lines = readlines(benchmark_file)
for solver in solvers
    for benchmark in benchmark_lines
        # Get the name of the network and property from the benchmark to use in the output file name
        network_file, property_file = split(benchmark, " ")
        network_path, network_name = splitdir(network_file)
        network_name_no_ext, ext = splitext(network_name)
        property_path, property_name = splitdir(property_file)
        property_name_no_ext, ext = splitext(property_name)

        # Create the result file, then write out the solver, benchmark, and result_file to query_file
        result_file = string(solver, ".", network_name_no_ext, ".", property_name_no_ext, ".txt")
        result_file = joinpath(results_path, result_file)
        query_line = string("--environment_path ", root_dir, " --optimizer ", solver, " --network_file ", network_file, " --property_file ", property_file, " --result_file ", result_file)

        open(query_file, "a") do f
            println(f, query_line)
        end
    end
end
