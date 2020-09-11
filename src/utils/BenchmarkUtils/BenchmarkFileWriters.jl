include("../../NeuralOptimization.jl")
include("../CreateAllOptimizers.jl")

comma_replacement = "[-]"

#=
Write a file where each line corresponds to the description of a problem to run.
This does not include the optimizer to run it on. The format is:

Format:
class, network_file, center, input_file, delta_list, objective_variables, objective_coefficients, maximize, timeout

The delta_list must be the length of the input (so that it can specify a hyperrectangle with different radius for
each input
=#
function write_benchmark_file(class_string, network_files, input_files, objective_functions, deltas, maximize, output_file, append_output=false)
    # Make the path to your output file if it doesnt exist
    mkpath(dirname(output_file))
    first_loop = true

    for network_file in network_files
        for (input_file, objective_function) in zip(input_files, objective_functions)
            for delta_list in deltas
                # Write the current setup to the output file
                # with this setup, we'll append if the append_output
                # flag is set to be true
                file_mode = (first_loop && !append_output) ? "w" : "a"
                first_loop = false
                open(output_file, file_mode) do f
                    # Writeout each column separated by a comma
                    # for each of the lists, we replace their commas and remove all spaces
                    delta_list_string = replace(string(delta_list), ","=>comma_replacement)
                    delta_list_string = replace(delta_list_string, " "=>"")
                    objective_variables_string = replace(string(objective_function.variables), ","=>comma_replacement)
                    objective_variables_string = replace(objective_variables_string, " "=>"")
                    objective_coefficients_string = replace(string(objective_function.coefficients), ","=>comma_replacement)
                    objective_coefficients_string = replace(objective_coefficients_string, " "=>"")
                    write(f,
                          class_string, ",",
                          network_file, ",",
                          input_file, ",",
                          delta_list_string, ",",
                          objective_variables_string, ",",
                          objective_coefficients_string, ",",
                          maximize ? "maximize" : "minimize", "\n"
                          )
                end
            end
        end
    end
end

#=
Write a file describing which optimizers to use in a benchmark.
This optimizer file combined with any benchmark file can be used
to create a query file, where each line will correspond to a complete
query that can then be run.

Each line corresponds to an optimizer_name, optimizer_string
=#
function write_optimizer_file(optimizer_names, optimizers, output_file, append_output=false)
    # Make the path to your output file if it doesnt exist
    mkpath(dirname(output_file))
    first_loop = true

    for (optimizer_name, optimizer) in zip(optimizer_names, optimizers)
        file_mode = (first_loop && !append_output) ? "w" : "a"
        first_loop = false
        open(output_file, file_mode) do f
            write(f, optimizer_name, ",", string(optimizer), "\n")
        end
    end
end

#=
Write a file where each line corresponds to a query to run.
optimizer_name, optimizer, class, network_file, center_input_file, delta, objective_variables, objective_coefficients, maximize, query_output_filename
=#
function write_query_file(benchmark_file, optimizer_file, query_result_path, output_file, append_output=false)
    # Make the path to your output file if it doesnt exist
    mkpath(dirname(output_file))

    # Combine each optimizer with each benchmark query
    optimizer_strings = readlines(optimizer_file)
    benchmark_strings = readlines(benchmark_file)
    combined_strings = vec([string(optimizer, ",", benchmark) for optimizer in optimizer_strings, benchmark in benchmark_strings])

    # Write out your queries - one on each line
    first_loop = true
    for cur_query in combined_strings
        file_mode = (first_loop && !append_output) ? "w" : "a"
        first_loop = false
        # For each query create an output filename where it will write its results
        # remove the full paths and replace them with basename
        query_parts = split(cur_query, ",")
        query_parts[4] = basename(query_parts[4])[1:end-5] # find the network basename to use in our output file
        query_parts[5] = basename(query_parts[5])[1:end-4] # TODO: Make this more rigorous - find the input file basename to use in our output file, remove file extension
        query_parts[6] = split(query_parts[6][2:end], comma_replacement)[1] # TODO: How to deal with names of deltas? replace list of deltas with the first delta for now
        deleteat!(query_parts, [2]) # remove optimizer full description from filename, can use the optimizer name
        query_output_filename = string(query_result_path, join(query_parts, "-"))

        open(output_file, file_mode) do f
            write(f, cur_query, ",", query_output_filename, "\n")
        end
    end
end


### Write a benchmark file
class_string = "MNIST"
network_files = ["/Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/Networks/MNIST/mnist10x10.nnet", "/Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/Networks/MNIST/mnist10x20.nnet"]
input_files =["/Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/Datasets/MNIST/MNISTlabel_0_index_0_.npy", "/Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/Datasets/MNIST/MNISTlabel_0_index_0_.npy"]
objective_one = NeuralOptimization.LinearObjective([1.0, -1.0], [1, 3]) # objective is to just maximize the first output
objective_two = NeuralOptimization.LinearObjective([1.0, -1.0], [2, 3]) # objective is to just maximize the first output
objective_functions = [objective_one, objective_two]

deltas = [.001 * ones(784), 0.0005 * ones(784)] # both hypercubes for now
maximize = true
output_file_benchmark = "/Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/BenchmarkOutput/benchmark_files/test_benchmark.csv"
append_to_file = false

println(string.(objective_functions))
write_benchmark_file(class_string, network_files, input_files, objective_functions, deltas, maximize, output_file_benchmark, append_to_file)


### Write an optimizer file
output_file_optimizers = "/Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/BenchmarkOutput/benchmark_files/test_optimizers.csv"
append_to_file = false
optimizers = [
              LBFGS_optimizer,
              PGD_optimizer,
              FGSM_optimizer,
              Marabou_optimizer_ReLUViolation,
              Marabou_optimizer_sbt_ReLUViolation,
              Marabou_optimizer_earliestReLU,
              Marabou_optimizer_sbt_earliestReLU,
              MarabouBinary_optimizer_ReLUViolation,
              MarabouBinary_optimizer_sbt_ReLUViolation,
              MarabouBinary_optimizer_earliestReLU,
              MarabouBinary_optimizer_sbt_earliestReLU,
              VanillaMIP_Gurobi_optimizer_8threads,
              VanillaMIP_Gurobi_optimizer_1thread,
              VanillaMIP_GLPK_optimizer,
              Sherlock_Gurobi_optimizer_8threads,
              Sherlock_Gurobi_optimizer_1thread,
              Sherlock_GLPK_optimizer,
              ]
optimizer_names = ["LBFGS",
                    "PGD",
                    "FGSM",
                    "MarabouReLUViolation",
                    "MarabouSbtReLUViolation",
                    "MarabouEarliestReLU",
                    "MarabouSbtEarliestReLU",
                    "MarabouBinaryReLUViolation",
                    "MarabouBinarySBTReLUViolation",
                    "MarabouBinaryEarliestReLU",
                    "MarabouBinarySBTEarliestReLU",
                    "VanillaGurobi8Threads",
                    "VanillaGurobi1Thread",
                    "VanillaGLPK",
                    "SherlockGurobi8Threads",
                    "SherlockGurobi1Thread",
                    "SherlockGLPK"]
write_optimizer_file(optimizer_names, optimizers, output_file_optimizers, append_to_file)

# Write a query file - this will have each of the queries we'd like to run
append_output = false
benchmark_file = output_file_benchmark
optimizer_file = output_file_optimizers
output_file_query ="/Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/BenchmarkOutput/benchmark_files/test_queries.csv"
query_result_path = "/Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/BenchmarkOutput/benchmark_files/results/"
write_query_file(benchmark_file, optimizer_file, query_result_path, output_file_query, append_output)
