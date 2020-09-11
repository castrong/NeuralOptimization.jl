ENV["JULIA_DEBUG"] = Main # turns on logging (@debug, @info, @warn) for "included" files

include("../../NeuralOptimization.jl")
include("../CreateAllOptimizers.jl")
using NPZ

# Benchmark a given type of network (e.g. ACASxu, MNIST)
# for each solver it will run queries on each network.
# Each input file will have a corresponding objective
# this objective will be run with its input file for each delta
function Benchmark_Category(optimizers, network_files, objectives, input_files, deltas, output_file, maximize=false, transpose_inputs=true, append_output=true, timeout=3600)
    times = []
    first_loop = true # if creating a new file, use this to create the file only on the first loop
    for network_file in network_files
            for (input_file, objective) in zip(input_files, objectives)
                for delta in deltas
                    for optimizer in optimizers
                    println("Starting Query")
                    println("    Network: ", basename(network_file))
                    println("    Input file: ", basename(input_file))
                    println("    Objective: ", objective)
                    println("    Delta: ", delta)
                    println("    Optimizer: ", optimizer)

                    """
                    Setup problem and run query
                    """
                    # Read in the network
                    network = NeuralOptimization.read_nnet(network_file)
                    num_inputs = size(network.layers[1].weights, 2)

                    # Create the problem: network, input constraint, objective, maximize or minimize
                    center_input = transpose_inputs ? transpose(npzread(input_file)) : npzread(input_file)
                    input = NeuralOptimization.Hyperrectangle(vec(center_input)[:], delta * ones(num_inputs)) # center and radius
                    problem = NeuralOptimization.OutputOptimizationProblem(network=network, input=input, objective=objective, max=maximize)
                    elapsed_time = @elapsed result = NeuralOptimization.optimize(optimizer, problem, timeout)
                    println("Result objective value: ", result.objective_value)

                    """
                    Write results to file
                    """
                    # Make the path to your output file if it doesnt exist
                    mkpath(dirname(output_file))
                    # if its your first time and you're overwriting, set "w". Otherwise, open in append mode
                    file_mode = (first_loop && !append_output) ? "w" : "a"
                    first_loop = false
                    open(output_file, file_mode) do f
                        # Write the column headers if we're creating a new file
                        if (file_mode == "w")
                            write(f, "network,inputfile,objective,delta,optimizer,status,optval,time\n")
                        end
                        # Writeout each column separated by a comma
                        write(f,
                              basename(network_file), ",",
                              basename(input_file), ",",
                              string(objective), ",",
                              string(delta), ",",
                              string(optimizer), ",",
                              string(result.status), ",",
                              string(result.objective_value), ",",
                              string(elapsed_time), "\n")
                    end
                end
            end
        end
    end
end


optimizers = [
#              LBFGS_optimizer,
#              PGD_optimizer,
#              FGSM_optimizer,
#              Marabou_optimizer_ReLUViolation,
              Marabou_optimizer_sbt_ReLUViolation,
#              Marabou_optimizer_earliestReLU,
              Marabou_optimizer_sbt_earliestReLU,
#              MarabouBinary_optimizer_ReLUViolation,
#              MarabouBinary_optimizer_sbt_ReLUViolation,
#              MarabouBinary_optimizer_earliestReLU,
#              MarabouBinary_optimizer_sbt_earliestReLU,
              VanillaMIP_Gurobi_optimizer_8threads,
#              VanillaMIP_Gurobi_optimizer_1thread,
#              VanillaMIP_GLPK_optimizer,
              Sherlock_Gurobi_optimizer_8threads,
#              Sherlock_Gurobi_optimizer_1thread,
#              Sherlock_GLPK_optimizer,
              ]

objective = NeuralOptimization.LinearObjective([1.0,], [1]) # objective is to just maximize the first output
network_files = ["Networks/AutoTaxi/AutoTaxi_128Relus_200Epochs_OneOutput.nnet"]
objectives = [objective] # an objective for each input file
input_files = ["./Datasets/AutoTaxi/AutoTaxi_12345.npy"]
deltas = [0.01]
transpose_inputs = true
maximize = true
append_output = true
timeout = 300
output_file = "./BenchmarkOutput/temp.csv"

center_input = npzread(input_files[1])
network = NeuralOptimization.read_nnet(network_files[1])
println("Center objective: ", NeuralOptimization.compute_objective(network, vec(transpose(center_input)), objective))

Benchmark_Category(optimizers,
                   network_files,
                   objectives,
                   input_files,
                   deltas,
                   output_file,
                   maximize,
                   transpose_inputs,
                   append_output,
                   timeout)
