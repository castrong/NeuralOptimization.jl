include("../../NeuralOptimization.jl")
include("../CreateAllOptimizers.jl")
comma_replacement = "[-]" # has to match with the comma replacement in BenchmarkFileWriters.jl!!!

#=
Run a set of queries given by a query file. This just consists of a loop
that passes on the strings to RunQuery.jl

We frame it as running an external command to make it easier to transition
to running on a cluster where this command would get more complicated
=#
function run_query_file(query_file)
    queries = readlines(query_file)
    for query in queries
        args = split(query, ",")
        # 11 args
        # optimizer_name,  optimizer, class, network_file, input_file, delta, objective_variables, objective_coefficients, maximize, query_output_filename, timeout
        mycommand = `julia ./src/utils/BenchmarkUtils/RunQuery.jl $(args[1]) $(args[2]) $(args[3]) $(args[4]) $(args[5]) $(args[6]) $(args[7]) $(args[8]) $(args[9]) $(args[10]) 60`
        run(mycommand)
    end
end


query_file ="/Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/BenchmarkOutput/benchmark_files/test_queries.csv"
run_query_file(query_file)
