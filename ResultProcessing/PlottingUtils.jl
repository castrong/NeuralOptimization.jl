using DataFrames
using CSV
using PGFPlots
using Query

function make_pairwise_comparisons(filename, output_path, column_markers, row_markers, categories, legend_entries, styles; file_start="", timeout=3600)
    full_style = join([string(category, "={", style, "}, ") for (category, style) in zip(categories, styles)])[1:end-2] # remove last comma
    mkpath(output_path) # make the output path if it doesn't exist already

    # Read in the data
    data = CSV.read(filename)

    # Only retain the rows containing a certain string marker, and the columns equal to a certain column marker
    filter!(row -> any(occursin.(row_markers, row[:property])), data)
    cols_to_delete = []
    for (i, name) in enumerate(names(data))
        if (i > 1) # skip the label column
            if (!(name in column_markers))
                push!(cols_to_delete, i)
            end
        end
    end
    deletecols!(data, cols_to_delete)

    data_matrix = convert(Matrix, data)
    header = [replace(name, "_"=>" ") for name in names(data)] # replace _ with - to avoid latex errors
    category_assignments = [categories[findfirst(x->x==true, occursin.(categories, cur_property))] for cur_property in data_matrix[:, 1]]

    # For each pair of columns (after 1) create a pairwise comparison
    for col_one = 2:(size(data_matrix, 2)-1)
        alg_one = header[col_one]
        alg_one_data = Vector{Float64}(data_matrix[:, col_one])
        for col_two = col_one+1:size(data_matrix, 2)
            # Pull out the header name and the data for this solver
            alg_two = header[col_two]
            alg_two_data = Vector{Float64}(data_matrix[:, col_two])

            # Create an axis and then plot a scatter plot
            println("Plotting pair: ", (col_one, col_two), " which is ", (alg_one, alg_two))
            completed_one = sum(alg_one_data .< timeout)
            completed_two = sum(alg_two_data .< timeout)
            println("    ", alg_one, " completed: ", completed_one)
            println("    ", alg_two, " completed: ", completed_two)

            title=string(row_markers..., " ", alg_one, " vs ", alg_two)
            plot = Axis(style="black, width=10cm, height=10cm", xlabel=string(alg_one, " (completed: ", completed_one, ")"), ylabel=string(alg_two, " (completed: ", completed_two, ")"), title=title)
            plot.legendStyle = "at={(1.0, 0.5)}, anchor = west"

            push!(plot, Plots.Scatter(alg_one_data, alg_two_data, category_assignments, scatterClasses=full_style, mark="x", style="black", legendentry=legend_entries))
            push!(plot, Plots.Linear([0, timeout], [0, timeout], style="black, dashed"))

            # Make the output file and save the plot
            output_name = string(file_start, "_", row_markers..., "_Comparison_", alg_one, "_", alg_two)
            output_file = joinpath(output_path, output_name)
            save(output_file*".pdf", plot)
            save(output_file*".tex", plot)
        end
    end
end


# filename = "/Users/castrong/Desktop/Research/NeuralOptimization.jl/ResultProcessing/Results/results_fullbenchmark.csv"
# output_path = "/Users/castrong/Desktop/Research/NeuralOptimization.jl/ResultProcessing/Results/Plots/"
#
# column_lists = [
#                 ["Marabou_mip10sec", "MIPVerify_mip10sec"],
#                 ["Marabou_mip10sec", "MIPVerify_mip1sec"],
#                 ["Marabou_mip10sec", "MIPVerify_lp"],
#                 ["MIPVerify_mip10sec", "MIPVerify_lp"],
#                 ["MIPVerify_mip10sec", "MIPVerify_mip1sec"],
#                 ]
#
# ######################################################
# # Breakdown on different types of networks
# row_markers = [""]
# categories = ["MNIST", "ACAS", "AutoTaxi"]
# legend_entries = ["MNIST", "ACAS", "AutoTaxi"] # labels for the legend for these categories
# styles = ["mark=+, blue", "mark=star,red", "mark=x,black"]
#
# for column_markers in column_lists
#     make_pairwise_comparisons(filename, output_path, column_markers, row_markers, categories, legend_entries, styles; file_start="")
# end
#
#
# ######################################################
# # Breakdown on ACAS benchmarks by property
# row_markers = ["ACAS"]
# categories = ["property_optimization_1", "property_optimization_2", "property_optimization_3", "property_optimization_4"]
# legend_entries = ["Property 1", "Property 2", "Property 3", "Property 4"]
# styles = ["mark=+, blue", "mark=star,red", "mark=x,black", "mark=o,cyan"]
#
# for column_markers in column_lists
#     make_pairwise_comparisons(filename, output_path, column_markers, row_markers, categories, legend_entries, styles; file_start="")
# end

result_dir = "/Users/castrong/Downloads/Results2"
output_file_optimum = "/Users/castrong/Desktop/Research/NeuralOptimization.jl/ResultProcessing/Results/optima.csv"
output_file_times = "/Users/castrong/Desktop/Research/NeuralOptimization.jl/ResultProcessing/Results/times.csv"
results_to_csv(result_dir, output_file_optimum, output_file_times)
