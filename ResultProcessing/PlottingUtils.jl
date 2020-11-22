using DataFrames
using CSV
using PGFPlots
using Query

# For approximate solvers, make a visualization of
# how their optima differ
function make_val_comparison(filename, output_path, column_markers, virtual_best_marker, row_markers, row_disqualifiers, legend_entries, styles; file_start="", timeout=7200)
    #full_style = join([string(category, "={", style, "}, ") for (category, style) in zip(categories, styles)])[1:end-2] # remove last comma
    mkpath(output_path) # make the output path if it doesn't exist already
    # Read in the data
    data = CSV.read(filename)

    println("filename: ", filename)

    # Only retain the rows containing a certain string marker, and the columns equal to a certain column marker
    filter!(row -> any(occursin.(row_markers, row[:property])), data)
    # Filter out any row disqualifiers
    filter!(row -> !any(occursin.(row_disqualifiers, row[:property])), data)

    # only keeps rows where all are not inf
    for col_name in column_markers
        filter!(row -> !any(occursin.(["Inf", "inf", "NaN"], string(row[Symbol(col_name)]))), data)
    end
    cols_to_delete = []
    for (i, name) in enumerate(names(data))
        if (i > 1) # skip the label column
            if (!(name in column_markers || name == virtual_best_marker))
                push!(cols_to_delete, i)
            end
        end
    end
    deletecols!(data, cols_to_delete)
    data_matrix = convert(Matrix, data)
    header = [replace(name, "_"=>" ") for name in names(data)] # replace _ with - to avoid latex errors

    # Each column corresponds to a solver. We'll add each solver to the
    # graph with the first column as the x, the rest give the y
    # Create a comparison against each
    # ASSUME THE VIRTUAL BEST IS IN THE LAST COLUMN
    alg_one = header[end]
    alg_one_data = Vector{Float64}(data_matrix[:, end])
    plot = Axis(style="black, width=10cm, height=10cm", axisEqual=true, xlabel="Approximate", ylabel="Virtual Best", title="Approximate vs. Exact")
    plot.legendStyle = "at={(0.9, 0.2)}, anchor = east"
    min_val = Inf
    max_val = -Inf
    for col_two = 2:(size(data_matrix, 2))-1
        # Pull out the header name and the data for this solver
        alg_two = header[col_two]
        println("data one: ", alg_one_data)
        println("data two: ", data_matrix[:, col_two])
        println("alg: ", alg_two)
        # if its a string convert to floats
        if data_matrix[:, col_two][1] isa String
            alg_two_data = parse.(Float64, data_matrix[:, col_two])
        else
            alg_two_data = Vector{Float64}(data_matrix[:, col_two])
        end
        push!(plot, Plots.Scatter(alg_two_data, alg_one_data, style=styles[col_two-1], legendentry=legend_entries[col_two-1]))
        min_val = min(min_val, minimum(alg_one_data), minimum(alg_two_data))
        max_val = max(max_val, maximum(alg_one_data), maximum(alg_two_data))
    end
    # add diagonal line
    push!(plot, Plots.Linear([min_val, max_val], [min_val, max_val], style="black, dashed", mark="none"))


    # Make the output file and save the plot
    output_name = string(file_start, "_", row_markers..., "_Comparison_", "Approx_vs_exact")
    output_file = joinpath(output_path, output_name)
    save(output_file*".pdf", plot)
    save(output_file*".tex", plot)
end


function make_pairwise_comparisons(filename, output_path, column_markers, row_markers, row_disqualifiers, categories, legend_entries, styles; file_start="", timeout=7200)
    full_style = join([string(category, "={", style, "}, ") for (category, style) in zip(categories, styles)])[1:end-2] # remove last comma
    mkpath(output_path) # make the output path if it doesn't exist already

    # Read in the data
    data = CSV.read(filename)

    # Only retain the rows containing a certain string marker, and the columns equal to a certain column marker
    filter!(row -> any(occursin.(row_markers, row[:property])), data)
    # Filter out any row disqualifiers
    filter!(row -> !any(occursin.(row_disqualifiers, row[:property])), data)

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
            plot = Axis(style="black, width=10cm, height=10cm", axisEqual=true, xlabel=string(alg_one, " (completed: ", completed_one, ")"), ylabel=string(alg_two, " (completed: ", completed_two, ")"), title=title)
            plot.legendStyle = "at={(1.0, 0.5)}, anchor = west"

            push!(plot, Plots.Scatter(alg_one_data, alg_two_data, category_assignments, scatterClasses=full_style, mark="x", style="black", legendentry=legend_entries))
            low_val = min(minimum(alg_one_data), minimum(alg_two_data)) / 2
            push!(plot, Plots.Linear([low_val, timeout], [low_val, timeout], style="black, dashed", mark="none"))

            # Make the output file and save the plot
            output_name = string(file_start, "_", row_markers..., "_Comparison_", alg_one, "_", alg_two)
            output_file = joinpath(output_path, output_name)
            save(output_file*".pdf", plot)
            save(output_file*".tex", plot)
        end
    end
end

# Define a nice 4 color pallet
#colors = colormap("RdBu", length(indices_to_plot))
# color_dict = Dict()
# num_colors = length(indices_to_plot)
# for i = 1:num_colors
#     define_color(string("mycolor", i), [colors[i].r, colors[i].g, colors[i].b])
#     color_dict[indices_to_plot[i]] = string("mycolor", i)
# end

# Colorblind accessible colors
#https://www.nature.com/articles/nmeth.1618
define_color("color1", [230, 159, 0])
define_color("color2", [0, 0, 0])
define_color("color3", [86, 180, 233])
define_color("color4", [0, 158, 115])



filename = "/Users/castrong/Desktop/Research/NeuralOptimization.jl/ResultProcessing/Results/result_vals.csv"
output_path = "/Users/castrong/Desktop/Research/NeuralOptimization.jl/ResultProcessing/Results/approx_2hours_newcolors/"


# column_lists = [
#                 ["marabou_mip0.1sec", "MIPVerify_mip1sec"],
#                 ["marabou_mip0.1sec", "MIPVerify_mip5sec"],
#                 ["marabou_mip0.5sec", "MIPVerify_mip1sec"],
#                 ["marabou_mip0.5sec", "MIPVerify_mip5sec"],
#                 ["marabou_mip1sec", "MIPVerify_mip1sec"],
#                 ["marabou_mip1sec", "MIPVerify_mip5sec"],
#                 ["marabou_mip0.1sec", "marabou_mip0.5sec"],
#                 ["marabou_mip0.1sec", "marabou_mip1sec"],
#                 ["marabou_mip0.5sec", "marabou_mip1sec"],
#                 ["MIPVerify_mip1sec", "MIPVerify_mip5sec"],
#                 ]

column_lists = [["MarabouOpt", "MIPVerify"],
                ["MarabouBin", "MIPVerify"],
                ["MarabouOpt", "MarabouBin"]]

#
# ### Aprox vs exact on output optimization
row_markers = [""]
row_disqualifiers = ["mininput"]
column_markers = ["fgsm", "pgd", "lbfgs", "virtual best"]
styles = [ "mark=+, color1", "mark=x, color2", "mark=star, color3", "mark=triangle,color4"]
legend_entries = ["FGSM", "PGD", "LBFGS", "Virtual Best"]
make_val_comparison(filename, output_path, column_markers, "virtual best", row_markers, row_disqualifiers, legend_entries, styles; file_start="output_opt_approx_exact", timeout=7200)

### Aprox vs exact on min input optimization
row_markers = ["mininput"]
row_disqualifiers = []
column_markers = ["lbfgs", "virtual best"]
styles = [ "mark=star, red", "mark=triangle,cyan"]
legend_entries = ["LBFGS", "Virtual Best"]
make_val_comparison(filename, output_path, column_markers, "virtual best", row_markers, row_disqualifiers, legend_entries, styles; file_start="input_opt_approx_exact", timeout=7200)


######################################################
# Breakdown on different types of networks for all benchmarks
# row_markers = [""]
# row_disqualifiers = []
# categories = ["acas_property_optimization", "AutoTaxi", "property_2_mininput", "MNIST"]
# legend_entries = ["ACAS output", "AutoTaxi output", "ACAS input", "MNIST input", ] # labels for the legend for these categories
# styles = ["mark=o,color1", "mark=triangle,color2", "mark=x,color3", "mark=+,color4"]
#
# for column_markers in column_lists
#     make_pairwise_comparisons(filename, output_path, column_markers, row_markers, row_disqualifiers, categories, legend_entries, styles; file_start="")
# end

# ######################################################
# # Breakdown on different types of networks for output optimization
# row_markers = [""]
# row_disqualifiers = ["mininput"]
# categories = ["ACAS", "AutoTaxi"]
# legend_entries = ["ACAS", "AutoTaxi"] # labels for the legend for these categories
# styles = [ "mark=star,blue", "mark=x,red"] # +, blue for old mnist
#
# for column_markers in column_lists
#     make_pairwise_comparisons(filename, output_path, column_markers, row_markers, row_disqualifiers, categories, legend_entries, styles; file_start="")
# end
#
#
# ######################################################
# # Breakdown on ACAS benchmarks by property for output optimization
# row_markers = ["ACAS"]
# row_disqualifiers = ["mininput"]
# categories = ["property_optimization_1", "property_optimization_2", "property_optimization_3", "property_optimization_4"]
# legend_entries = ["Property 1", "Property 2", "Property 3", "Property 4"]
# styles = ["mark=+, blue", "mark=star,red", "mark=x,black", "mark=o,cyan"]
#
# for column_markers in column_lists
#     make_pairwise_comparisons(filename, output_path, column_markers, row_markers, row_disqualifiers, categories, legend_entries, styles; file_start="")
# end
#
#
# ######################################################
# # Breakdown on different types of networks for input optimization
# row_markers = ["mininput"]
# row_disqualifiers = []
# categories = ["ACAS", "mnist"]
# legend_entries = ["ACAS", "MNIST"] # labels for the legend for these categories
# styles = [ "mark=star,blue", "mark=x,red"] # +, blue for old mnist
#
# for column_markers in column_lists
#     make_pairwise_comparisons(filename, output_path, column_markers, row_markers, row_disqualifiers, categories, legend_entries, styles; file_start="")
# end
