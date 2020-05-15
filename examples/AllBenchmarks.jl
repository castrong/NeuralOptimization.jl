network_files = ["./Networks/AutoTaxi/AutoTaxi_32Relus_200Epochs_OneOutput.nnet", "./Networks/AutoTaxi/AutoTaxi_64Relus_200Epochs_OneOutput.nnet", "./Networks/AutoTaxi/AutoTaxi_128Relus_200Epochs_OneOutput.nnet"]
objectives = [objective] # an objective for each input file
input_files = ["./Datasets/AutoTaxi/AutoTaxi_2323.npy"]
deltas = [0.001, 0.005, 0.01, 0.015]
transpose_inputs = true
maximize = true
append_output = false
timeout = 3600
output_file = "./BenchmarkOutput/AutoTaxi_1.csv"

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
