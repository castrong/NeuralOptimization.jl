# A helper script that just initializes all of the optimizers
# Assumes NeuralOptimization.jl is already included
using Gurobi
using GLPK

# Approximate
LBFGS_optimizer = NeuralOptimization.LBFGS()
FGSM_optimizer = NeuralOptimization.FGSM()
PGD_optimizer = NeuralOptimization.PGD()

# NSVerify, Sherlock, and MIPVerify
VanillaMIP_Gurobi_optimizer_8threads = NeuralOptimization.VanillaMIP(optimizer=Gurobi.Optimizer, threads=8)
VanillaMIP_Gurobi_optimizer_1thread = NeuralOptimization.VanillaMIP(optimizer=Gurobi.Optimizer, threads=1)
VanillaMIP_GLPK_optimizer = NeuralOptimization.VanillaMIP(optimizer=GLPK.Optimizer)
Sherlock_Gurobi_optimizer_8threads = NeuralOptimization.Sherlock(optimizer=Gurobi.Optimizer, threads=8)
Sherlock_Gurobi_optimizer_1thread = NeuralOptimization.Sherlock(optimizer=Gurobi.Optimizer, threads=1)
Sherlock_GLPK_optimizer = NeuralOptimization.Sherlock(optimizer=GLPK.Optimizer)
MIPVerify_optimizer = NeuralOptimization.MIPVerify()

# Marabou variations
Marabou_optimizer_ReLUViolation = NeuralOptimization.Marabou(usesbt=false, dividestrategy = "ReLUViolation")
Marabou_optimizer_sbt_ReLUViolation = NeuralOptimization.Marabou(usesbt=true, dividestrategy = "ReLUViolation")
Marabou_optimizer_earliestReLU = NeuralOptimization.Marabou(usesbt=false, dividestrategy = "EarliestReLU")
Marabou_optimizer_sbt_earliestReLU = NeuralOptimization.Marabou(usesbt=true, dividestrategy = "EarliestReLU")
MarabouBinary_optimizer_ReLUViolation = NeuralOptimization.MarabouBinarySearch(dividestrategy = "ReLUViolation")
MarabouBinary_optimizer_sbt_ReLUViolation = NeuralOptimization.MarabouBinarySearch(usesbt=true, dividestrategy = "ReLUViolation")
MarabouBinary_optimizer_earliestReLU = NeuralOptimization.MarabouBinarySearch(dividestrategy = "EarliestReLU")
MarabouBinary_optimizer_sbt_earliestReLU = NeuralOptimization.MarabouBinarySearch(usesbt=true, dividestrategy = "EarliestReLU")

# Name to object dictionary
optimizer_name_to_object = Dict(
    "LBFGS"=>NeuralOptimization.LBFGS,
    "FGSM"=>NeuralOptimization.FGSM,
    "PGD"=>NeuralOptimization.PGD,
    "VanillaMIP"=>NeuralOptimization.VanillaMIP,
    "Sherlock"=>NeuralOptimization.Sherlock,
    "Marabou"=>NeuralOptimization.Marabou,
    "MarabouBinarySearch"=>NeuralOptimization.MarabouBinarySearch
    )
