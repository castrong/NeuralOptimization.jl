# A helper script that just initializes all of the optimizers
using GLPK
using Gurobi

# Approximate
LBFGS_optimizer = NeuralOptimization.LBFGS()
FGSM_optimizer = NeuralOptimization.FGSM()
PGD_optimizer = NeuralOptimization.PGD()

# NSVerify and Sherlock
VanillaMIP_Gurobi_optimizer_8threads = NeuralOptimization.VanillaMIP(optimizer=Gurobi.Optimizer, m=1e3, threads=8)
VanillaMIP_Gurobi_optimizer_1thread = NeuralOptimization.VanillaMIP(optimizer=Gurobi.Optimizer, m=1e3, threads=1)
VanillaMIP_GLPK_optimizer = NeuralOptimization.VanillaMIP(optimizer=GLPK.Optimizer, m=1e3)
Sherlock_Gurobi_optimizer_8threads = NeuralOptimization.Sherlock(optimizer=Gurobi.Optimizer, m=1e3, threads=8)
Sherlock_Gurobi_optimizer_1thread = NeuralOptimization.Sherlock(optimizer=Gurobi.Optimizer, m=1e3, threads=1)
Sherlock_GLPK_optimizer = NeuralOptimization.Sherlock(optimizer=GLPK.Optimizer)

# Marabou variations
Marabou_optimizer_ReLUViolation = NeuralOptimization.Marabou(use_sbt=false, divide_strategy = "ReLUViolation")
Marabou_optimizer_sbt_ReLUViolation = NeuralOptimization.Marabou(use_sbt=true, divide_strategy = "ReLUViolation")
Marabou_optimizer_earliestReLU = NeuralOptimization.Marabou(use_sbt=false, divide_strategy = "EarliestReLU")
Marabou_optimizer_sbt_earliestReLU = NeuralOptimization.Marabou(use_sbt=true, divide_strategy = "EarliestReLU")
MarabouBinary_optimizer_ReLUViolation = NeuralOptimization.MarabouBinarySearch(divide_strategy = "ReLUViolation")
MarabouBinary_optimizer_sbt_ReLUViolation = NeuralOptimization.MarabouBinarySearch(use_sbt=true, divide_strategy = "ReLUViolation")
MarabouBinary_optimizer_earliestReLU = NeuralOptimization.MarabouBinarySearch(divide_strategy = "EarliestReLU")
MarabouBinary_optimizer_sbt_earliestReLU = NeuralOptimization.MarabouBinarySearch(use_sbt=true, divide_strategy = "EarliestReLU")
