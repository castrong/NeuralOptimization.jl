module NeuralOptimization

import Pkg;
import Base.show

using JuMP # Domain specific modeling language to frame our problems
using GLPK # Open source optimization library
using Gurobi # Licensed optimization library - access licenses here: https://www.gurobi.com/academia/academic-program-and-licenses/
using Parameters # For a cleaner interface when creating models with named parameters
using Interpolations # only for PiecewiseLinear
using LazySets # For set descriptions of our input and output
using Optim # for (L-) BFGS
using Printf # For writing out .nnet files
using NPZ # For reading and writing .npy files
using PyCall # For Marabou, also to read .npz with certain data types unsupported by NPZ
using BenchmarkTools # For our benchmark timing
using Requires

# # Python libraries that we'll need for Marabou
py"""
import time
import sys
import numpy as np
import copy
from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy import MarabouUtils
"""

# We have to pin an old version of Flux to get it to work with Adversarial.jl
# Pkg.free("Flux")
# Pkg.add(Pkg.PackageSpec(name="Flux", version="0.8.3"))
# Pkg.pin(Pkg.PackageSpec(name="Flux", version="0.8.3"))
using Flux;

#Pkg.add(Pkg.PackageSpec(url="https://github.com/jaypmorgan/Adversarial.jl.git")); # Adversarial.jl
while true
   try
      using Adversarial
      println("Adversarial imported!")
      break
   catch e
      println("Failed, sleep for 0.5 seconds")
      sleep(0.5)
   end
end

using LinearAlgebra
import LazySets: dim, HalfSpace # necessary to avoid conflict with Polyhedra

# For optimization methods:
import JuMP.MOI.OPTIMAL, JuMP.MOI.INFEASIBLE, JuMP.MOI.TIME_LIMIT

# TODO: What should this be like long term? want to have a clean process
# of specifying things but they're not supported by all optimizers
function model_creator(solver)
    if (solver.optimizer == Gurobi.Optimizer)
        println("Creating Gurobi model")
        println("Threads: ", solver.threads)
        print(solver)
        return JuMP.Model(with_optimizer(solver.optimizer, OutputFlag=solver.output_flag, Threads=solver.threads))
    else
        println("Creating optimizer not Gurobi")
        return JuMP.Model(with_optimizer(solver.optimizer))
    end
end

JuMP.value(vars::Vector{VariableRef}) = value.(vars)

# Include utils that help to define the networks and problems
include("utils/activation.jl")
include("utils/network.jl")
include("utils/problem.jl")
include("utils/util.jl")
include("utils/variables.jl")
include("utils/objectives.jl")
include("utils/constraints.jl")

# To help with printouts
macro Name(arg)
    string(arg)
end

using Requires # to require a certian version of flux
function __init__()
  @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("utils/flux.jl")
end

# Export helpful objects from your includes
export
    Solver,
    Network,
    Problem,
    LinearObjective,
    Result,
    read_nnet,
    read_property_file,
    compute_output,
    compute_objective,
    compute_gradient,
    forward_network,
    optimize,
    read_nnet

include("approximate/LBFGS.jl")
include("approximate/FGSM.jl")
include("approximate/PGD.jl")
include("exact/VanillaMIP.jl")
include("exact/Sherlock.jl")
include("exact/Marabou.jl")
include("exact/MarabouBinarySearch.jl")
include("exact/MIPVerify.jl")


export LBFGS
export FGSM
export VanillaMIP
export Sherlock
export Marabou
export MarabouBinarySearch
export MIPVerify

end
