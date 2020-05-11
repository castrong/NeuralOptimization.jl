module NeuralOptimization

import Pkg;

using JuMP # Domain specific modeling language to frame our problems
using GLPK # Open source optimization library
using Gurobi # Licensed optimization library - access licenses here: https://www.gurobi.com/academia/academic-program-and-licenses/
using Parameters # For a cleaner interface when creating models with named parameters
using Interpolations # only for PiecewiseLinear
using LazySets
using Optim # for (L-) BFGS
using Printf # For writing out .nnet files
using NPZ # For reading and writing .npy files
using PyCall # Also to read .npz with certain data types unsupported by NPZ
np = pyimport("numpy")

# We have to pin an old version of Flux to get it to work with Adversarial.jl
Pkg.free("Flux")
Pkg.add(Pkg.PackageSpec(name="Flux", version="0.8.3"))
Pkg.pin(Pkg.PackageSpec(name="Flux", version="0.8.3"))
using Flux;

Pkg.add(Pkg.PackageSpec(url="https://github.com/jaypmorgan/Adversarial.jl.git")); # Adversarial.jl
using Adversarial;

using LinearAlgebra
import LazySets: dim, HalfSpace # necessary to avoid conflict with Polyhedra

# For optimization methods:
import JuMP.MOI.OPTIMAL, JuMP.MOI.INFEASIBLE, JuMP.MOI.TIME_LIMIT

# TODO: What should this be like long term? want to have a clean process
# of specifying things but they're not supported by all optimizers
function model_creator(solver)
    println("in model creator")
    if (solver.optimizer == Gurobi.Optimizer)
        println("in first option")
        return Model(with_optimizer(solver.optimizer))#, OutputFlag=solver.output_flag, Threads=solver.threads))
        println("end first option")
    else
        println("in second option")
        return Model(with_optimizer(solver.optimizer))
    end
end
JuMP.Model(solver) = Model(with_optimizer(solver.optimizer))#model_creator(solver)

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
    compute_output,
    compute_objective,
    compute_gradient,
    forward_network,
    optimize,
    read_nnet

include("approximate/LBFGS.jl")
include("approximate/FGSM.jl")
include("exact/VanillaMIP.jl")
include("exact/Sherlock.jl")
include("exact/Marabou.jl")

export LBFGS
export FGSM
export VanillaMIP
export Sherlock
export Marabou

end
