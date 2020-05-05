module NeuralOptimization

import Pkg;

using JuMP # Domain specific modeling language to frame our problems
using GLPK # Open source optimization library
using Gurobi # Licensed optimization library - access licenses here: https://www.gurobi.com/academia/academic-program-and-licenses/
using Parameters # For a cleaner interface when creating models with named parameters
using Interpolations # only for PiecewiseLinear
using LazySets
using Optim # for (L-) BFGS

# We have to pin an old version of Flux to get it to work with Adversarial.jl
Pkg.free("Flux")
Pkg.add(Pkg.PackageSpec(name="Flux", version="0.8.3"))
Pkg.pin(Pkg.PackageSpec(name="Flux", version="0.8.3"))
using Flux;

Pkg.add(Pkg.PackageSpec(url="https://github.com/jaypmorgan/Adversarial.jl.git")); # Adversarial.jl
using Adversarial;
using PyCall; # For calling to Marabou

using LinearAlgebra
import LazySets: dim, HalfSpace # necessary to avoid conflict with Polyhedra

# For optimization methods:
import JuMP.MOI.OPTIMAL, JuMP.MOI.INFEASIBLE, JuMP.MOI.TIME_LIMIT

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

function __init()__
    println("Hello world");
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

export LBFGS
export FGSM
export VanillaMIP
export Sherlock

end
