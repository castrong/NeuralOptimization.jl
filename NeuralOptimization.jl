module NeuralOptimization

import Pkg;

using JuMP # Domain specific modeling language to frame our problems
using GLPK # Open source optimization library
using Gurobi # Licensed optimization library - access licenses here: https://www.gurobi.com/academia/academic-program-and-licenses/
using Parameters # For a cleaner interface when creating models with named parameters
using Interpolations # only for PiecewiseLinear
using LazySets
using Optim # for (L-) BFGS
Pkg.add(Pkg.PackageSpec(url="https://github.com/jaypmorgan/Adversarial.jl.git")); # Adversarial.jl
using Adversarial;
using PyCall; # For calling to Marabou 

using LinearAlgebra
import LazySets: dim, HalfSpace # necessary to avoid conflict with Polyhedra

# Include utils that help to define the networks and problems
include("utils/activation.jl")
include("utils/network.jl")
include("utils/problem.jl")
include("utils/util.jl")


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
    optimize



include("approximate/LBFGS.jl")
export LBFGS

end
