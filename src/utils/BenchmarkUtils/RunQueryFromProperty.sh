#!/bin/bash
# Call RunQueryFromProperty if it's not MIPVerify.jl. If it is, then call RunMIPVerifyFromPropertyFile.jl

export HOME=/barrett/scratch/haozewu/
export GRB_LICENSE_FILE=/barrett/scratch/haozewu/Optimization/gurobi.lic
export GUROBI_HOME=/barrett/scratch/haozewu/Optimization/gurobi902/linux64
export JULIA_NUM_THREADS=1

export PYTHONPATH="${PYTHONPATH}:/barrett/scratch/haozewu/Optimization/Marabou"

query_line="$@" # all command line arguments separated by spaces
echo "Query line: $query_line"
if [[ $query_line == *"MIPVerify"* ]];
then
  echo "Running MIPVerify"
  /barrett/scratch/haozewu/Optimization/julia-1.4.1/bin/julia /barrett/scratch/haozewu/Optimization/MIPVerifyWrapper/RunMIPVerifyFromPropertyFile.jl $query_line
else
  echo "Running non MIPVerify"
  /barrett/scratch/haozewu/Optimization/julia-1.4.1/bin/julia /barrett/scratch/haozewu/Optimization/NeuralOptimization.jl/src/utils/BenchmarkUtils/RunQueryFromProperty.jl $query_line
fi
