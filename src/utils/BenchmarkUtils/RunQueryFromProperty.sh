#!/bin/bash
# Call RunQueryFromProperty if it's not MIPVerify.jl. If it is, then call RunMIPVerifyFromPropertyFile.jl

query_line="$@" # all command line arguments separated by spaces
echo "Query line: $query_line"
if [[ $query_line == *"MIPVerify"* ]];
then
  echo "Running MIPVerify"
  julia /Users/castrong/Desktop/Research/MIPVerifyWrapper/RunMIPVerifyFromPropertyFile.jl $query_line
else
  echo "Running non MIPVerify"
  julia /Users/castrong/Desktop/Research/NeuralOptimization.jl/src/utils/BenchmarkUtils/RunQueryFromProperty.jl $query_line
fi
