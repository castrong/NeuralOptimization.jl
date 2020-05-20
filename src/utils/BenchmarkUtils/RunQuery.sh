#!/bin/bash
query_line=$1
julia ./src/utils/BenchmarkUtils/RunQuery.jl $query_line
