module DirectNOCP


using JuMP
using LinearAlgebra
using OrdinaryDiffEq
using Printf


abstract type DirectNonlinearOptimalControlProblem end

include("memoization.jl")
include("impulsive_problem.jl")

end # module DirectNOCP
