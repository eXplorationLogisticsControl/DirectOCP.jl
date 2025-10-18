module DirectOCP


using JuMP
using LinearAlgebra
using OrdinaryDiffEq
using Printf


abstract type OptimalControlProblem end

include("memoization.jl")
include("impulsive_problem.jl")

end # module DirectOCP
