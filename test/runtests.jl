"""Run tests"""

using Test

include(joinpath(@__DIR__, "../src/DirectNOCP.jl"))

get_plot = false

@testset "ImpulsiveProblem" begin
    include("test_impulsive.jl")
end
