"""Run tests"""

using Test

include(joinpath(@__DIR__, "../src/DirectOCP.jl"))

get_plot = false

@testset "ImpulsiveProblem" begin
    include("test_impulsive.jl")
end

@testset "ContinuousProblem" begin
    include("test_continuous.jl")
end