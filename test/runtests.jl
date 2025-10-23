using Test
using AeroInviscid
using AeroGeometry
using LinearAlgebra

@testset "AeroInviscid.jl Tests" begin
    include("test_joukowski.jl")
end
