using RepLieGroups
using Test

@testset "RepLieGroups.jl" begin
    # Write your tests here.
    @testset "O3-ClebschGordan" begin include("test_obsolete_cg.jl"); end
    @testset "O3" begin include("test_obsolete_o3.jl"); end
end
