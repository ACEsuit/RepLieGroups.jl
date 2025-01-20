using RepLieGroups
using Test

@testset "RepLieGroups.jl" begin
    # Write your tests here.
    @testset "SYYVector" begin include("test_yyvector.jl"); end
    
    @testset "CGcoef vs PartialWaveFunctions" begin include("test_cg_vs_partialwavefunctions.jl"); end
    
    @testset "RPE basis" begin include("new_rpe_test.jl"); end
    
    @testset "O3-ClebschGordan" begin include("test_obsolete_cg.jl"); end
    
    @testset "O3" begin include("test_obsolete_o3.jl"); end
end
