using RepLieGroups
using Test

include("utils/utils_for_tests.jl")

@testset "RepLieGroups.jl" begin
    # Write your tests here.
    @testset "SYYVector" begin include("test_yyvector.jl"); end
    
    @testset "CGcoef vs PartialWaveFunctions" begin include("test_cg_vs_partialwavefunctions.jl"); end
    
    @testset "O3-ClebschGordan" begin include("test_obsolete_cg.jl"); end
    
    @testset "O3" begin include("test_obsolete_o3.jl"); end

    @testset "O3 new" begin include("new_rpe_test.jl"); end

    @testset "O3 new real" begin include("new_rpe_test_real.jl"); end
end
