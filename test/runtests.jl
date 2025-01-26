using RepLieGroups
using Test

isdefined(Main, :___UTILS_FOR_TESTS___) || include("utils/utils_for_tests.jl")

##

@testset "RepLieGroups.jl" begin

    @testset "SYYVector" begin include("test_yyvector.jl"); end
    
    @testset "Clebsch Gordan Coeffs" begin include("test_clebschgordans.jl"); end

    # @testset "O3 new" begin include("new_rpe_test.jl"); end

    @testset "O3 new real" begin include("new_rpe_test_real.jl"); end
    
    @testset "RE SEMI PI" begin include("test_re_semi_pi.jl"); end
end
