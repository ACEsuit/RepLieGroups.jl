using RepLieGroups
using Test

isdefined(Main, :___UTILS_FOR_TESTS___) || include("utils/utils_for_tests.jl")

##

@testset "RepLieGroups.jl" begin

    @testset "SYYVector" begin include("test_yyvector.jl"); end
    
    @testset "Clebsch Gordan Coeffs" begin include("test_clebschgordans.jl"); end

    @testset "Representation" begin include("test_representation.jl"); end

    @testset "Coupling Coeffs" begin include("test_coupling.jl"); end

    # @testset "O3 new" begin include("BenchMarkTests/test_benchmark.jl"); end

end
