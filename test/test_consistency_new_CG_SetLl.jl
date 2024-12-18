using StaticArrays, LinearAlgebra, RepLieGroups
using Test

N_test = 100
@info "Testing consistency of SetLl_new and SetLl and SetLl0 && CG_new and CG"
for _ = 1:N_test
    # SetLl_new vs SetLl
    N = rand(3:5) # SetLl has a bug when N=2 so here N starts from 3
    L = rand(0:5) # order of equivariance
    l = SVector{N}([rand(1:8) for _ = 1:N])
    SL1 = RepLieGroups.SetLl_new(l,L)
    SL2 = RepLieGroups.SetLl(l,N,L)
    @test [ SL1[i][2:end] for i in 1:length(SL1) ] == [ SL2[i][2:end] for i in 1:length(SL2) ]

    # SetLl_new vs SetLl0
    N = rand(2:5)
    l = SVector{N}([rand(1:8) for _ = 1:N])
    SL1 = RepLieGroups.SetLl_new(l,0)
    SL2 = RepLieGroups.SetLl0(l,N)
    @test [ SL1[i][2:end] for i in 1:length(SL1) ] == [ SL2[i][2:end] for i in 1:length(SL2) ]

    # CG_new vs CG
    L = rand(0:5)
    m = RepLieGroups.ML(l,N,L)[rand(1:length(RepLieGroups.ML(l,N,L)))]

    for L in SL1
        CG1 = RepLieGroups.CG_new(l,typeof(l)(m),L)
        CG2 = RepLieGroups.CG(l,m,L,length(l))
        @test CG1 ≈ CG2
    end
end

