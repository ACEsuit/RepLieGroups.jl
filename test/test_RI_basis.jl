using SpheriCart, StaticArrays, LinearAlgebra, RepLieGroups, WignerD, 
      Combinatorics
using RepLieGroups.O3: Rot3DCoeffs, Rot3DCoeffs_real
O3 = RepLieGroups.O3
using Test

# Evaluation of spherical harmonics
function eval_cY(rbasis::SphericalHarmonics{LMAX}, 𝐫) where {LMAX}  
    Yr = rbasis(𝐫)
    Yc = zeros(Complex{eltype(Yr)}, length(Yr))
    for l = 0:LMAX
       # m = 0 
       i_l0 = SpheriCart.lm2idx(l, 0)
       Yc[i_l0] = Yr[i_l0]
       # m ≠ 0 
       for m = 1:l 
          i_lm⁺ = SpheriCart.lm2idx(l,  m)
          i_lm⁻ = SpheriCart.lm2idx(l, -m)
          Ylm⁺ = Yr[i_lm⁺]
          Ylm⁻ = Yr[i_lm⁻]
          Yc[i_lm⁺] = (-1)^m * (Ylm⁺ + im * Ylm⁻) / sqrt(2)
          Yc[i_lm⁻] = (Ylm⁺ - im * Ylm⁻) / sqrt(2)
       end
    end 
    return Yc
 end
 
 function rand_sphere() 
    u = @SVector randn(3)
    return u / norm(u) 
 end
 
 function rand_rot() 
    K = @SMatrix randn(3,3)
    return exp(K - K') 
 end

 function f(Rs, q; coeffs=coeffs, MM=MM, ll=ll) 
    Lmax = maximum(ll)
    real_basis = SphericalHarmonics(Lmax)
    YY = []
    for i in 1:length(ll)
        push!(YY, eval_cY(real_basis, Rs[i]))
    end
    out = zero(eltype(YY[1]))
    for (c, mm) in zip(coeffs[q, :], MM) 
        ind = Int[]
        for i in 1:length(ll)
            push!(ind, SpheriCart.lm2idx(ll[i], mm[i]))
        end
        out += c * prod(YY[i][ind[i]] for i in 1:length(ll))
    end
    return out 
end


# for the moment the code with generalized CG only works with L=0
L = 0
cc = Rot3DCoeffs(L)
ll = SA[1,1,2,2,4]

# version with svd
@time coeffs1, MM1 = O3.re_basis(cc, ll)
nbas = size(coeffs1, 1)

#version with gen CG coefficients
@time coeffs2, MM2 = ri_basis_new(ll)

# simple test on size
@test size(coeffs1) == size(coeffs2)
@test size(MM1) == size(MM2)

P1 = sortperm(MM1)
P2 = sortperm(MM2)
MMsorted1 = MM1[P1]
MMsorted2 = MM2[P2]
# check that same mm values
@test MMsorted1 == MMsorted2

coeffsp1 = coeffs1[:,P1]
coeffsp2 = coeffs2[:,P2]

# test that full rank
@test rank(coeffsp1) == size(coeffsp1,1)
@test rank(coeffsp2) == size(coeffsp2,1)

# check that the coef span the same space - test fails
@test nbas == rank([coeffsp1; coeffsp2], rtol = 1e-12)


Rs = [rand_sphere() for _ in 1:length(ll)]
Q = rand_rot() 
QRs = [Q*Rs[i] for i in 1:length(Rs)]
fRs1 = [ f(Rs, q; coeffs=coeffs1, MM=MM1, ll=ll) for q = 1:nbas ]
fRs1Q = [ f(QRs, q; coeffs=coeffs1, MM=MM1, ll=ll) for q = 1:nbas ]

# check invariance (for now)
@test norm(fRs1 .- fRs1Q) < 1e-12

fRs2 = [ f(Rs, q; coeffs=coeffs2, MM=MM2, ll=ll) for q = 1:nbas ]
fRs2Q = [ f(QRs, q; coeffs=coeffs2, MM=MM2, ll=ll) for q = 1:nbas ]

# check invariance (for now)
@test norm(fRs2 .- fRs2Q) < 1e-12

# Test on batch
ntest = 1000
A1 = zeros(nbas, ntest)
A2 = zeros(nbas, ntest)
for i = 1:ntest 
   Rs = [rand_sphere() for _ in 1:length(ll)]
   for q = 1:nbas
       fRs = f(Rs, q; coeffs = coeffs1, MM=MM1, ll=ll)
       @assert abs.(imag(fRs)) < 1e-16
       A1[q, i] = real(fRs)

       fRs2 = f(Rs, q; coeffs = coeffs2, MM=MM2, ll=ll)
       @assert abs.(imag(fRs2)) < 1e-16
       A2[q, i] = real(fRs2)
   end
end

# check that functions span same space
rk = rank([A1;A2]; rtol = 1e-12)  
@test rk == nbas

# ----------------------------
# Extension to equivariance 
# ----------------------------

# Test SetLl0
N = 5
l = rand(1:10, N)
SL1 = RepLieGroups.SetLl(l,N,0)
SL2 = RepLieGroups.SetLl0(l,N)
@test SL1 == SL2

# Test SetML 
N = 5
l = rand(1:10, N)
SM1 = RepLieGroups.ML(l,N,0)
SM2 = RepLieGroups.ML0(l,N)
@test SM1 == SM2

 
# # Version CO
# L = 1
# cc = Rot3DCoeffs(L)
# ll = SA[1,1,2,3,4]

# # version with svd
# @time coeffs1, MM1 = O3.re_basis(cc, ll)
# nbas = size(coeffs1, 1)

# # Version GD
# @time coeffs2, MM2 = re_basis_new(ll,L)

# # simple test on size
# @test size(coeffs1) == size(coeffs2)
# @test size(MM1) == size(MM2)

# P1 = sortperm(MM1)
# P2 = sortperm(MM2)
# MMsorted1 = MM1[P1]
# MMsorted2 = MM2[P2]
# # check that same mm values
# @test MMsorted1 == MMsorted2

# coeffsp1 = coeffs1[:,P1]
# coeffsp2 = coeffs2[:,P2]

# # test that full rank
# @test rank(coeffsp1) == size(coeffsp1,1)
# @test rank(coeffsp2) == size(coeffsp2,1)

# # check that the coef span the same space 
# @test nbas == rank([coeffsp1; coeffsp2], rtol = 1e-12)

# Rs = [rand_sphere() for _ in 1:length(ll)]
# Q = rand_rot() 
# QRs = [Q*Rs[i] for i in 1:length(Rs)]
# fRs1 = [ f(Rs, q; coeffs=coeffs1, MM=MM1, ll=ll) for q = 1:nbas ]
# fRs1Q = [ f(QRs, q; coeffs=coeffs1, MM=MM1, ll=ll) for q = 1:nbas ]
