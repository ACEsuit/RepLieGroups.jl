
using SpheriCart, StaticArrays, LinearAlgebra, RepLieGroups, WignerD, 
      Combinatorics
using RepLieGroups.O3: Rot3DCoeffs, Rot3DCoeffs_real
O3 = RepLieGroups.O3
using Test


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

function rand_radial(rmax) 
   return rmax*rand()
end

function rand_rot() 
   K = @SMatrix randn(3,3)
   return exp(K - K') 
end


"""
This implements a permutation-invariant and O(3) invariant function of 
4 (or more) variables. 
"""
function f(Rs, q::Integer; coeffs, MM, ll, nn)
   cheb(r, n) = cos(n * acos(r))

   # q = selection of the basis function 
   @assert length(coeffs) >= q 
   @assert length(MM) >= q 
   # N = number of mm terms to combine ... 
   #   ... actually no idea what this is???
   N = length(coeffs[1])
   # @show N 
   # correlation order 
   ORD = length(ll) 
   @assert length(nn) == ORD
   # @show ORD 

   Lmax = maximum(ll)
   real_basis = SphericalHarmonics(Lmax)
   Y = [ eval_cY(real_basis, 𝐫) for 𝐫 in Rs ]
   xx = [ (0.1 + norm(𝐫))/(1.1) for 𝐫 in Rs]
   T = [ cheb(x, n) for x in xx, n in nn ]
   A = [ sum(Y[j] * T[j, in] for j = 1:length(Rs))   for in = 1:ORD ]
   out = zero(eltype(A[1]))::ComplexF64 
   for (c, mm) in zip(coeffs[q, :], MM)
      @assert length(mm) == ORD
      ii = [ SpheriCart.lm2idx(ll[α], mm[α]) for α in 1:ORD ]
      out += c * prod(A[α][ii[α]] for α = 1:ORD)
   end
   return real(out)
end

# we can look for linear independence... 
function f_rand_batch(coeffs, MM, ntest, ll, nn) 
   N = 2 * length(ll) # length(MM[1])
   nbas = size(coeffs, 1)
   BB = zeros(nbas, ntest)
   for i = 1:ntest 
      Rs = [rand_sphere() * rand() for _ in 1:N]
      for q = 1:nbas
         BB[q, i] = f(Rs, q; coeffs = coeffs, MM=MM, ll=ll, nn=nn)
      end
   end
   return A
end

##
# CASE 2: 4-correlations, L = 0 (revisited)
L = 0
cc = Rot3DCoeffs(L)
# now we fix an ll = (l1, l2, l3) triple ask for all possible linear combinations 
# of the tensor product basis   Y[l1, m1] * Y[l2, m2] * Y[l3, m3] * Y[l4, m4]
# that are invariant under O(3) rotations.
ll = SA[1,2,2,2,3]
N = length(ll)
nn = ones(Int64, N) # for the moment, nn has to be only ones
@assert length(ll) == length(nn)
@time coeffs1, MM1 = O3.re_basis(cc, ll)
nbas_ri1 = size(coeffs1, 1)

@time A = f_rand_batch(coeffs1,MM1, 1_000, ll, nn)
rk1 = rank(A, rtol = 1e-12) #does not correspond to what is in our paper
U, S, V = svd(A)
coeffs_ind1 = Diagonal(S[1:rk1]) \ (U[:, 1:rk1]' * coeffs1)

# Version GD
@time coeffs_rpi, MM_rpi = MatFmi(nn,ll)
@show size(coeffs_rpi)
@time coeffs2, MM2 = ri_basis_new(ll)

# @time A_rpi = f_rand_batch(coeffs_rpi, MM_rpi, 1_000, ll, nn) #this should depend on nn
# rk_rpi = rank(A_rpi, rtol = 1e-12) #does not correspond to what is in our paper


rk2 = rank(coeffs_rpi,rtol = 1e-12)
@test rk1 == rk2 #error here
U, S, V = svd(coeffs_rpi)
coeffs_ind2 = Diagonal(S[1:rk2]) \ (U[:, 1:rk2]' * coeffs2)
# multiset_permutations([-1 -1 1 1], 4)
# sort([[1, 1], [2, 3], [2, 2], [1, 4]])
@test rank([coeffs_ind1;coeffs_ind2]) == rk1
