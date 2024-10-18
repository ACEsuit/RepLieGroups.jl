
using SpheriCart, StaticArrays, LinearAlgebra, RepLieGroups, WignerD, 
      Combinatorics
using Rotations
using RepLieGroups.O3: Rot3DCoeffs
O3 = RepLieGroups.O3

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


##
# CASE 2: 4-correlations, L = 0 (revisited)
L = 0
cc = Rot3DCoeffs(0)
# now we fix an ll = (l1, l2, l3) triple ask for all possible linear combinations 
# of the tensor product basis   Y[l1, m1] * Y[l2, m2] * Y[l3, m3] * Y[l4, m4]
# that are invariant under O(3) rotations.
ll = SA[3, 3, 2, 2, 2]
coeffs, MM = O3.re_basis(cc, ll)
nbas = size(coeffs, 1)
# coeffs = nbasis x length(MM) matrix 
#     MM = vector of (m1, m2, m3, m4) tuples 
# for 4-correlations and higher, the number of possible couplings can be 
# greater than one. An interesting result of this is that even if those 
# basis functions as tensor products of Ylms are linearly independent, they 
# need no longer be linearly independent once we impose permutation-invariance. 

"""
This implements a permutation-invariant and O(3) invariant function of 
4 (or more) variables. 
"""
function f(Rs, q::Integer; coeffs=coeffs, MM=MM)
   real_basis = SphericalHarmonics(3)
   Y = [ eval_cY(real_basis, 𝐫) for 𝐫 in Rs ]
   A = sum(Y)   # this is a permutation-invariant embedding
   out = zero(eltype(A))
   for (c, mm) in zip(coeffs[q, :], MM)
      ii = [SpheriCart.lm2idx(ll[α], mm[α]) for α in 1:4]
      out += c * prod(A[ii])
   end
   return real(out)
end


Rs = [rand_sphere() for _ in 1:4]
[ f(Rs, q) for q = 1:nbas ]

# we can look for linear independence... 

function f_rand_batch(coeffs, MM, ntest) 
   nbas = size(coeffs, 1)
   A = zeros(nbas, ntest)
   for i = 1:ntest 
      Rs = [rand_sphere() for _ in 1:4]
      for q = 1:nbas
         A[q, i] = f(Rs, q; coeffs = coeffs)
      end
   end
   return A
end

A = f_rand_batch(coeffs, MM, 1_000)

# the rank of this matrix is only 3, not nbas = 5!
rk = rank(A)  
# 3 

# this is in fact very clear from the SVD 
svdvals(A)
# 5-element Vector{Float64}:
#  25.568548451339442
#   6.754493909270821
#   5.45667827344765
#   5.398138581058422e-15
#   1.6731699590390224e-15

##
# In ACE we use a semi-analytic construction to make the basis functions 
# linearly indepdendent. 
# in a full ACE code, this is a bit more complex since n channels are added 
# to the story. This can be found here: 
# https://github.com/ACEsuit/ACE1.jl/blob/8ac52d2128241a01d8b9a036f41b1d5106cbeb07/src/rpi/rotations3d.jl#L334
# https://github.com/ACEsuit/ACE1.jl/blob/8ac52d2128241a01d8b9a036f41b1d5106cbeb07/src/rpi/rotations3d.jl#L348

# For simplicity, we can just use the SVD to construct a 
# linearly independent basis.

U, S, V = svd(A)
coeffs_ind = Diagonal(S[1:rk]) \ (U[:, 1:rk]' * coeffs)

##
# now let's re-compute A 

ntest = 1_000
A_ind = zeros(rk, ntest)
for i = 1:ntest 
   Rs = [rand_sphere() for _ in 1:4]
   for q = 1:rk
      A_ind[q, i] = f(Rs, q; coeffs = coeffs_ind)
   end
end

# the rank of this matrix is 3, which is what we expect
rk_ind = rank(A_ind)  
# 3 

# And we even scaled the coupling coeffs so that the singular values are 
# now all close to 1. 
svdvals(A_ind)
# 3-element Vector{Float64}:
#  0.997130965666211
#  0.8853910366671397
#  0.8668533817486788