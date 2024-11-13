
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

function eval_cheb(𝐫::AbstractVector, nmax)
   r = norm(𝐫)
   x = (0.1 + r) / 1.2
   return [ cos( (n-1) * acos(x) ) for n = 1:nmax ]
end 

function rand_sphere() 
   u = @SVector randn(3)
   return u / norm(u) 
end

rand_ball() = rand_sphere() * rand()


function rand_rot() 
   K = @SMatrix randn(3,3)
   return exp(K - K') 
end



function eval_basis(Rs; coeffs, MM, ll, nn)
   @assert minimum(nn) >= 1 # radial basis indexing starts at 1 not 0. 
   @assert size(coeffs, 2) == length(MM) 

   # correlation order 
   ORD = length(ll) 
   @assert length(nn) == ORD
   @assert all( length(mm) == ORD for mm in MM )
   @assert length(Rs) == ORD # only for the non-sym basis!!

   # spherical harmonics 
   real_basis = SphericalHarmonics(maximum(ll))
   Y = [ eval_cY(real_basis, 𝐫) for 𝐫 in Rs ]

   # radial basis 
   T = [ eval_cheb(𝐫, maximum(nn)) for 𝐫 in Rs ]
      
   BB = zeros(size(coeffs, 1))
   for i_mm = 1:length(MM)
      mm = MM[i_mm]
      ii_lm = [ SpheriCart.lm2idx(ll[α], mm[α]) for α in 1:ORD ]
      BB += coeffs[:, i_mm] * prod( Y[α][ii_lm[α]] * T[α][nn[α]] for α = 1:ORD )
   end 

   return real.(BB)
end


function eval_sym_basis(Rs; coeffs, MM, ll, nn)
   @assert minimum(nn) >= 1 # radial basis indexing starts at 1 not 0. 
   @assert size(coeffs, 2) == length(MM) 

   # correlation order 
   ORD = length(ll) 
   @assert length(nn) == ORD
   @assert all( length(mm) == ORD for mm in MM )

   # spherical harmonics 
   real_basis = SphericalHarmonics(maximum(ll))
   Y = [ eval_cY(real_basis, 𝐫) for 𝐫 in Rs ]

   # radial basis 
   T = [ eval_cheb(𝐫, maximum(nn)) for 𝐫 in Rs ]
   
   # pooled tensor product operation -> A[i_lm, n]
   A = sum( Y[j] * T[j]' for j = 1:length(Rs) )
   
   BB = zeros(size(coeffs, 1))
   for i_mm = 1:length(MM)
      mm = MM[i_mm]
      ii_lm = [ SpheriCart.lm2idx(ll[α], mm[α]) for α in 1:ORD ]
      BB += coeffs[:, i_mm] * prod( A[ii_lm[α], nn[α]] for α = 1:ORD )
   end 

   return real.(BB)
end


function rand_batch(ntest; coeffs, MM, ll, nn) 
   ORD = length(ll) # length of each group 
   BB = zeros(size(coeffs, 1), ntest)
   for i = 1:ntest 
      # construct a random set of particles with 𝐫 ∈ ball(radius=1)
      Rs = [ rand_ball() for _ in 1:ORD ]
      BB[:, i] = eval_basis(Rs; coeffs=coeffs, MM=MM, ll=ll, nn=nn) 
   end
   return BB
end

function sym_rand_batch(ntest; coeffs, MM, ll, nn) 
   ORD = length(ll) # length of each group (could be > ORD)
   BB = zeros(size(coeffs, 1), ntest)
   for i = 1:ntest 
      Rs = [ rand_ball() for _ in 1:ORD ]
      BB[:, i] = eval_sym_basis(Rs; coeffs=coeffs, MM=MM, ll=ll, nn=nn)
   end
   return BB
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
nn = @SVector ones(Int64, N) # for the moment, nn has to be only ones
@assert length(ll) == length(nn)
@time coeffs1, MM1 = O3.re_basis(cc, ll)
nbas_ri1 = size(coeffs1, 1)
rank(coeffs1, rtol = 1e-12)

ntest = 1000

X = rand_batch(ntest; coeffs=coeffs1, MM=MM1, ll=ll, nn=nn)
rank(X; rtol=1e-12)

Xsym = sym_rand_batch(ntest; coeffs=coeffs1, MM=MM1, ll=ll, nn=nn)
rk1 = rank(Xsym; rtol=1e-12)


# Version GD
@time coeffs_rpi, MM_rpi = MatFmi(nn,ll)
@show size(coeffs_rpi)
@time coeffs2, MM2 = ri_basis_new(ll)
@show size(coeffs2)

rk2 = rank(coeffs_rpi,rtol = 1e-12)
@test rk1 == rk2 #error here

U, S, V = svd(coeffs_rpi)
coeffs_ind2 = Diagonal(S[1:rk2]) \ (U[:, 1:rk2]' * coeffs2)

Xsym_new = sym_rand_batch(ntest; coeffs=coeffs_ind2, MM=MM2, ll=ll, nn=nn)
rank(Xsym_new; rtol=1e-12)


# # multiset_permutations([-1 -1 1 1], 4)
# # sort([[1, 1], [2, 3], [2, 2], [1, 4]])
# @test rank([coeffs_ind1;coeffs_ind2]) == rk1
