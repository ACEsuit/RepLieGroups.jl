

using Test, RepLieGroups, StaticArrays, Polynomials4ML
using RepLieGroups.O3: ClebschGordan, Rot3DCoeffs, re_basis, 
            _mrange, MRange
using Polynomials4ML: CYlmBasis, index_y, RYlmBasis 
using Polynomials4ML.Testing: print_tf
using LinearAlgebra

##

function eval_basis(ll, Ure, Mll, X)
   @assert length(X) == length(ll)
   @assert all(length.(X) .== 3) 

   TV = promote_type(eltype(Ure), eltype(X[1]))
   val = zeros(TV, size(Ure, 1))
   
   basis = RYlmBasis(sum(ll))
   Ylm = [ evaluate(basis, x) for x in X ] 

   for (i, mm) in enumerate(Mll)
      prod_Ylm = prod( Ylm[j][index_y(l, m)] 
                       for (j, (l, m)) in enumerate(zip(ll, mm)) )
      val += Ure[:, i] * prod_Ylm
   end

   return val 
end

function rand_rot() 
   K = (@SMatrix randn(3,3))
   K = K - K' 
   return exp(K) 
end

##

cgen = Rot3DCoeffs(0)
maxl = [0, 7, 5, 3, 2]
for ν = 2:5
   @info("Testing invariance of coupled basis: L = 0, ν = $ν")
   for ntest = 1:(200 ÷ ν)
      ll = rand(0:maxl[ν], ν)
      if !iseven(sum(ll)); continue; end 
      ll = SVector(ll...)      
      Ure, Mll = re_basis(cgen, ll)
      if size(Ure, 1) == 0; continue; end

      X = [ (@SVector rand(3)) for i in 1:length(ll) ]
      Q = rand_rot() 
      B1 = eval_basis(ll, Ure, Mll, X)
      B2 = eval_basis(ll, Ure, Mll, Ref(Q) .* X)
      # print_tf(@test B1 ≈ B2)
      @show norm(B1 - B2, Inf)
   end
   println()
end

## Task 1: Check the order of the rSH - which convention is used
# NOTE: There are several different ways to define rSH, including the one used in Polynomials4ML.jl
#       Another typical one is something like "[px, py, pz]-convention", which allows the rSH transform
#       like an EuclideanVector. In the following, the A(l) matrix is the transformation from the two 
#       conventions (the one used in Polynomials4ML and the Euclidean one). _ctran(l), on the other hand,
#       is the transformation from cSH to the Euclidean rSH. All those transformations are expected to be 
#       used to construct the coupling coefficients for the rSH ACE.

using SparseArrays
function A(l::Int64)
   if l == 1
      return [0 0 -1; 1 0 0; 0 1 0]'# [0 1 0;0 0 1; -1 0 0]'
   else return I
   end
end

function _ctran(l::Int64,m::Int64,μ::Int64)
   if abs(m) ≠ abs(μ)
      return 0
   elseif abs(m) == 0
      return 1
   elseif m > 0 && μ > 0
      return 1/sqrt(2)
   elseif m > 0 && μ < 0
      return (-1)^m/sqrt(2)
   elseif m < 0 && μ > 0
      return  - im * (-1)^m/sqrt(2)
   else
      return im/sqrt(2)
   end
end

_ctran(l::Int64) = sparse(Matrix{ComplexF64}([ _ctran(l,m,μ) for m = -l:l, μ = -l:l ]))

ctran(l::Int64) = SMatrix{2l+1,2l+1}(A(l) * _ctran(l))

@info("Test whether or not we found a correct transformation from the two rSHs")
basis = RYlmBasis(1)
for ntest = 1:30
   x = @SVector rand(3)
   Ylm = evaluate(basis, x)[2:4]
   Ylm = A(1)' * Ylm
   Q = rand_rot()
   Ylm_r = evaluate(basis, Q * x)[2:4]
   Ylm_r = A(1)' * Ylm_r
    print_tf(@test Q' * Ylm_r ≈ Ylm)
end
println()
