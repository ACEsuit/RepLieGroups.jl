

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
#       conventions (the one used in Polynomials4ML and the Euclidean one), while ctran(l) is the transformation 
#       from cSH to the Euclidean rSH. All those transformations are expected to be used to construct the coupling coefficients for the rSH ACE.

using SparseArrays

# NOTE: There is a way to write the following transformations more elegantly, 
#       i.e. the Condon-Shortley-ish thing;
#       I am keeping this A function for now to see how this is connected to Euclidean rSH.
function A(l::Int64)
   if l == 1
      return [0 0 -1; 1 0 0; 0 1 0]'# [0 1 0;0 0 1; -1 0 0]'
   elseif l == 2
      return [-1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 0 0 0 1 0; 0 0 0 0 1]'
   elseif l == 3
      return [1 0 0 0 0 0 0; 0 -1 0 0 0 0 0; 0 0 -1 0 0 0 0; 0 0 0 -1 0 0 0; 0 0 0 0 -1 0 0; 0 0 0 0 0 1 0; 0 0 0 0 0 0 1]
   elseif l == 4
      return [1 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 1 0; 0 0 1 0 0 0 0 0 0; 0 0 0 0 0 1 0 0 0; 0 0 0 0 -1 0 0 0 0; 0 0 0 -1 0 0 0 0 0; 0 0 0 0 0 0 -1 0 0; 0 -1 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 -1]
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

ctran(l::Int64) = l == 1 ? SMatrix{2l+1,2l+1}([0 1 0;0 0 1; -1 0 0]' * _ctran(l)) : SMatrix{2l+1,2l+1}(_ctran(l))

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

@info("Test whether or not we found a correct transformation from cSH to the Euclidean rSH")
basis = CYlmBasis(1)
for ntest = 1:30
   x = @SVector rand(3)
   Ylm = evaluate(basis, x)[2:4]
   Ylm = ctran(1) * Ylm
   Q = rand_rot()
   Ylm_r = evaluate(basis, Q * x)[2:4]
   Ylm_r = ctran(1) * Ylm_r
   print_tf(@test Q' * Ylm_r ≈ Ylm)
end
println()

## NOTE: Let C_{lm} be the Polynomials4ML rSH, Y_{lm} the Euclidean rSH and Y_l^m, the cSH. 
#        The above then means: (1) A(1)' * C_{lm} = Y_{lm}; (2) Ctran(1) * Y_l^m = Y_{lm} and hence
#        C_{lm} = A(1) * Ctran(1) * Y_l^m. The following code tests it.

using WignerD, Rotations ## Call the two packages for Wigner matrices as we are currently not able to use latest ACE...
Lmax = 4
basis = RYlmBasis(Lmax)
for L = 0:Lmax
   @info("Test whether or not we found a correct transformation from cSH to rSH for L = $L")
   for ntest = 1:30
      x = @SVector rand(3)
      θ = rand() * 2pi
      Q = RotXYZ(0, 0, θ)
      D = wignerD(L, 0, 0, θ)
      Ylm = evaluate(basis, x)[L^2+1:(L+1)^2]
      Ylm = ctran(L)' * A(L)' * Ylm
      Ylm_r = evaluate(basis, Q * x)[L^2+1:(L+1)^2]
      Ylm_r = ctran(L)' * A(L)' * Ylm_r
      print_tf(@test D * Ylm_r ≈ Ylm)
   end
   println()
end
