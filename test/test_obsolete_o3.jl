

using Test, RepLieGroups, StaticArrays, Polynomials4ML
using RepLieGroups.O3: ClebschGordan, Rot3DCoeffs, re_basis, 
            _mrange, MRange
using Polynomials4ML: CYlmBasis, index_y
using Polynomials4ML.Testing: print_tf

##

function eval_basis(ll, Ure, Mll, X)
   @assert length(X) == length(ll)
   @assert all(length.(X) .== 3) 

   TV = promote_type(eltype(Ure), eltype(X[1]))
   val = zeros(TV, size(Ure, 1))
   
   basis = CYlmBasis(sum(ll))
   Ylm = [ evaluate(basis, x) for x in X ] 

   for (i, mm) in enumerate(Mll)
      prod_Ylm = prod( Ylm[j][index_y(l, m)] 
                       for (j, (l, m)) in enumerate(zip(ll, mm)) )
      val += Ure[:, i] * prod_Ylm
   end

   return val 
end

function rand_rot() 
   K = randn(3,3)
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

      X = [ rand(3) for i in 1:length(ll) ]
      Q = rand_rot() 
      B1 = eval_basis(ll, Ure, Mll, X)
      B2 = eval_basis(ll, Ure, Mll, Ref(Q) .* X)
      print_tf(@test B1 ≈ B2)
   end
   println()
end

