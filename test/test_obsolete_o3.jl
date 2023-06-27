

using Test, RepLieGroups, StaticArrays, Polynomials4ML
using RepLieGroups.O3: ClebschGordan, Rot3DCoeffs, Rot3DCoeffs_real, Rot3DCoeffs_long, 
            re_basis, _mrange, MRange, Ctran, clebschgordan
using Polynomials4ML: CYlmBasis, index_y, RYlmBasis 
using Polynomials4ML.Testing: print_tf
using LinearAlgebra
using WignerD, Rotations, BlockDiagonals

##

function eval_basis(ll, Ure, Mll, X; Real = true)
   @assert length(X) == length(ll)
   @assert all(length.(X) .== 3) 

   # NOTE: It seems that we will not go beyond vector valued functions in this package...?
   _convert = Real ? real : complex # identity
   val = _convert(zeros(typeof(Ure[1]), size(Ure,1)))
   
   if Real
      basis = RYlmBasis(sum(ll))
   else
      basis = CYlmBasis(sum(ll))
   end
   Ylm = [ evaluate(basis, x) for x in X ] 

   for (i, mm) in enumerate(Mll)
      prod_Ylm = prod( Ylm[j][index_y(l, m)] 
                       for (j, (l, m)) in enumerate(zip(ll, mm)) )
      val .+= _convert(Ure[:,i] * prod_Ylm)
   end

   return val 
end

function rand_rot() 
   K = (@SMatrix randn(3,3))
   K = K - K' 
   return exp(K) 
end

##
@info("Testing the correctness of Ctran(L)")
Lmax = 4
basis1 = CYlmBasis(Lmax)
basis2 = RYlmBasis(Lmax)
for L = 0:Lmax
   @info("Testing whether or not we found a correct transformation between cSH to rSH for L = $L")
   for ntest = 1:30
      x = @SVector rand(3)
      Ylm = evaluate(basis1, x)[L^2+1:(L+1)^2]
      Ylm_r = evaluate(basis2, x)[L^2+1:(L+1)^2]
      print_tf(@test norm(Ctran(L)' * Ylm_r - collect(Ylm)) < 1e-12)
   end
   println()
end

@info("Testing the D-matrix for cSH")
Lmax = 4
basis = CYlmBasis(Lmax)
for ntest = 1:30
   local θ
   x = @SVector rand(3)
   θ = rand() * 2pi
   Q = RotXYZ(0, 0, θ)
   Ylm = evaluate(basis, x)
   Ylm_r = evaluate(basis, Q * x)
   for L = 0:Lmax
      YL = Ylm[L^2+1:(L+1)^2]
      YrL = Ylm_r[L^2+1:(L+1)^2]
      D = wignerD(L, 0, 0, θ)
      print_tf(@test norm(D * YrL- collect(YL)) < 1e-12)
   end
end
println()

@info("The cSH D-matrix recursion")
Lmax = 4
θ = rand() * 2pi
Dset = [ wignerD(L, 0, 0, θ) for L = 0:Lmax ]

for ntest = 1:100
   l1 = rand(0:Lmax)
   m1 = rand(-l1:l1)
   μ1 = rand(-l1:l1)
   l2 = rand(0:Lmax-l1)
   m2 = rand(-l2:l2)
   μ2 = rand(-l2:l2)

   val1 = Dset[l1+1][m1+l1+1,μ1+l1+1] * Dset[l2+1][m2+l2+1,μ2+l2+1]
   val2 = 0
   for λ = abs(l1-l2):l1+l2
      if -λ ≤ m1+m2 ≤ λ && -λ ≤ μ1+μ2 ≤ λ
         val2 += clebschgordan(l1,m1,l2,m2,λ,m1+m2) * clebschgordan(l1,μ1,l2,μ2,λ,μ1+μ2) * Dset[λ+1][m1+m2+λ+1,μ1+μ2+λ+1]
      end
   end
   print_tf(@test norm(val1 - val2) < 1e-12)
end
println()
   
@info("Testing the D-matrix for rSH")
Lmax = 4
basis = RYlmBasis(Lmax)
for ntest = 1:30
   local θ
   x = @SVector rand(3)
   θ = rand() * 2pi
   Q = RotXYZ(0, 0, θ)
   Ylm = evaluate(basis, x)
   Ylm_r = evaluate(basis, Q * x)
   for L = 0:Lmax
      YL = Ylm[L^2+1:(L+1)^2]
      YrL = Ylm_r[L^2+1:(L+1)^2]
      D = wignerD(L, 0, 0, θ)
      print_tf(@test norm(Ctran(L) * D * Ctran(L)' * YrL- collect(YL)) < 1e-12)
   end
end
println()

## NOTE: In contradict to my intuition, the recursion for rSH D-matrix is not the same as the cSH one. 
#        The test code in side_tests.jl shows this. And the correct recursion rely not only on a single 
#        previous value but 16 (or 8, if we investigate the property of the new D matrix). Here comes the 
#        correct recursion!
include("side_tests.jl")

@info("The rSH D-matrix recursion")
Lmax = 4
θ = rand() * 2pi
Dset = [ Ctran(L) * wignerD(L, 0, 0, θ) * Ctran(L)' for L = 0:Lmax ]
DDset = [ wignerD(L, 0, 0, θ) for L = 0:Lmax ]

for ntest = 1:100
   l1 = rand(0:Lmax)
   m1 = rand(-l1:l1)
   μ1 = rand(-l1:l1)
   l2 = rand(0:Lmax-l1)
   m2 = rand(-l2:l2)
   μ2 = rand(-l2:l2)

   val1 = Dset[l1+1][m1+l1+1,μ1+l1+1] * Dset[l2+1][m2+l2+1,μ2+l2+1]
   val2 = 0
   for λ = abs(l1-l2):l1+l2
      for m in unique([m1,-m1])
         for mm in unique([m2,-m2])
            for μ in unique([μ1,-μ1])
               for μμ in unique([μ2,-μ2])
                  if abs(m+mm) ≤ λ && abs(μ+μμ) ≤ λ
                     c1 = Ctran(l1,m1,m) * Ctran(l2,m2,mm) * Ctran(l1,μ1,μ)' * Ctran(l2,μ2,μμ)'
                     c2 = clebschgordan(l1,m,l2,mm,λ,m+mm) * clebschgordan(l1,μ,l2,μμ,λ,μ+μμ)
                     # val2 += c1 * DDset[l1+1][m+l1+1,μ+l1+1] * DDset[l2+1][mm+l2+1,μμ+l2+1]
                     # val2 += c1 * c2 * DDset[λ+1][m+mm+λ+1,μ+μμ+λ+1]
                     for p in unique([m+mm,-m-mm])
                       for q in unique([μ+μμ,-μ-μμ])
                          c3 = Ctran(l1+l2,p,m+mm)' * Ctran(l1+l2,q,μ+μμ)
                              val2 += c1 * c2 * c3 * Dset[λ+1][p+λ+1,q+λ+1]
                       end
                     end
                  end
               end
            end
         end
      end
   end
   print_tf(@test norm(val1 - val2) < 1e-12)
end
println()

@info("Equivariance of coupled cSH based basis")  
for L = 0:2
   cgen = Rot3DCoeffs(L)
   maxl = [0, 7, 5, 3, 2]
   for ν = 2:5
      @info("Testing equivariance of coupled cSH based basis: L = $L, ν = $ν")
      for ntest = 1:(200 ÷ ν)
         local θ
         ll = rand(0:maxl[ν], ν)
         if !iseven(sum(ll)+L); continue; end 
         ll = SVector(ll...)      
         Ure, Mll = re_basis(cgen, ll)
         if size(Ure, 1) == 0; continue; end

         X = [ (@SVector rand(3)) for i in 1:length(ll) ]
         θ = rand() * 2pi
         Q = RotXYZ(0, 0, θ)
         B1 = eval_basis(ll, Ure, Mll, X; Real = false)
         B2 = eval_basis(ll, Ure, Mll, Ref(Q) .* X; Real = false)
         # TODO: combine into a single test 
         if L == 0
            print_tf(@test norm(B1 - B2)<1e-12)
         else
            D = wignerD(L, 0, 0, θ)
            print_tf(@test norm(B1 - Ref(D) .* B2)<1e-12)
         end
      end
      println()
   end
end

@info("Equivariance of coupled rSH based basis")  
# TODO: add tests for L = 1, 2, 3, 4
for L = 0:0
   cgen = Rot3DCoeffs_real(L)
   maxl = [0, 7, 5, 3, 2]
   for ν = 2:5
      @info("Testing equivariance of coupled rSH based basis: L = $L, ν = $ν")
      for ntest = 1:(200 ÷ ν)
         local θ
         ll = rand(0:maxl[ν], ν)
         if !iseven(sum(ll)+L); continue; end 
         ll = SVector(ll...)      
         Ure, Mll = re_basis(cgen, ll)
         if size(Ure, 1) == 0; continue; end

         X = [ (@SVector rand(3)) for i in 1:length(ll) ]
         θ = rand() * 2pi
         Q = RotXYZ(0, 0, θ)
         B1 = eval_basis(ll, Ure, Mll, X; Real = true)
         B2 = eval_basis(ll, Ure, Mll, Ref(Q) .* X; Real = true)
         if L == 0
            print_tf(@test norm(B1 - B2)<1e-12)
         else
            D = Ctran(L) * wignerD(L, 0, 0, θ) * Ctran(L)'
            print_tf(@test norm(B1 - Ref(D) .* B2)<1e-12)
         end
      end
      println()
   end
end

@info("Equivariance of coupled cSH based LONG basis")  
for L = 0:2
   cgen = Rot3DCoeffs_long(L)
   maxl = [0, 7, 5, 3, 2]
   for ν = 2:5
      @info("Testing equivariance of coupled cSH based LONG basis: L = $L, ν = $ν")
      for ntest = 1:(200 ÷ ν)
         local θ
         ll = rand(0:maxl[ν], ν)
         if L == 0 
            if !iseven(sum(ll)+L); continue; end 
         end
         ll = SVector(ll...)      
         Ure, Mll = re_basis(cgen, ll)
         if size(Ure, 1) == 0; continue; end

         X = [ (@SVector rand(3)) for i in 1:length(ll) ]
         θ = rand() * 2pi
         Q = RotXYZ(0, 0, θ)

         B1 = eval_basis(ll, Ure, Mll, X; Real = false)
         B2 = eval_basis(ll, Ure, Mll, Ref(Q) .* X; Real = false)
         D = BlockDiagonal([ wignerD(l, 0, 0, θ) for l = 0:L] )
         print_tf(@test norm(B1 - Ref(D) .* B2)<1e-12)
      end
      println()
   end
end

@info("Equivariance of each 'subblock' of the cSH based LONG basis")  
Lmax = 4
cgen = Rot3DCoeffs_long(Lmax)
maxl = [0, 7, 5, 3, 2]
for ntest = 1:30
   ν = rand(2:5)
   ll = rand(0:maxl[ν], ν)
   ll = SVector(ll...)      
   Ure, Mll = re_basis(cgen, ll)
   if size(Ure, 1) == 0; continue; end
   
   X = [ (@SVector rand(3)) for i in 1:length(ll) ]
   θ = rand() * 2pi
   Q = RotXYZ(0, 0, θ)
   
   B1 = eval_basis(ll, Ure, Mll, X; Real = false)
   B2 = eval_basis(ll, Ure, Mll, Ref(Q) .* X; Real = false)
   
   for l = 0:Lmax
      B1l = [ B1[i][Val(l)] for i = 1:length(B1) ]
      B2l = [ B2[i][Val(l)] for i = 1:length(B2) ]
      D = wignerD(l, 0, 0, θ)
      print_tf(@test norm(B1l - Ref(D) .* B2l)<1e-12)
   end
end
