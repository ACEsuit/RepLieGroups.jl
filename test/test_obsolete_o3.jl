using Test, RepLieGroups, StaticArrays, SpheriCart, Combinatorics
using RepLieGroups.O3: Ctran, re_rpe, rpe_basis_new, gram
using PartialWaveFunctions: clebschgordan
using LinearAlgebra
using WignerD, Rotations, BlockDiagonals

include("utils/utils_for_tests.jl")

##
@info("Testing the correctness of Ctran(L)")
Lmax = 4
for L = 0:Lmax
   @info("Testing whether or not we found a correct transformation between cSH to rSH for L = $L")
   for ntest = 1:30
      x = @SVector rand(3)
      Ylm = cYlm(L,x)[L^2+1:(L+1)^2]
      Ylm_r = rYlm(L,x)[L^2+1:(L+1)^2]
      print_tf(@test norm(Ctran(L)' * Ylm_r - collect(Ylm)) < 1e-12)
   end
   println()
end

@info("Testing the D-matrix for cSH")
Lmax = 4
for ntest = 1:30
   local θ, Q
   x = @SVector rand(3)
   θ = rand(3) * 2pi
   Q = RotZYZ(θ...)
   Ylm = cYlm(Lmax,x;method=:LC)
   Ylm_r = cYlm(Lmax,Q * x;method=:LC)
   for L = 0:Lmax
      YL = Ylm[L^2+1:(L+1)^2]
      YrL = Ylm_r[L^2+1:(L+1)^2]
      D = transpose(WignerD.wignerD(L, θ...)) 
      # Here transpose is needed because of the different convention (between ours and WignerD.jl's) of the D-matrix; 
      # Same for the cases below.
      print_tf(@test norm(D * YrL- collect(YL)) < 1e-12)
   end
end
println()

@info("The cSH D-matrix recursion")
Lmax = 4
θ = rand(3) * 2pi
Dset = [ transpose(WignerD.wignerD(L, θ...)) for L = 0:Lmax ]

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
for ntest = 1:30
   local θ, Q
   x = @SVector rand(3)
   θ = rand(3) * 2pi
   Q = RotZYZ(θ...)
   Ylm = rYlm(Lmax, x)
   Ylm_r = rYlm(Lmax, Q * x)
   for L = 0:Lmax
      YL = Ylm[L^2+1:(L+1)^2]
      YrL = Ylm_r[L^2+1:(L+1)^2]
      D = transpose(WignerD.wignerD(L, θ...))
      print_tf(@test norm(Ctran(L) * D * Ctran(L)' * YrL- collect(YL)) < 1e-12)
   end
end
println()

## NOTE: In contradict to my intuition, the recursion for rSH D-matrix is not the same as the cSH one. 
#        The test code in side_tests.jl shows this. And the correct recursion rely not only on a single 
#        previous value but 16 (or 8, if we investigate the property of the new D matrix). Here comes the 
#        correct recursion!
## LZ:   These tests are removed because they are not needed for constructing the rSH based equivariant basis anymore - they still pass, though
# include("side_tests.jl")  

@info("The rSH D-matrix recursion")
Lmax = 4
θ = rand(3) * 2pi
Dset = [ Ctran(L) * transpose(WignerD.wignerD(L, θ...)) * Ctran(L)' for L = 0:Lmax ]
DDset = [ transpose(WignerD.wignerD(L, θ...)) for L = 0:Lmax ]

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
                     c1 = Ctran(m1,m) * Ctran(m2,mm) * Ctran(μ1,μ)' * Ctran(μ2,μμ)'
                     c2 = clebschgordan(l1,m,l2,mm,λ,m+mm) * clebschgordan(l1,μ,l2,μμ,λ,μ+μμ)
                     # val2 += c1 * DDset[l1+1][m+l1+1,μ+l1+1] * DDset[l2+1][mm+l2+1,μμ+l2+1]
                     # val2 += c1 * c2 * DDset[λ+1][m+mm+λ+1,μ+μμ+λ+1]
                     for p in unique([m+mm,-m-mm])
                       for q in unique([μ+μμ,-μ-μμ])
                          c3 = Ctran(p,m+mm)' * Ctran(q,μ+μμ)
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

# The following tests are to be moved, in a new form, to new_rpe_test.jl

@info("Equivariance of coupled cSH & rSH based basis")  
for L = 0:4
   local θ, ll, Ure, Ure_r, U_rpe, U_rpe_r, Mll, Mll_r 
   local X, Q, B1, B2, B3, B4, B5, B6, B7, B8
   local rk, rk_r, ntest
   local BB, BB_r, BB_sym, BB_sym_r

   # generate an nnll list for each L for testing
   lmax = 4
   nmax = 4
   nnll_list = [] 

   for ORD = 2:6
      for ll in with_replacement_combinations(1:lmax, ORD) 
         # 0 or 1 above ?
         if !iseven(sum(ll)+L); continue; end  # This is to ensure the reflection symmetry
         if sum(ll) > 2 * lmax; continue; end 
         for Inn in CartesianIndices( ntuple(_->1:nmax, ORD) )
            nn = [ Inn.I[α] for α = 1:ORD ]
            if sum(nn) > sum(1:nmax); continue; end
            nnll = [ (ll[α], nn[α]) for α = 1:ORD ]
            if !issorted(nnll); continue; end
            push!(nnll_list, (SVector(nn...), SVector(ll...)))
         end
      end
   end

   long_nnll_list = nnll_list 
   short_nnll_list = nnll_list[1:10:end]
   ultra_short_nnll_list = nnll_list[1:100:end]

   verbose = true 

   @info("Using ultra short nnll list for testing the case L = $L")
   nnll_list = ultra_short_nnll_list

   for (itest, (nn, ll)) in enumerate(nnll_list)
      # @show nn, ll
      N = length(ll)
      @assert length(ll) == length(nn)

      Ure, _, _, Mll = re_rpe(nn, ll, L) # cSH based re_basis
      Ure_r, _, _, Mll_r = re_rpe(nn, ll, L; flag = :SpheriCart) # rSH based re_basis
      Urpe, Mll = rpe_basis_new(nn, ll, L) # cSH based rpe_basis
      Urpe_r, Mll_r = rpe_basis_new(nn, ll, L; flag = :SpheriCart) # rSH based rpe_basis

      rk = rank(gram(Urpe), rtol = 1e-12)
      rk_r = rank(gram(Urpe_r), rtol = 1e-12)
      # @info("Check that the two rpe bases, generated by rSH and cSH, have the same dimensionality")
      print_tf(@test rk == rk_r)
      # NOTE: re_basis and rpe_basis share the same Mll / Mll_r

      if norm(Ure) == norm(Ure_r) == 0 ; continue; end # This would mean that both the bases are empty; 
      # @info("Check that the Ure and Ure_r basis span the spaces that have the same dimension, and the dimensionality is the same as the size of Ure and Ure_r (full rank)")
      print_tf(@test size(Ure, 1) == size(Ure_r, 1) == rank(gram(Ure), rtol = 1e-12) == rank(gram(Ure_r), rtol = 1e-12))

      X = [ (@SVector rand(3)) / sqrt(3) for i in 1:length(ll) ]
      θ = rand(3) * 2pi
      Q = RotZYZ(θ...)
      B1 = eval_basis(ll, Ure, Mll, X; Real = false)
      B2 = eval_basis(ll, Ure, Mll, Ref(Q) .* X; Real = false)
      B3 = eval_basis(ll, Ure_r, Mll_r, X; Real = true)
      B4 = eval_basis(ll, Ure_r, Mll_r, Ref(Q) .* X; Real = true)
      B5 = eval_basis(ll, Urpe, Mll, X; Real = false)
      B6 = eval_basis(ll, Urpe, Mll, Ref(Q) .* X; Real = false)
      B7 = eval_basis(ll, Urpe_r, Mll_r, X; Real = true)
      B8 = eval_basis(ll, Urpe_r, Mll_r, Ref(Q) .* X; Real = true)
      # @info("Check the equivariance of the basis")
      # TODO: combine into a single test 
      if L == 0
         print_tf(@test norm(B1 - B2)<1e-12)
         print_tf(@test norm(B3 - B4)<1e-12)
         print_tf(@test norm(B5 - B6)<1e-12)
         print_tf(@test norm(B7 - B8)<1e-12)
      else
         D = transpose(WignerD.wignerD(L, θ...))
         D_r = Ctran(L) * D * Ctran(L)'
         print_tf(@test norm(B1 - Ref(D) .* B2)<1e-12)
         print_tf(@test norm(B3 - Ref(D_r) .* B4)<1e-12)
         print_tf(@test norm(B5 - Ref(D) .* B6)<1e-12)
         print_tf(@test norm(B7 - Ref(D_r) .* B8)<1e-12)
      end

      # @info("Check the linear independence of the basis")
      ntest = 1000

      Xs = make_batch(ntest, length(ll))
      BB = rand_batch(; coeffs=Urpe, MM=Mll, ll=ll, nn=nn, batch = Xs, Real = false)
      print_tf(@test rank(gram(BB); rtol=1e-12) == size(BB,1) == rk)
      BB_r = rand_batch(; coeffs=Urpe_r, MM=Mll_r, ll=ll, nn=nn, batch = Xs, Real = true)
      print_tf(@test rank(gram(BB_r); rtol=1e-12) == size(BB_r,1) == rk_r)

      BB_sym = sym_rand_batch(; coeffs=Urpe, MM=Mll, ll=ll, nn=nn, batch = Xs, Real = false)
      print_tf(@test rank(gram(BB_sym); rtol=1e-12) == size(BB_sym,1) == rk)
      BB_sym_r = sym_rand_batch(; coeffs=Urpe_r, MM=Mll_r, ll=ll, nn=nn, batch = Xs, Real = true)
      print_tf(@test rank(gram(BB_sym_r); rtol=1e-12) == size(BB_sym_r,1) == rk_r)

   end
   println()
end

# NOTE: In the following, we test the LONG equivariant basis (SYYVector), which we don't have in the new version, 
# and it might not be needed anymore. I leave the test code here for now in case we want that back at some point. 

# @info("Equivariance of coupled cSH based LONG basis")  
# for L = 0:2
#    cgen = Rot3DCoeffs_long(L)
#    maxl = [0, 7, 5, 3, 2]
#    for ν = 2:5
#       @info("Testing equivariance of coupled cSH based LONG basis: L = $L, ν = $ν")
#       for ntest = 1:(200 ÷ ν)
#          local θ
#          ll = rand(0:maxl[ν], ν)
#          if L == 0 
#             if !iseven(sum(ll)+L); continue; end 
#          end
#          ll = SVector(ll...)      
#          Ure, Mll = re_basis(cgen, ll)
#          if size(Ure, 1) == 0; continue; end

#          X = [ (@SVector rand(3)) for i in 1:length(ll) ]
#          θ = rand(3) * 2pi
#          Q = RotZYZ(θ...)

#          B1 = eval_basis(ll, Ure, Mll, X; Real = false)
#          B2 = eval_basis(ll, Ure, Mll, Ref(Q) .* X; Real = false)
#          D = BlockDiagonal([ transpose(WignerD.wignerD(l, θ...)) for l = 0:L] )
#          print_tf(@test norm(B1 - Ref(D) .* B2)<1e-12)
#       end
#       println()
#    end
# end

# @info("Testing equivariance of each 'subblock' of the cSH based LONG basis")  
# Lmax = 4
# cgen = Rot3DCoeffs_long(Lmax)
# maxl = [0, 7, 5, 3, 2]
# for ntest = 1:30
#    local ν, ll, Ure, Mll, X, θ, Q, B1, B2
#    ν = rand(2:5)
#    ll = rand(0:maxl[ν], ν)
#    ll = SVector(ll...)      
#    Ure, Mll = re_basis(cgen, ll)
#    if size(Ure, 1) == 0; continue; end
   
#    X = [ (@SVector rand(3)) for i in 1:length(ll) ]
#    θ = rand(3) * 2pi
#    Q = RotZYZ(θ...)
   
#    B1 = eval_basis(ll, Ure, Mll, X; Real = false)
#    B2 = eval_basis(ll, Ure, Mll, Ref(Q) .* X; Real = false)
   
#    for l = 0:Lmax
#       B1l = [ B1[i][Val(l)] for i = 1:length(B1) ]
#       B2l = [ B2[i][Val(l)] for i = 1:length(B2) ]
#       D = transpose(WignerD.wignerD(l, θ...))
#       print_tf(@test norm(B1l - Ref(D) .* B2l)<1e-12)
#    end
# end

# println()