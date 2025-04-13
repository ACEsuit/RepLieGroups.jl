
using StaticArrays, LinearAlgebra, RepLieGroups, WignerD, Combinatorics, Rotations#, Polynomials4ML
using WignerD: wignerD
using Polynomials4ML.Testing: print_tf
using RepLieGroups.O3: Rot3DCoeffs_real, Ctran
using Test

# Test the new RPE basis up to L = 4
for L = 0:4
   @info("Testing L = $L")
   cc = Rot3DCoeffs_real(L)

   # generate a nnll_list for testing
   lmax = 4
   nmax = 4
   nnll_list = [] 

   for ORD = 2:5
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
   @show length(long_nnll_list)
   @show length(short_nnll_list)
   @show length(ultra_short_nnll_list)

   verbose = true 

   @info("Using ultra short nnll list for testing")
   nnll_list = ultra_short_nnll_list

   # @info("Using short nnll list for testing")
   # nnll_list = short_nnll_list

   # @info("Using long nnll list for testing")
   # nnll_list = long_nnll_list


   for (itest, (nn, ll)) in enumerate(nnll_list)
      N = length(ll)
      @assert length(ll) == length(nn)

      # random configurations
      Rs = rand_config(length(ll))
      θ = rand(3) * 2pi
      Q = RotZYZ(θ...)
      D = Ctran(L) * transpose(wignerD(L, θ...)) * Ctran(L)'
      QRs = [Q*Rs[i] for i in 1:length(Rs)]

      # tests for the re_basis, for L = 0 only since the re_basis is not implemented for L > 0 in the old version
      # @info("Testing the re_basis")
      if L == 0
         cc = Rot3DCoeffs_real(L)
         t_re = @elapsed coeffs_ind, _, _, MM = re_rpe(nn,ll,L; flag = :SpheriCart)
         t_re_old = @elapsed coeffs_ind2, MM2 = re_basis(cc, ll)
         rk1 = rank(gram(coeffs_ind),rtol = 1e-12)
         rk2 = rank(gram(coeffs_ind2),rtol = 1e-12)
         @test rk1 == rk2

         fRs1 = eval_basis(Rs; coeffs = coeffs_ind, MM = MM, ll = ll, nn = nn, Real = true)
         fRs1Q = eval_basis(QRs; coeffs = coeffs_ind, MM = MM, ll = ll, nn = nn, Real = true)
         @test norm(fRs1 - fRs1Q) < 1e-14
         
         fRs1 = eval_basis(Rs; coeffs = coeffs_ind2, MM = MM2, ll = ll, nn = nn, Real = true)
         fRs1Q = eval_basis(QRs; coeffs = coeffs_ind2, MM = MM2, ll = ll, nn = nn, Real = true)
         @test norm(fRs1 - fRs1Q) < 1e-14

         P1 = sortperm(MM)
         P2 = sortperm(MM2)
         MMsorted1 = MM[P1]
         MMsorted2 = MM2[P2]
         # check that same mm values
         @test MMsorted1 == MMsorted2

         coeffsp1 = coeffs_ind[:,P1]
         coeffsp2 = coeffs_ind2[:,P2]

         # Check that coefficients span same space
         # @info("Testing that the old and new coupling coefficients span the same space")
         @test rank(gram([coeffsp1;coeffsp2]); rtol=1e-12) == rank(gram(coeffsp1); rtol=1e-12) == rank(gram(coeffsp2); rtol=1e-12)


         # Do the rand batch on the same set of points
         ORD = length(ll) # length of each group 
         ntest = 1000
         BB1 = complex.(zeros(typeof(coeffs_ind[1]), size(coeffs_ind, 1), ntest))
         BB2 = complex.(zeros(typeof(coeffs_ind2[1]), size(coeffs_ind2, 1), ntest))
         for i = 1:ntest 
            # construct a random set of particles with 𝐫 ∈ ball(radius=1)
            Rs = [ rand_ball() for _ in 1:ORD ]
            BB1[:, i] = eval_basis(Rs; coeffs=coeffs_ind, MM=MM, ll=ll, nn=nn) 
            BB2[:, i] = eval_basis(Rs; coeffs=coeffs_ind2, MM=MM2, ll=ll, nn=nn) 
         end

         # Check that values span same space
         @test rank(gram([BB1;BB2]); rtol=1e-11) == rank(gram(BB1); rtol=1e-11) == rank(gram(BB2); rtol=1e-11)
         if verbose 
            # @info("Test $itest: t_re_old = $t_re_old, t_re = $t_re")
            @info("Test $itest: t_re_old = $t_re_old, t_re = $t_re")
         else 
            print(".")
         end

      end

      # tests for rpe_basis: we do not have a reference before, so the thing I checked here is 
      # whether we have the correct symmetry and whether the resulting space have the same dimensionality
      # as the one with the complex spherical harmonics

      t_rpe_real = @elapsed coeffs_ind, MM = rpe_basis_new(nn,ll,L; flag = :SpheriCart)
      rk = rank(gram(coeffs_ind),rtol = 1e-12)
      # @show rk

      Rs = rand_config(length(ll))
      θ = rand(3) * 2pi
      Q = RotZYZ(θ...)
      D = Ctran(L) * transpose(wignerD(L, θ...)) * Ctran(L)'
      QRs = [Q*Rs[i] for i in 1:length(Rs)]

      fRs1 = eval_basis(Rs; coeffs = coeffs_ind, MM = MM, ll = ll, nn = nn, Real = true)
      fRs1Q = eval_basis(QRs; coeffs = coeffs_ind, MM = MM, ll = ll, nn = nn, Real = true)
      # @info("Testing the equivariance of the new RPE basis")
      L == 0 ? (@test norm(fRs1 - fRs1Q) < 1e-14) : (@test norm(fRs1 - Ref(D) .* fRs1Q) < 1e-14) 

      ntest = 1000

      RR = make_batch(ntest, length(ll))
      # Check the linear independence of the basis
      
      X = rand_batch(; coeffs=coeffs_ind, MM=MM, ll=ll, nn=nn, batch = RR, Real = true)
      print_tf(@test rank(gram(X); rtol=1e-12) == size(X,1))

      Xsym = sym_rand_batch(; coeffs=coeffs_ind, MM=MM, ll=ll, nn=nn, batch = RR, Real = true)
      print_tf(@test rank(gram(Xsym); rtol=1e-12) == rk)

      # Check the rank of the space that rSH and cSH span - even if the rank is zero 
      t_rpe_complex = @elapsed coeffs_ind2, MM2 = rpe_basis_new(nn,ll,L; flag = :cSH)
      rk2 = rank(gram(coeffs_ind2),rtol = 1e-12)
      print_tf(@test rk == rk2)
      if L == 0
         println()
      end

   end
   if L != 0
      println()
   end
end