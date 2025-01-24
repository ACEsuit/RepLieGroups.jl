
using StaticArrays, LinearAlgebra, RepLieGroups, WignerD, Combinatorics, Rotations#, Polynomials4ML
using WignerD: wignerD
using RepLieGroups.O3: Rot3DCoeffs
using Test

include("../src/utils/utils_for_tests.jl")

# Test the new RPE basis up to L = 4
for L = 0:4
   @info("Testing L = $L")
   cc = Rot3DCoeffs(L)

   # generate a nnll_list for testing
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
      # t_re_old = @elapsed O3.re_basis(cc, ll)
      t_rpe_old = @elapsed coeffs_ind1_origin, MM1_origin = rpe_basis(cc, nn, ll) # This is the rpe coupling coefficients in EQM, which is our reference
      rk1 = rank(gram(coeffs_ind1_origin); rtol=1e-12) # rank of the reference coupling coefficients

      # coeffs1, MM1 = O3.re_basis(cc, ll)
      # nbas_ri1 = size(coeffs1, 1)
      # rk1 = rank(gram(coeffs1); rtol=1e-12)
      # U, S, V = svd(gram(coeffs1))
      # coeffs_ind1 = Diagonal(S[1:rk1]) * (U[:, 1:rk1]' * coeffs1)
      # @test sort(MM1) == sort(MM1_origin)

      # NOTE: Such constructed coeff_ind1 should span a larger space as the original coeffs1 
      # which can be seen by comparing the functions `gram`` and `_gramian`
      # However, the function `gram`` should work for the FMatrix as the permutation has been taken care of.

      # @test rank(gram(coeffs_ind1_origin); rtol=1e-12) <= rank(gram(coeffs_ind1); rtol=1e-12)

      Rs = rand_config(length(ll))
      θ = rand(3) * 2pi
      Q = RotZYZ(θ...)
      D = transpose(wignerD(L, θ...)) 
      QRs = [Q*Rs[i] for i in 1:length(Rs)]
      # fRs1 = eval_basis(Rs; coeffs = coeffs_ind1, MM = MM1, ll = ll, nn = nn)
      # fRs1Q = eval_basis(QRs; coeffs = coeffs_ind1, MM = MM1, ll = ll, nn = nn)
      # @test norm(fRs1 - Ref(D) .* fRs1Q) < 1e-15
      fRs1 = eval_basis(Rs; coeffs = coeffs_ind1_origin, MM = MM1_origin, ll = ll, nn = nn)
      fRs1Q = eval_basis(QRs; coeffs = coeffs_ind1_origin, MM = MM1_origin, ll = ll, nn = nn)
      # @info("Testing the equivariance of the old RPE basis")
      L == 0 ? (@test norm(fRs1 - fRs1Q) < 1e-14) : (@test norm(fRs1 - Ref(D) .* fRs1Q) < 1e-14)

      ntest = 1000

      RR = make_batch(ntest, length(ll))

      X = rand_batch(; coeffs=coeffs_ind1_origin, MM=MM1_origin, ll=ll, nn=nn, batch = RR)
      @test rank(gram(X); rtol=1e-12) == size(X,1)

      Xsym = sym_rand_batch(; coeffs=coeffs_ind1_origin, MM=MM1_origin, ll=ll, nn=nn, batch = RR)
      @test rank(gram(Xsym); rtol=1e-12) == rk1

      # if RepLieGroups.SetLl_new(ll,L) |> length != 0
      if rk1 > 0
         # Version GD 
         # rewritten as a new interface
         # t_re = @elapsed re_rpe(nn,ll,L) # this is slightly longer than the new re, because it computes also the FMatrix for RPE
         t_rpe = @elapsed coeffs_ind2, MM2 = rpe_basis_new(nn,ll,L)
         # computes the RI coupling coefs and RPI coefs at the same time

         # rk2 = rank(gram(coeffs2),rtol = 1e-12)
         rk2 = rank(gram(coeffs_ind2),rtol = 1e-12)
         # @info("Testing rank_old = rank_new")
         @test rk1 == rk2
         
         Xsym_new = rand_batch(; coeffs=coeffs_ind2, MM=MM2, ll=ll, nn=nn, batch=RR) #this is symmetric
         @test rank(gram(Xsym_new); rtol=1e-12) == rk2

         # NOTE FROM CO: same batch is used so can compare!!!
         fRs1 = eval_basis(Rs; coeffs = coeffs_ind2, MM = MM2, ll = ll, nn = nn)
         fRs1Q = eval_basis(QRs; coeffs = coeffs_ind2, MM = MM2, ll = ll, nn = nn)
         # @info("Testing the equivariance of the new RPE basis")
         L == 0 ? (@test norm(fRs1 - fRs1Q) < 1e-14) : (@test norm(fRs1 - Ref(D) .* fRs1Q) < 1e-14)
         # Up to here, we checked that coeff_ind1_origin and coeff_ind2 has the same rank and span the space with correct equivariance, 
         # and hence they span the same space.

         P1 = sortperm(MM1_origin)
         P2 = sortperm(MM2)
         MMsorted1 = MM1_origin[P1]
         MMsorted2 = MM2[P2]
         # check that same mm values
         @test MMsorted1 == MMsorted2

         coeffsp1 = coeffs_ind1_origin[:,P1]
         coeffsp2 = coeffs_ind2[:,P2]

         # Check that coefficients span same space
         # @info("Testing that the old and new coupling coefficients span the same space")
         @test rank(gram([coeffsp1;coeffsp2]); rtol=1e-12) == rank(gram(coeffsp1); rtol=1e-12) == rank(gram(coeffsp2); rtol=1e-12)


         # Do the rand batch on the same set of points
         ORD = length(ll) # length of each group 
         BB1 = complex.(zeros(typeof(coeffs_ind1_origin[1]), size(coeffs_ind1_origin, 1), ntest))
         BB2 = complex.(zeros(typeof(coeffs_ind2[1]), size(coeffs_ind2, 1), ntest))
         for i = 1:ntest 
            # construct a random set of particles with 𝐫 ∈ ball(radius=1)
            Rs = [ rand_ball() for _ in 1:ORD ]
            BB1[:, i] = eval_basis(Rs; coeffs=coeffs_ind1_origin, MM=MM1_origin, ll=ll, nn=nn) 
            BB2[:, i] = eval_basis(Rs; coeffs=coeffs_ind2, MM=MM2, ll=ll, nn=nn) 
         end

         # Check that values span same space
         @test rank(gram([BB1;BB2]); rtol=1e-11) == rank(gram(BB1); rtol=1e-11) == rank(gram(BB2); rtol=1e-11)

         if verbose 
            # @info("Test $itest: t_re_old = $t_re_old, t_re = $t_re")
            @info("Test $itest: t_rpe_old = $t_rpe_old, t_rpe = $t_rpe")
         else 
            print(".")
         end
      end
   end
end