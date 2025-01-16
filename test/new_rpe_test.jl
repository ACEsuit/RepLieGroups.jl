
using SpheriCart, StaticArrays, LinearAlgebra, RepLieGroups, WignerD, 
      Combinatorics, Rotations
using WignerD: wignerD
using RepLieGroups.O3: Rot3DCoeffs, Rot3DCoeffs_real, Rot3DCoeffs_long, coco_dot
using RepLieGroups: gram
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

rand_config(nX::Integer) = [ rand_ball() for _ in 1:nX ]
rand_config(nX::UnitRange) = rand_config(rand(nX))

make_batch(ntest, nX) = [ rand_config(nX) for _ = 1:ntest ] 

# --------------------------------------------------

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

   if size(coeffs,1) == 0
      return zeros(valtype(coeffs), 0)
   end
   BB = zeros(typeof(coeffs[1]), size(coeffs, 1))
   for i_mm = 1:length(MM)
      mm = MM[i_mm]
      ii_lm = [ SpheriCart.lm2idx(ll[α], mm[α]) for α in 1:ORD ]
      BB += coeffs[:, i_mm] * prod( Y[α][ii_lm[α]] * T[α][nn[α]] for α = 1:ORD )
   end 

   return BB
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
   
   BB = zeros(typeof(coeffs[1]), size(coeffs, 1))
   for i_mm = 1:length(MM)
      mm = MM[i_mm]
      ii_lm = [ SpheriCart.lm2idx(ll[α], mm[α]) for α in 1:ORD ]
      BB += coeffs[:, i_mm] * prod( A[ii_lm[α], nn[α]] for α = 1:ORD )
   end 

   return BB
end



function rand_batch(; coeffs, MM, ll, nn, 
                      ntest = 100, 
                      batch = make_batch(ntest, length(ll)) ) 
   if size(coeffs,1) == 0
      return zeros(valtype(coeffs), 0, length(batch))
   end
   BB = complex.(zeros(typeof(coeffs[1]), size(coeffs, 1), length(batch)))
   for (i, Rs) in enumerate(batch)
      BB[:, i] = eval_basis(Rs; coeffs=coeffs, MM=MM, ll=ll, nn=nn) 
   end
   return BB
end

function sym_rand_batch(; coeffs, MM, ll, nn, 
                        ntest = 100, 
                        batch = make_batch(ntest, length(ll)) ) 
   if size(coeffs,1) == 0
      return BB = zeros(valtype(coeffs), 0, length(batch))
   end
   BB = complex.(zeros(typeof(coeffs[1]), size(coeffs, 1), length(batch)))
   for (i, Rs) in enumerate(batch)
      BB[:, i] = eval_sym_basis(Rs; coeffs=coeffs, MM=MM, ll=ll, nn=nn)
   end
   return BB
end

# The following two functions are hacked from the EQM package, just using as reference and for comparison
function rpe_basis(A::Union{Rot3DCoeffs,Rot3DCoeffs_long,Rot3DCoeffs_real}, nn::SVector{N, TN}, ll::SVector{N, Int}) where {N, TN}
   t_re_old = @elapsed Ure, Mre = O3.re_basis(A, ll)
   @show t_re_old
   G = _gramian(nn, ll, Ure, Mre)
   S = svd(G)
   rk = rank(Diagonal(S.S); rtol =  1e-7)
   Urpe = S.U[:, 1:rk]'
   return Diagonal(sqrt.(S.S[1:rk])) * Urpe * Ure, Mre
end


function _gramian(nn, ll, Ure, Mre)
   N = length(nn)
   nre = size(Ure, 1)
   G = zeros(Complex{Float64}, nre, nre)
   for σ in permutations(1:N)
      if (nn[σ] != nn) || (ll[σ] != ll); continue; end
      for (iU1, mm1) in enumerate(Mre), (iU2, mm2) in enumerate(Mre)
         if mm1[σ] == mm2
            for i1 = 1:nre, i2 = 1:nre
               G[i1, i2] += coco_dot(Ure[i1, iU1], Ure[i2, iU2])
            end
         end
      end
   end
   return G
end


##
# CASE 2: 4-correlations
for L = 0:4
   @info("Testing L = $L")
   cc = Rot3DCoeffs(L)
   # now we fix an ll = (l1, l2, l3) triple ask for all possible linear combinations 
   # of the tensor product basis   Y[l1, m1] * Y[l2, m2] * Y[l3, m3] * Y[l4, m4]
   # that are invariant under O(3) rotations.
   # ll = SA[1,2,2,2,3]
   # ll_list = [SA[2,2,2,2], SA[2,2,2,2], SA[2,2,2,4], SA[2,2,3,3], SA[1,1,2,2,2], SA[1,1,2,2,2], SA[1,2,2,2,3], SA[2,2,2,2,2] ]
   # nn_list = [SA[1,1,1,2], SA[1,1,2,3], SA[1,1,1,1], SA[1,1,1,1], SA[1,2,1,1,1], SA[2,2,1,1,2], SA[1,1,1,1,1], SA[1,1,1,1,1] ]

   

   lmax = 4
   nmax = 4
   nnll_list = [] 

   for ORD = 2:6
      for ll in with_replacement_combinations(1:lmax, ORD) 
         # 0 or 1 above ?
         if !iseven(sum(ll)+L); continue; end 
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

   # @info("Using ultra short nnll list for testing")
   # nnll_list = ultra_short_nnll_list

   @info("Using short nnll list for testing")
   nnll_list = short_nnll_list

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
         @test rk1 == rk2
         
         Xsym_new = rand_batch(; coeffs=coeffs_ind2, MM=MM2, ll=ll, nn=nn, batch=RR) #this is symmetric
         @test rank(gram(Xsym_new); rtol=1e-12) == rk2

         # NOTE FROM CO: same batch is used so can compare!!!
         # @show rank(Xsym) 
         # @show rank(Xsym_new)
         # @show rank([Xsym; Xsym_new], rtol = 1e-12)
         fRs1 = eval_basis(Rs; coeffs = coeffs_ind2, MM = MM2, ll = ll, nn = nn)
         fRs1Q = eval_basis(QRs; coeffs = coeffs_ind2, MM = MM2, ll = ll, nn = nn)
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
         @test rank(gram([coeffsp1;coeffsp2]); rtol=1e-12) == rank(gram(coeffsp2); rtol=1e-12) == rank(gram(coeffsp2); rtol=1e-12)


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