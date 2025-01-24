using SpheriCart, StaticArrays, Polynomials4ML
using RepLieGroups.O3: Rot3DCoeffs, coco_dot, re_basis
using RepLieGroups: gram
using Polynomials4ML: complex_sphericalharmonics

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
    basis = complex_sphericalharmonics(maximum(ll))
    Y = [ basis(𝐫) for 𝐫 in Rs ]
 
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
    basis = complex_sphericalharmonics(maximum(ll))
    Y = [ basis(𝐫) for 𝐫 in Rs ]
 
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
 # they will be moved to RepLieGroups soon
 function rpe_basis(A::Union{Rot3DCoeffs}, nn::SVector{N, TN}, ll::SVector{N, Int}) where {N, TN}
    t_re_old = @elapsed Ure, Mre = re_basis(A, ll)
    # @show t_re_old
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