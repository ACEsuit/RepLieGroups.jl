module O3

import RepLieGroups

using PartialWaveFunctions
using Combinatorics
using LinearAlgebra
using StaticArrays

export re_basis_new, ri_basis_new, ind_corr_s1, ind_corr_s2, MatFmi, ML0, MatFmi2, ri_rpi, re_rpe, rpe_basis_new, re_semi_pi


# ------------------------------------------------------- 

## NOTE: Ctran(L) is the transformation matrix from rSH to cSH. More specifically, 
#        if we write Polynomials4ML rSH as R_{lm} and cSH as Y_{lm} and their corresponding 
#        vectors of order L as R_L and Y_L, respectively. Then R_L = Ctran(L) * Y_L.
#        This suggests that the "D-matrix" for the Polynomials4ML rSH is Ctran(l) * D(l) * Ctran(L)', 
#        where D, the D-matrix for cSH. This inspires the following new CG recursion.

# transformation matrix from RSH to CSH for different conventions
function Ctran(i::Int64,j::Int64;convention = :SpheriCart)
	if convention == :cSH
		return i == j
	end
	
	order_dict = Dict(:SpheriCart => [1,2,3,4], :CondonShortley => [4,3,2,1], :FHIaims => [4,2,3,1])
	val_list = [(-1)^(i), im, (-1)^(i+1)*im, 1] ./ sqrt(2)
	if abs(i) != abs(j)
		return 0 
	elseif i == j == 0
		return 1
	elseif i > 0 && j > 0
		return val_list[order_dict[convention][1]]
	elseif i < 0 && j < 0
		return val_list[order_dict[convention][2]]
	elseif i < 0 && j > 0
		return val_list[order_dict[convention][3]]
	elseif i > 0 && j < 0
		return val_list[order_dict[convention][4]]
	end
end

Ctran(l::Int64; convention = :SpheriCart) = sparse(Matrix{ComplexF64}([ Ctran(m,μ;convention=convention) for m = -l:l, μ = -l:l ])) |> dropzeros


# -----------------------------------------------------

# The generalized Clebsch Gordan Coefficients; variables of this function are 
# fully inherited from the first ACE paper. 
function GCG(l::SVector{N,Int64},m::SVector{N,Int64},L::SVector{N,Int64},M_N::Int64;flag=:cSH) where N
    # @assert -L[N] ≤ M_N ≤ L[N] 
    if m_filter(m, M_N;flag=flag) == false || L[1] < abs(m[1])
        return 0.
    end

    if flag == :cSH
        M = m[1]
        C = 1.
        for k in 2:N
            if L[k] < abs(M+m[k])
                return 0.
            else
                C *= PartialWaveFunctions.clebschgordan(L[k-1],M,l[k],m[k],L[k],M+m[k])
                M += m[k]
            end
        end
        return C
    else
        C = 0.
        for M in signed_m(M_N)
            ext_mset = filter( x -> sum(x) == M, signed_mmset(m) )
        
            for mm in ext_mset
                mm = SA[mm...]
                @assert sum(mm) == M
                C_loc = GCG(l,mm,L,M;flag=:cSH)
                coeff = Ctran(M_N,M;convention=flag)' * prod( Ctran(m[i],mm[i];convention=flag) for i in 1:N )
                C_loc *= coeff
                C += C_loc
            end
        end
        return abs(C - real(C)) < 1e-12 ? real(C) : C # We actually expect real values 
    end

end

# Only when M_N = sum(m) can the CG coefficient be non-zero, so when missing M_N, we return either 
# (1)the full CG coefficient given l, m and L, as a rank 1 vector; 
# (2)or the only one element that can possibly be non-zero on the above vector.
# I suspect that the first option will not be used anyhow, but I keep it for now.
function GCG(l::SVector{N,Int64},m::SVector{N,Int64},L::SVector{N,Int64};vectorize::Bool=true,flag=:cSH) where N 
    if flag == :cSH
        return vectorize ? GCG(l,m,L,sum(m);flag=flag) * Float64.(I(2L[N]+1)[sum(m)+L[N]+1,:]) : GCG(l,m,L,sum(m);flag=flag)
    else
        if vectorize == false && L[N] != 0
            error("For the rSH basis, the CG coefficient is always a vector execpt for the case of L=0.")
        else
            return L[N] == 0 ? GCG(l,m,L,L[N];flag=flag) : SA[[ GCG(l,m,L,M_N;flag=flag) for M_N in -L[N]:L[N] ]...] 
        end
    end
end

# Function that returns a L set given an `l`. The elements of the set start with l[1] and end with L. 
function SetLl(l::SVector{N,Int64}, L::Int64) where N
    T = typeof(l)
    if N==1
        return l[1] == L ? [T(l[1])] : Vector{T}[]
    elseif N==2        
        return abs(l[1]-l[2]) ≤ L ≤ l[1] + l[2] ? [T(l[1],L)] : Vector{T}[]
    end
    
    set = [ [l[1];] ]
    for k in 2:N
        set_tmp = set
        set = Vector{Any}[]
        for a in set_tmp
            if k < N
                for b in abs(a[k-1]-l[k]):a[k-1]+l[k]
                    push!(set, [a; b])
                end
            elseif k == N
                if (abs.(a[N-1]-l[N]) <= L)&&(L <= (a[N-1]+l[N]))
                    push!(set, [a; L])
                end
            end
        end
    end  

    return T.(set)
end

SetLl(l::SVector{N,Int64}) where N = union([SetLl(l, L) for L in 0:sum(l)]...)

function Sn(nn,ll)
    # should assert that lexicographical order
    N = length(ll)
    @assert length(ll) == length(nn)
    perm_indices = [1]
    for i in 2:N
        if ll[i] != ll[perm_indices[end]] || nn[i] != nn[perm_indices[end]]
            push!(perm_indices,i)
        end
    end
    return [perm_indices;N+1]
end

function submset(lmax, lth)
    # lmax stands for the l value of the subsection while lth is the length of this subsection
    if lth == 1
        return [[l] for l in -lmax:lmax]
    else
        tmp = submset(lmax, lth-1)
        mset = Vector{Vector{Int64}}([])
        for t in tmp
            set = identity.([[t..., l] for l in t[end]:lmax])
            push!(mset, set...)
        end
    end
    return mset
end

signed_m(m) = unique([m,-m]) # The set of integers that has the same absolute value as m
signed_mmset(m) = Iterators.product([signed_m(m[i]) for i in 1:length(m)]...) |> collect # The set of vectors whose i-th element has the same absolute value as m[i] for all i

function m_filter(mm::Union{Vector{Int64},SVector{N,Int64}}, k::Int64; flag=:cSH) where N
    if flag == :cSH
        return sum(mm) == k
    else
        # for the rSH, the criterion is that whether there exists a combinition of [+/- m_i]_i, such that the sum of the combination equals to k
     mmset = signed_mmset(mm)
     for m in mmset
      if sum(m) == k
       return true
      end
     end
     return false
    end
end

# Function that generates the set of ordered m's given `n` and `l` with sum of m's equaling to k.
function m_generate(n::T,l::T,L,k;flag=:cSH) where T
    @assert abs(k) ≤ L
    S = Sn(n,l)
    Nperm = length(S)-1
    ordered_mset = [submset(l[S[i]], S[i+1]-S[i]) for i = 1:Nperm]
    MM = []
    Total_length = 0
    for m_ord in Iterators.product(ordered_mset...)
        m_ord_reshape = vcat(m_ord...)
        if m_filter(m_ord_reshape, k; flag = flag)
            class_m = vcat(Iterators.product([multiset_permutations(m_ord[i], S[i+1]-S[i]) for i in 1:Nperm]...)...)
            push!(MM, [vcat(mm...) for mm in class_m])
            Total_length += length(class_m)
        end
    end
    return [ T.(MM[i]) for i = 1:length(MM) ], Total_length
end

# Function that generates the set of ordered m's given `n` and `l` with the abosolute sum of m's being smaller than L.
m_generate(n,l,L;flag=:cSH) = union([m_generate(n,l,L,k;flag)[1] for k in -L:L]...), sum(length.(union([m_generate(n,l,L,k;flag)[1] for k in -L:L]...))) # orginal version: sum(m_generate(n,l,L,k;flag)[2] for k in -L:L), but this cannot be true anymore b.c. the m_classes can intersect


# Function that generates the coupling coefficient of the RE basis given `n` and `l`., the FMatrix for generating the RPE basis is also generated here.
function re_rpe(n::SVector{N,Int64},l::SVector{N,Int64},L::Int64;flag=:cSH) where N
    Lset = SetLl(l,L)
    r = length(Lset)
    T = L == 0 ? Float64 : SVector{2L+1,Float64}
    if r == 0 
        return zeros(T, 1, 1), zeros(T, 1, 1), [zeros(Int,N)], [zeros(Int,N)]
    else 
        MMmat, size_m = m_generate(n,l,L;flag=flag) # classes of m's
        FMatrix=zeros(T, r, length(MMmat)) # Matrix containing f(m,i)
        UMatrix=zeros(T, r, size_m) # Matrix containing the the coupling coefficients D
        MM = [] # all possible m's
        for i in 1:r
            c = 0
            for (j,m_class) in enumerate(MMmat)
                for m in m_class
                    c += 1
                    cg_coef = GCG(l,m,Lset[i];vectorize=(L!=0),flag=flag)
                    FMatrix[i,j]+= cg_coef
                    UMatrix[i,c] = cg_coef
                end
            end
            @assert c==size_m
        end 
        for m_class in MMmat
            for m in m_class
                push!(MM,m)
            end
        end      
    end
    return UMatrix, FMatrix, MMmat, MM
end

function gram(X::Matrix{SVector{N,T}}) where {N,T}
    G = zeros(T, size(X,1), size(X,1))
    for i = 1:size(X,1)
       for j = i:size(X,1)
          G[i,j] = sum(dot(X[i,t], X[j,t]) for t = 1:size(X,2))
          i == j ? nothing : (G[j,i]=G[i,j]')
       end
    end
    return G
 end

 gram(X::Matrix{<:Number}) = X * X'

function rpe_basis_new(nn::SVector{N, Int64}, ll::SVector{N, Int64}, L::Int64; flag = :cSH) where N
    t_re = @elapsed UMatrix, FMatrix, MMmat, MM = re_rpe(nn, ll, L; flag = flag) # time of constructing the re_basis
    # @show t_re # should be removed in the final version
    U, S, V = svd(gram(FMatrix))
    rk = findall(x -> x > 1e-10, S) |> length # rank(Diagonal(S); rtol =  1e-12) # Somehow rank is not working properly here - also this line is faster than sum(S.>1e-12)
    return Diagonal(S[1:rk].^(-1/2)) * (U[:, 1:rk]' * UMatrix), MM
 end

 # ============================ RE_SEMI_PI basis ============================
function swap(xx::SVector{N,T},i::Int64,j::Int64) where {N, T} 
    i, j = sort([i,j])
    return i == j ? xx : SA[xx[1:i-1]..., xx[j], xx[i+1:j-1]..., xx[i], xx[j+1:end]...]
end

function swap(xx::SVector{N,T},i::Vector{Int64},j::Vector{Int64}) where {N, T} 
   for k in 1:length(i)
       xx = swap(xx,i[k],j[k])
   end
   return xx
end

function pick(set,n; ordered = true)
    if n == 1
        return set
    end
    tmp = []
    for i = 1:length(set)
       push!(tmp, [ [mm; set[i]] for mm in pick([set[1:i-1]..., set[i+1:end]...], n-1) ]... ) 
    end
    return ordered ? unique(sort.(tmp)) : unique(tmp)
end

 # Given nn and ll, generate the Ltot RE basis which are PI for the first N1 variables and also PI for the rest
function re_semi_pi(nn::SVector{N,Int64},ll::SVector{N,Int64},Ltot::Int64,N1::Int64) where N
    @assert 0 < N1 < N
 
    ll1 = SA[ll[1:N1]...]
    ll2 = SA[ll[N1+1:end]...]
    nn1 = SA[nn[1:N1]...]
    nn2 = SA[nn[N1+1:end]...]
 
    m_class = m_generate(nn,ll,Ltot)[1]
    MM = []
    for i = 1:length(m_class)
       for j = 1:length(m_class[i])
          push!(MM, m_class[i][j])
       end
    end
    MM = identity.(MM)
    MM_dict = Dict(MM[i] => i for i = 1:length(MM))
    T = Ltot == 0 ? Float64 : SVector{2Ltot+1, Float64}
    C_re_semi_pi = []
    counter = 0
    for L1 in 0:sum(ll1)
       for L2 in abs(L1-Ltot):minimum([L1+Ltot,sum(ll2)])
          # if isodd(L1+L2+Ltot); continue; end # This is wrong - all orders of L1 L2 are needed
          # global C1, _,_, M1 = re_rpe(nn1,ll1,L1)
          # global C2, _,_, M2 = re_rpe(nn2,ll2,L2)
          C1,M1 = rpe_basis_new(nn1,ll1,L1)
          C2,M2 = rpe_basis_new(nn2,ll2,L2)
          # global Basis_func = Ltot == 0 ? zeros(Float64, size(C1,1),size(C2,1),length(M1)*length(M2)) : zeros(SVector{2Ltot+1, Float64}, size(C1,1),size(C2,1),length(M1)*length(M2))
          for i1 in 1:size(C1,1)
             for i2 in 1:size(C2,1)
                cc = [ zero(T) for _ = 1:length(MM) ]
                for (k1,m1) in enumerate(M1)
                   for (k2,m2) in enumerate(M2)
                      if sum(m1) == sum(m2) == 0 && isodd(L1+L2+Ltot); continue; end # That is because C^{Ltot,0}_{L1,0,L2,0} = 0 for odd L1+L2+Ltot
                      if abs(sum(m1)+sum(m2))<=Ltot
                         k = MM_dict[SA[m1...,m2...]] # findfirst(m -> m == SA[m1...,m2...], MM)
                         # @assert !isnothing(k)
                         # Basis_func[i1,i2,counter] = Ltot == 0 ? 
                         #       clebschgordan(L1,sum(m1),L2,sum(m2),Ltot,sum(m1)+sum(m2))*C1[i1,k1][sum(m1)+L1+1]*C2[i2,k2][sum(m2)+L2+1] :
                         #       clebschgordan(L1,sum(m1),L2,sum(m2),Ltot,sum(m1)+sum(m2))*C1[i1,k1][sum(m1)+L1+1]*C2[i2,k2][sum(m2)+L2+1]*I(2Ltot+1)[sum(m1)+sum(m2)+Ltot+1,:]
                         cc[k] = Ltot == 0 ? clebschgordan(L1,sum(m1),L2,sum(m2),Ltot,sum(m1)+sum(m2))*C1[i1,k1][sum(m1)+L1+1]*C2[i2,k2][sum(m2)+L2+1] :
                                             clebschgordan(L1,sum(m1),L2,sum(m2),Ltot,sum(m1)+sum(m2))*C1[i1,k1][sum(m1)+L1+1]*C2[i2,k2][sum(m2)+L2+1]*I(2Ltot+1)[sum(m1)+sum(m2)+Ltot+1,:] 
                      end
                   end
                end
                if norm(cc) > 1e-12
                    push!(C_re_semi_pi, cc) # each element of C_re_semi_pi is a row of the final UMatrix
                    counter += 1
                else
                    @warn("zero dropped") # If we have some zero basis, the code will warn us
                end
             end
          end
          # BB = reshape(Basis_func,size(C1,1)*size(C2,1),length(M1)*length(M2))
          # total_rank += rank(gram(BB))
          # @show size(BB), rank(gram(BB))
       end
    end
    @assert length(C_re_semi_pi) == counter
    C_re_semi_pi = identity.([C_re_semi_pi[i][j] for i = 1:counter, j = 1:length(MM)])
 
    return C_re_semi_pi, MM
 end

 function rpe_basis_new(nn::SVector{N, Int64}, ll::SVector{N, Int64}, L::Int64, N1::Int64; symmetrization_method = :kernel) where N
    nice_partition = Sn(nn,ll).-1 # a list of partitions that gives non-intersecting sets
    if length(nice_partition) > 2 && N1 in nice_partition[2:end-1]
        @assert length( intersect([(nn[i],ll[i]) for i = 1:N1], [(nn[i],ll[i]) for i = N1+1:N]) ) == 0
        println("Non-intersect partition - return directly rpe")
        println()
        return re_semi_pi(nn,ll,L,N1)
    else
        C_re_semi_pi, MM = re_semi_pi(nn,ll,L,N1)
        if size(C_re_semi_pi,1) == 0
            return C_re_semi_pi, MM
        end
        MM_dict = Dict(MM[i] => i for i = 1:length(MM))

        # Last symmetrization
        if symmetrization_method == :explicit
            println("Two groups intersect - explicit symmetrization is to be performed")
            println()
            n_block = findfirst(n -> n > N1, nice_partition)
            N_init = n_block == nothing ? 1 : nice_partition[n_block-1]+1
            N_final = n_block == nothing ? N : nice_partition[n_block]
            
            # final symmetrization
            C_new = deepcopy(C_re_semi_pi)
            # we need to sum up the coefficients with qualified permutations to get a fully permutation invariant basis
            # These three lines gives all qualified permutations
            for tt = 1:minimum([N1-N_init+1, N_final-N1]) # numbers of variables to be swapped
                for iset in pick(N_init:N1,tt)  # pick a set of variables in the first set to be swapped
                    for jset in pick(N1+1:N_final,tt) # pick a set of variables in the second set to be swapped
                        MM_new = [ swap(mm,iset,jset) for mm in MM ]
                        ord = [ MM_dict[MM_new[i]] for i = 1:length(MM_new) ] # sortperm(MM_new, by = x -> findfirst(==(x), MM))
                        C_new += C_re_semi_pi[:,ord] # swap and add
                    end
                end
            end
            
            C_tmp = [ C_new[i,j][sum(MM[j])+L+1] for i = 1:size(C_new,1), j = 1:size(C_new,2) ]
            U, S, V = svd(C_tmp)
            rk = findall(x -> x > 1e-12, S) |> length # rank(Diagonal(S); rtol =  1e-12) # Somehow rank is not working properly here - also this line is faster than sum(S.>1e-12)
            return Diagonal(S[1:rk].^(-1)) * (U[:, 1:rk]' * C_new), MM
        elseif symmetrization_method == :kernel
            println("Two groups intersect - symmetrization by finding the left kernel of C - C_{x1-y1} is to be performed")
            println()

            # since all the element in C_re_semi_pi are vectors having one nonzero && the position of nonzeros aligns with MM, we can extract the scalar part only
            C_new = [ C_re_semi_pi[i,j][sum(MM[j])+L+1] for i = 1:size(C_re_semi_pi,1), j = 1:size(C_re_semi_pi,2) ]
            MM_new = [ swap(mm,N1,N1+1) for mm in MM ]
            ord = [ MM_dict[MM_new[i]] for i = 1:length(MM_new) ]
            C_new -= C_new[:,ord] # swap and subtract

            # left_ker = nullspace(C_new_scalar', atol = 1e-8)' # not as efficient as an svd
            U, S, V = svd(C_new)
            left_ker = U[:,S .< 1e-12]'

            return left_ker * C_re_semi_pi, MM
        end
    end
 end
