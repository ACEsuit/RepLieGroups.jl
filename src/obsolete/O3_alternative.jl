# Alternative to the computation of rotation equivariant coupling coefficients

using PartialWaveFunctions
using Combinatorics
using LinearAlgebra

export re_basis_new, ri_basis_new, ind_corr_s1, ind_corr_s2, MatFmi, ML0, MatFmi2, ri_rpi, re_rpe, rpe_basis_new

# The generalized Clebsch Gordan Coefficients; variables of this function are fully inherited from the first ACE paper
function GCG(l::SVector{N,Int64},m::SVector{N,Int64},L::SVector{N,Int64},M_N::Int64;) where N
    # @assert -L[N] ≤ M_N ≤ L[N] 
    if M_N ≠ sum(m) || L[1] < abs(m[1])
        return 0.
    end
    
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
end

# Only when M_N = sum(m) can the CG coefficient be non-zero, so when missing M_N, we return either 
# (1)the full CG coefficient given l, m and L, as a rank 1 vector; 
# (2)or the only one element that can possibly be non-zero on the above vector.
# I suspect that the first option will not be used anyhow, but I keep it for now.
GCG(l::SVector{N,Int64},m::SVector{N,Int64},L::SVector{N,Int64};vectorize::Bool=false) where N = vectorize ? GCG(l,m,L,sum(m)) * Float64.(I(2L[N]+1)[sum(m)+L[N]+1,:]) : GCG(l,m,L,sum(m))

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

# Function that generates the set of ordered m's given `n` and `l` with sum of m's equaling to k.
function m_generate(n::T,l::T,L,k) where T
    @assert abs(k) ≤ L
    S = Sn(n,l)
    Nperm = length(S)-1
    ordered_mset = [submset(l[S[i]], S[i+1]-S[i]) for i = 1:Nperm]
    MM = []
    Total_length = 0
    for m_ord in Iterators.product(ordered_mset...)
        m_ord_reshape = vcat(m_ord...)
        if sum(m_ord_reshape) == k
            class_m = vcat(Iterators.product([multiset_permutations(m_ord[i], S[i+1]-S[i]) for i in 1:Nperm]...)...)
            push!(MM, [vcat(mm...) for mm in class_m])
            Total_length += length(class_m)
        end
    end
    return [ T.(MM[i]) for i = 1:length(MM) ], Total_length
end

# Function that generates the set of ordered m's given `n` and `l` with the abosolute sum of m's being smaller than L.
m_generate(n,l,L=0) = union([m_generate(n,l,L,k)[1] for k in -L:L]...), sum(m_generate(n,l,L,k)[2] for k in -L:L)


# Function that generates the coupling coefficient of the RE basis given `n` and `l`., the FMatrix for generating the RPE basis is also generated here.
function re_rpe(n::SVector{N,Int64},l::SVector{N,Int64},L::Int64) where N
    Lset = SetLl(l,L)
    r = length(Lset)
    T = L == 0 ? Float64 : SVector{2L+1,Float64}
    if r == 0 
        return zeros(T, 1, 1), zeros(T, 1, 1), [zeros(Int,N)], [zeros(Int,N)]
    else 
        MMmat, size_m = m_generate(n,l,L) # classes of m's
        FMatrix=zeros(T, r, length(MMmat)) # Matrix containing f(m,i)
        UMatrix=zeros(T, r, size_m) # Matrix containing the the coupling coefficients D
        MM = [] # all possible m's
        for i in 1:r
            c = 0
            for (j,m_class) in enumerate(MMmat)
                for m in m_class
                    c += 1
                    cg_coef = GCG(l,m,Lset[i];vectorize=(L!=0))
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

function rpe_basis_new(nn::SVector{N, Int64}, ll::SVector{N, Int64}, L::Int64) where N
    t_re = @elapsed UMatrix, FMatrix, MMmat, MM = re_rpe(nn, ll, L) # time of constructing the re_basis
    # @show t_re # should be removed in the final version
    U, S, V = svd(gram(FMatrix))
    rk = rank(Diagonal(S); rtol =  1e-12)
    return Diagonal(S[1:rk]) * (U[:, 1:rk]' * UMatrix), MM
 end