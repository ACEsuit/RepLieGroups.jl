module O3

import RepLieGroups

using PartialWaveFunctions
using Combinatorics
using LinearAlgebra
using StaticArrays
using SparseArrays

export coupling_coeffs


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

function lexi_ord(nn::SVector{N, Int64}, ll::SVector{N, Int64}) where N
    pairs = collect(zip(ll, nn))         # create (lᵢ, nᵢ) pairs
    sort!(pairs)                         # sort lexicographically: first by lᵢ, then by nᵢ
    return SVector{N}(last.(pairs)), SVector{N}(first.(pairs))
end

"""
    O3.coupling_coeffs(L, ll, nn; PI, basis)
    O3.coupling_coeffs(L, ll; PI, basis)

Compute coupling coefficients for the spherical harmonics basis, where 
- `L` must be an `Integer`;
- `ll, nn` must be vectors or tuples of `Integer` of the same length.
- `PI`: whether or not the coupled basis is permutation-invariant (or the 
corresponding tensor symmetric); default is `true` when `nn` is provided 
and `false` when `nn` is not provided.
- `basis`: which basis is being coupled, default is `complex`, alternative
choice is `real`, which is compatible with the `SpheriCart.jl` convention.  
"""
function coupling_coeffs(L::Integer, ll, nn = nothing; 
                         PI = !(isnothing(nn)), 
                         basis = complex)

    # convert L into the format required internally 
    _L = Int(L) 

    # convert ll into an SVector{N, Int}, as required internally 
    N = length(ll) 
    _ll = try 
        _ll = SVector{N, Int}(ll...)
    catch 
        error("""coupling_coeffs(L::Integer, ll, ...) requires ll to be 
               a vector or tuple of integers""")
    end

    # convert nn into an SVector{N, Int}, as required internally 
    if isnothing(nn) 
        if PI 
            _nn = SVector{N, Int}(ntuple(i -> 0, N)...)
        else 
            _nn = SVector{N, Int}((1:N)...)
        end
    elseif length(nn) != N 
        error("""coupling_coeffs(L::Integer, ll, nn) requires ll and nn to be 
               of the same length""")
    else
        _nn = try 
            _nn = SVector{N, Int}(nn...)
        catch 
            error("""coupling_coeffs(L::Integer, ll, nn) requires nn to be 
                   a vector or tuple of integers""")
        end
    end 

    if basis == complex 
        flag = :cSH 
    elseif basis == real 
        flag = :SpheriCart
    elseif basis isa Symbol
        flag = basis 
    else 
        error("unknown basis type: $basis")
    end
    
    return _coupling_coeffs(_L, _ll, _nn; PI = PI, flag = flag)
end
    

# Function that generates the coupling coefficient of the RE basis (PI = false) 
# or RPE basis (PI = true) given `nn` and `ll`. 
function _coupling_coeffs(L::Int64, ll::SVector{N, Int64}, nn::SVector{N, Int64}; 
                          PI = true, flag = :cSH) where N

    # TODO: when PI, (nn, ll) should be ordered 
    if PI
        nn, ll = lexi_ord(nn, ll)
    end

    Lset = SetLl(ll,L)
    r = length(Lset)
    T = L == 0 ? Float64 : SVector{2L+1,Float64}
    if r == 0 
        return zeros(T, 0, 1), [zeros(Int,N)] # no valid coupling, return empty array
    else 
        MMmat, size_m = m_generate(nn,ll,L;flag=flag) # classes of m's
        FMatrix=zeros(T, r, length(MMmat)) # Matrix containing f(m,i)
        UMatrix=zeros(T, r, size_m) # Matrix containing the the coupling coefficients D
        MM = [] # all possible m's
        for i in 1:r
            c = 0
            for (j,m_class) in enumerate(MMmat)
                for mm in m_class
                    c += 1
                    cg_coef = GCG(ll,mm,Lset[i];vectorize=(L!=0),flag=flag)
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

    if !PI
        # return RE coupling coeffs if the permutation invariance is not needed
        return UMatrix, MM
    else
        U, S, V = svd(gram(FMatrix))
        rk = rank(Diagonal(S); rtol =  1e-12)
        # return the RE-PI coupling coeffs
        return Diagonal(S[1:rk]) * (U[:, 1:rk]' * UMatrix), MM
    end
end

end