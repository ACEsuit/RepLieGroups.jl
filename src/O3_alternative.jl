# Alternative to the computation of rotation equivariant coupling coefficients

using PartialWaveFunctions
using Combinatorics
using LinearAlgebra

export re_basis_new, ri_basis_new, ind_corr_s1, ind_corr_s2, MatFmi, ML0, MatFmi2, ri_rpi, re_rpe, rpe_basis_new

function CG(l,m,L,N) 
    M=m[1]+m[2]
    if L[2]<abs(M)
        return 0.
    else
        C=PartialWaveFunctions.clebschgordan(l[1],m[1],l[2],m[2],L[2],M) 
    end
    for k in 3:N
        if L[k]<abs(M+m[k])
            return 0.
        elseif L[k-1]<abs(M)
            return 0.
        else
            C*=PartialWaveFunctions.clebschgordan(L[k-1],M,l[k],m[k],L[k],M+m[k])
            M+=m[k]
        end
    end
    return C
end

# The variables of this function are fully inherited from the first ACE paper
function CG_new(l::SVector{N,Int64},m::SVector{N,Int64},L::SVector{N,Int64},M_N::Int64;) where N
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
CG_new(l::SVector{N,Int64},m::SVector{N,Int64},L::SVector{N,Int64};vectorize::Bool=false) where N = vectorize ? CG_new(l,m,L,sum(m)) * Float64.(I(2L[N]+1)[sum(m)+L[N]+1,:]) : CG_new(l,m,L,sum(m))

function SetLl0(l,N)
    set = Vector{Int64}[]
    if N==2
        if l[1]==l[2]
            return [[0;0]]
        else 
            return Vector{Int64}[]
        end
    else 
        for k in abs(l[1]-l[2]):l[1]+l[2]
            push!(set, [0; k])
        end
        for k in 3:N-1
            setL=set
            set=Vector{Int64}[]
            for a in setL
                for b in abs(a[k-1]-l[k]):a[k-1]+l[k]
                    push!(set, [a; b])
                end
            end
        end  
        setL=set
        set=Vector{Int64}[]
        for a in setL
            if a[N-1]==l[N]
                push!(set, [a; 0])
            end
        end
        return set
    end
end

function SetLl(l,N,L)
    set = Vector{Int64}[]
    for k in abs(l[1]-l[2]):l[1]+l[2]
        push!(set, [0; k])
    end
    for k in 3:N-1
        setL=set
        set=Vector{Int64}[]
        for a in setL
            for b in abs(a[k-1]-l[k]):a[k-1]+l[k]
                push!(set, [a; b])
            end
        end
    end
    setL=set
    set=Vector{Int64}[]
    for a in setL
        if (abs.(a[N-1]-l[N]) <= L)&&(L <= (a[N-1]+l[N]))
            push!(set, [a; L])
        end
    end
    return set
end

# Function that returns a L set given an `l`. The elements of the set start with l[1] and end with L. 
function SetLl_new(l::SVector{N,Int64}, L::Int64) where N
    T = typeof(l)
    if N==2        
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

# Function that computes the set ML0
function ML0(l,N)
    setML = [[i] for i in -abs(l[1]):abs(l[1])]
    for k in 2:N-1
        set = setML
        setML = Vector{Int64}[]
        for m in set
            append!(setML, [m; lk] for lk in -abs(l[k]):abs(l[k]) )
        end
    end
    setML0=Vector{Int64}[]
    for m in setML
        s=sum(m)
        if  abs(s) < abs(l[N])+1
            push!(setML0, [m; -s])
        end
    end
    return setML0
end

# Function that computes the set ML (relative to equivariance L)
function ML(l,N,L)
    setML = [[i] for i in -abs(l[1]):abs(l[1])]
    for k in 2:N-1
        set = setML
        setML = Vector{Int64}[]
        for m in set
            append!(setML, [m; lk] for lk in -abs(l[k]):abs(l[k]) )
        end
    end
    setML0=Vector{Int64}[]
    for m in setML
        s=sum(m)
        for mn in -L-s:L-s
            if  abs(mn) < abs(l[N])+1
                push!(setML0, [m; mn])
            end
        end
    end
    return setML0
end

# Function that returns a set of all possible m's given an `l` with sum of m's equaling to k.
function Mlk(l::SVector{N,Int64}, k::Int64) where N
    T = typeof(l)
    setMlk = [[i] for i in -abs(l[1]):abs(l[1])]
    for i in 2:N
        set = setMlk
        setMlk = Vector{Int64}[]
        if i < N
            for m in set
                append!(setMlk, [m; li] for li in -abs(l[i]):abs(l[i]) )
            end
        else
            for m in set
                s = sum(m)
                if abs(k-s) ≤ abs(l[N])
                    push!(setMlk, [m; k-s])
                end
            end
        end
    end

    return T.(setMlk)
end

# Function that returns a set of all possible m's given an `l` with the absolute value of sum of m's being smaller than L
MlL(l::SVector{N,Int64}, L::Int64) where N = sort(union([Mlk(l, k) for k in -L:L]...))

function ri_basis_new(l)
    N=size(l,1)
    L=SetLl0(l,N)
    r=size(L,1)
    if r==0 
        return zeros(Float64, 1, 1), [zeros(Int64,N)]
    else 
        setML0=ML0(l,N)
        sizeML0=length(setML0)
        U=zeros(Float64, r, sizeML0)
        M = Vector{Int64}[]
        for (j,m) in enumerate(setML0)
            push!(M,m)
            for i in 1:r
                U[i,j]=CG(l,m,L[i],N)
            end
        end
    end
    return U,M
end

function re_basis_new(l,L)
    N=size(l,1)
    Ll=SetLl(l,N,L)
    r=size(Ll,1)
    if r==0 
        return zeros(Float64, 0, 0)
    else 
        setML0=ML(l,N,L)
        sizeML0=length(setML0)
        U=zeros(Float64, r, sizeML0)
        M = Vector{Int64}[]
        for (j,m) in enumerate(setML0)
            push!(M,m)
            for i in 1:r
                U[i,j]=CG(l,m,Ll[i],N)
            end
        end
    end
    return U,M
end


# Function that computes the permutations that let n and l invariant
function Snl(N,n,l)
    if n==n[1]*ones(N)
        if l==l[1]*ones(N)
            return permutations(1:N)
        end
    end
    if N==1
        return Set([[1]])
    elseif (n[N-1],l[N-1])!=(n[N],l[N])
        S=Set()
        Sn=Snl(N-1,n[1:N-1],l[1:N-1])
        for x in Sn
            append!(x,[N])
            union!(S,Set([x]))
        end
    else
        S=Set()
        k=N
        while (n[k-1],l[k-1])==(n[k],l[k]) && k>2
            k-=1
        end
        if k==2 && (n[1],l[1])==(n[2],l[2])
            return Set(permutations(1:N))
        else
            Sn=Snl(k-1,n[1:k-1],l[1:k-1])
            for x in Sn
                for s in Set(permutations(k:N))
                    y=copy(x)
                    append!(y,s)
                    union!(S,Set([y]))
                end
            end
        end
    end
    return S
end


#Function that computes the set of classes using the set Ml0 and the possible permutations
function class(setML0,sigma,N,l)
    setclass=Vector{Vector{Int64}}[]
    pop!(setML0,zeros(Int64,N))
    while setML0!=Set()
        x=pop!(setML0)
        p=[x]
        for s in sigma
            y=x[s]
            if y in setML0
                append!(p,[y])
                pop!(setML0,y)
            end
        end
        append!(setclass,[p])
    end
    setclasses=Vector{Vector{Int64}}[]
    for x in setclass
        for y in setclass
            if x==y
                if minimum(x)==minimum(-x)
                    if iseven(sum(l))
                        append!(setclasses,[x])
                    end
                end
            elseif minimum(x)==minimum(-y)
                if y<x
                    append!(setclasses,[x])
                end
            end
        end
    end
    if iseven(sum(l))
        append!(setclasses,[[zeros(N)]])
    end
    setclasses
end



# Function that computes the matrix ( f(m,i) )
function MatFmi(n,l)
    N=size(l,1)
    L=SetLl0(l,N)
    r=size(L,1)
    if r==0 
        return zeros(Float64, 1, 1), [zeros(Int,N)]
    else 
        ML00 = ML0(l,N)
        setML0=Set(ML00)
        sigma = Snl(N,n,l)
        setclass=class(setML0,sigma,N,l)
        sizeML0=length(setclass)
        Matrix=zeros(Float64, r, sizeML0)
        for i in 1:r
            for j in 1:sizeML0
                for m in setclass[j]
                    Matrix[i,j]+=CG(l,m,L[i],N)
                end
            end
        end
    end
    return Matrix, ML00
end


# Function that computes the matrix ( f(m,i) )
function MatFmi2(n,l)
    N=size(l,1)
    L=SetLl0(l,N)
    r=size(L,1)
    if r==0 
        return zeros(Float64, 1, 1), [zeros(Int,N)]
    else 
        ML00 = ML0(l,N)
        setML0=Set(ML00)
        sigma = Snl(N,n,l)
        setclass=class(setML0,sigma,N,l)
        sizeML0=length(setclass)
        sizeMMs=length(ML00)
        FMatrix=zeros(Float64, r, sizeML0)
        UMatrix=zeros(Float64, r, sizeMMs)
        MM = []
        for i in 1:r
            c = 1
            for j in 1:sizeML0
                for m in setclass[j]
                    cg_coef = CG(l,m,L[i],N)
                    FMatrix[i,j]+= cg_coef
                    UMatrix[i,c] = cg_coef
                    c += 1
                end
            end
        end
        for j in 1:sizeML0
            for m in setclass[j]
                push!(MM,m)
            end
        end      
    end
    return UMatrix, FMatrix, ML00, MM
end


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

function ri_rpi(n,l)
    N=size(l,1)
    L=SetLl0(l,N)
    r=size(L,1)
    if r==0 
        return zeros(Float64, 1, 1), zeros(Float64, 1, 1), [zeros(Int,N)], [zeros(Int,N)] 
    else 
        MMmat, size_m = m_generate(n,l)
        FMatrix=zeros(Float64, r, length(MMmat))
        UMatrix=zeros(Float64, r, size_m)
        MM = []
        for i in 1:r
            c = 0
            for (j,m_class) in enumerate(MMmat)
                for m in m_class
                    c += 1
                    cg_coef = CG(l,m,L[i],N)
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

function re_rpe(n::SVector{N,Int64},l::SVector{N,Int64},L::Int64) where N
    Lset = SetLl_new(l,L)
    r = length(Lset)
    T = L == 0 ? Float64 : SVector{2L+1,Float64}
    if r == 0 
        return zeros(T, 1, 1), zeros(T, 1, 1), [zeros(Int,N)], [zeros(Int,N)]
    else 
        MMmat, size_m = m_generate(n,l,L)
        FMatrix=zeros(T, r, length(MMmat))
        UMatrix=zeros(T, r, size_m)
        MM = []
        for i in 1:r
            c = 0
            for (j,m_class) in enumerate(MMmat)
                for m in m_class
                    c += 1
                    cg_coef = CG_new(l,m,Lset[i];vectorize=(L!=0))
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

function gram(X)
    G = zeros(ComplexF64, size(X,1), size(X,1))
    for i = 1:size(X,1)
       for j = i:size(X,1)
          G[i,j] = sum(dot(X[i,t], X[j,t]') for t = 1:size(X,2))
          i == j ? nothing : (G[j,i]=G[i,j]')
       end
    end
    return G
 end

 function rpe_basis_new(nn::SVector{N, Int64}, ll::SVector{N, Int64}, L::Int64) where N
    t_re = @elapsed UMatrix, FMatrix, MMmat, MM = re_rpe(nn, ll, L)
    @show t_re # should be removed in the final version
    U, S, V = svd(gram(FMatrix))
    rk = rank(Diagonal(S); rtol =  1e-12)
    return Diagonal(S[1:rk]) * (U[:, 1:rk]' * UMatrix), MM
 end