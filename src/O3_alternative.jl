# Alternative to the computation of rotation equivariant coupling coefficients

using PartialWaveFunctions
using Combinatorics
using LinearAlgebra

export re_basis_new

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

function SetLl0(l,N)
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
        if a[N-1]==l[N]
            push!(set, [a; 0])
        end
    end
    return set
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

function re_basis_new(l)
    N=size(l,1)
    L=SetLl0(l,N)
    r=size(L,1)
    if r==0 
        return zeros(Float64, 0, 0)
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