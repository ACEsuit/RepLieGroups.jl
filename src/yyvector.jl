using StaticArrays 

struct SYYVector{N, T, L}  <: StaticVector{N, T}
   data::NTuple{N, T}
end

function SYYVector{L}(data::NTuple{N, T}) where {L, T, N}
   @assert N == (L+1)^2
   return SYYVector{N, T, L}(data)
end

# TODO: @boundscheck / @propagate_inbounds
@inline Base.getindex(y::SYYVector, i::Integer) = y.data[i] 
@inline Base.getindex(y::SYYVector, i::Int) = y.data[i] 


@inline _lm2i(l, m) = m + l + (l*l) + 1


@inline Base.getindex(y::SYYVector, l::Integer, m::Integer) = y[_lm2i(l, m)]
@inline Base.getindex(y::SYYVector, l::Int, m::Int) = y[_lm2i(l, m)]

@inline Base.getindex(y::SYYVector, ::Val{l}) where l = 
      SVector(ntuple(i -> y[i+l^2], 2*l+1))

Base.Tuple(y::SYYVector) = y.data 
