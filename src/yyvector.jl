using StaticArrays 

struct SYYVector{L, N, T}  <: StaticVector{N, T}
   data::NTuple{N, T}
end

SYYVector(data::NTuple{N, T}) where {N, T} = 
         try SYYVector{Int(sqrt(N)-1), N, T}(data) 
         catch 
            error("length of the input should be L^2 for some Int L!") 
         end

# TODO: @boundscheck / @propagate_inbounds
Base.@propagate_inbounds function Base.getindex(y::SYYVector, i::Int)
	@boundscheck checkbounds(y,i)
	return y.data[i] 
end

@inline _lm2i(l, m) = l^2 + m + l + 1
@inline _i2lm(i) = ( ceil(Int,sqrt(i)) - 1, i - ceil(Int,sqrt(i))^2 + ceil(Int,sqrt(i)) - 1)::Tuple{Int,Int}

@inline Base.getindex(y::SYYVector, lm::Tuple{Int,Int}) = y[_lm2i(lm[1], lm[2])]

@inline Base.getindex(y::SYYVector, ::Val{l}) where l = 
      SVector(ntuple(i -> y[i+l^2], 2*l+1))

Base.Tuple(y::SYYVector) = y.data 
