using StaticArrays 

struct SYYVector{L, N, T}  <: StaticVector{N, T}
   data::NTuple{N, T}
end

SYYVector(data::NTuple{N, T}) where {N, T} = 
         try SYYVector{Int(sqrt(N)-1), N, T}(data) 
         catch 
            error("length of the input should be L^2 for some Int L!") 
         end
         # Int(sqrt(N)) == sqrt(N) ? SYYVector{Int(sqrt(N)-1), N, T}(data) : error("length of the input should be L^2 for some Int L")

# TODO: @boundscheck / @propagate_inbounds
@inline Base.getindex(y::SYYVector, i::Int) = y.data[i] 
# function Base.getindex(y::SYYVector, i::Int)
#    @show i ## i = 4, 8, 14, 22 ?? What happened when display(StaticVector !?)
#    return y.data[i]
# end

@inline _lm2i(l, m) = l^2 + m + l + 1
@inline _i2lm(i) = ( ceil(Int,sqrt(i)) - 1, i - ceil(Int,sqrt(i))^2 + ceil(Int,sqrt(i)) - 1)::Tuple{Int,Int}

@inline Base.getindex(y::SYYVector, l::Int, m::Int) = y[_lm2i(l, m)]

@inline Base.getindex(y::SYYVector, ::Val{l}) where l = 
      SVector(ntuple(i -> y[i+l^2], 2*l+1))

Base.Tuple(y::SYYVector) = y.data 
