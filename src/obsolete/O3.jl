module O3

using StaticArrays, SparseArrays
using LinearAlgebra: norm, rank, svd, Diagonal, tr

export ClebschGordan, Rot3DCoeffs, Rot3DCoeffs_new


# ------------------- recursion details for different Ls 

# Val{0} stands for L = 0 so invariants, let's focus on that first. 

coco_type(::Val{0}, T::Type{<: Number}) = T 

coco_init(::Val{0}, T) = [ complex(T(1));; ]

coco_init(::Val{0}, T, l, m, μ) = (
                  l == m == μ == 0 ? complex(T(1)) : complex(T(0))  )

coco_zeros(::Val{0}, T, ll, mm, kk) = complex(T(0))

## NOTE: for rSH, we can enforce sum(mm) == 0 but for cSH this is not the case. 
#        We need a new filter, which is for now a little bit rough. The exact 
#        condition now shall be "there exist {t_i}_i", such that \sum (-1)^(t_i)*mm_i = 0

coco_filter(::Val{0}, ll, mm) = iseven(sum(ll)) && (sum(mm) == 0)

coco_filter(::Val{0}, ll, mm, kk) = iseven(sum(ll)) && (sum(mm) == sum(kk) == 0)

coco_filter_new(::Val{0}, ll, mm) = iseven(sum(ll))#  && (sum(mm) == 0)

coco_filter_new(::Val{0}, ll, mm, kk) = iseven(sum(ll))#  && (sum(mm) == sum(kk) == 0)

coco_dot(u1::Number, u2::Number) = u1' * u2

# -------------------


"""
`ClebschGordan: ` storing precomputed Clebsch-Gordan coefficients; see
`?clebschgordan` for the convention that is use.
"""
struct ClebschGordan{T}
	vals::Dict{Tuple{Int, Int, Int, Int, Int, Int}, T}
end

struct Rot3DCoeffs{L, T}
   vals::Vector{Dict}      # val[N] = coeffs for correlation order N
   cg::ClebschGordan{T}
end

struct Rot3DCoeffs_new{L, T}
   vals::Vector{Dict}      # val[N] = coeffs for correlation order N
   cg::ClebschGordan{T}
end

_ValL(::Union{Rot3DCoeffs{L},Rot3DCoeffs_new{L}}) where {L} = Val{L}() 

# -----------------------------------
# iterating over an m collection
# -----------------------------------

_mvec(::CartesianIndex{0}) = SVector{0, Int}()

_mvec(mpre::CartesianIndex) = SVector(Tuple(mpre)...)

struct MRange{L, N, T2}
   ll::SVector{N, Int}
   cartrg::T2
end

_getL(::MRange{L}) where {L} = L 
_ValL(::MRange{L}) where {L} = Val{L}()

## TODO: for rSH and cSH, the filter shall not be the same...
Base.length(mr::MRange) =
		sum(mt -> coco_filter_new(_ValL(mr), mr.ll, _mvec(mt)), mr.cartrg)

"""
Given an l-vector `ll` iterate over all combinations of `mm` vectors  of
the same length such that `sum(mm) == 0`
"""
_mrange(L::Integer, ll) = _mrange(Val{L}(), ll)

function _mrange(::Val{L}, ll) where {L}
   N = length(ll) 
   cartrg = CartesianIndices(ntuple(i -> -ll[i]:ll[i], length(ll)))
	return MRange{L, N, typeof(cartrg)}(ll, cartrg)
end

# TODO: should we impose here that (ll, mm) are lexicographically ordered?

function Base.iterate(mr::MRange, idx::Integer=0)
	while true
		idx += 1
		if idx > length(mr.cartrg)
			return nothing
		end
		mm = _mvec(mr.cartrg[idx])
		if coco_filter(_ValL(mr), mr.ll, mm)
			return mm, idx
		end
	end
	error("we should never be here")
end




# ----------------------------------------------------------------------
#     ClebschGordan code
# ----------------------------------------------------------------------


cg_conditions(j1,m1, j2,m2, J,M) =
	cg_l_condition(j1, j2, J)   &&
	cg_m_condition(m1, m2, M)   &&
	(abs(m1) <= j1) && (abs(m2) <= j2) && (abs(M) <= J)

cg_l_condition(j1, j2, J) = (abs(j1-j2) <= J <= j1 + j2)

cg_m_condition(m1, m2, M) = (M == m1 + m2)


"""
`clebschgordan(j1, m1, j2, m2, J, M, T=Float64)` :
A reference implementation of Clebsch-Gordon coefficients based on
https://hal.inria.fr/hal-01851097/document
Equation (4-6)
This heavily uses BigInt and BigFloat and should therefore not be employed
for performance critical tasks, but only precomputation.
The ordering of parameters corresponds to the following convention:
```
clebschgordan(j1, m1, j2, m2, J, M) = C_{j1m1j2m2}^{JM}
```
where
```
   D_{m1k1}^{l1} D_{m2k2}^{l2}}
	=
	∑_j  C_{l1m1l2m2}^{j(m1+m2)} C_{l1k1l2k2}^{j2(k1+k2)} D_{(m1+m2)(k1+k2)}^{j}
```
"""
function clebschgordan(j1, m1, j2, m2, J, M, T=Float64)
	if !cg_conditions(j1, m1, j2, m2, J, M)
		return zero(T)
	end

   N = (2*J+1) *
       factorial(big(j1+m1)) * factorial(big(j1-m1)) *
       factorial(big(j2+m2)) * factorial(big(j2-m2)) *
       factorial(big(J+M)) * factorial(big(J-M)) /
       factorial(big( j1+j2-J)) /
       factorial(big( j1-j2+J)) /
       factorial(big(-j1+j2+J)) /
       factorial(big(j1+j2+J+1))

   G = big(0)
   # 0 ≦ k ≦ j1+j2-J
   # 0 ≤ j1-m1-k ≤ j1-j2+J   <=>   j2-J-m1 ≤ k ≤ j1-m1
   # 0 ≤ j2+m2-k ≤ -j1+j2+J  <=>   j1-J+m2 ≤ k ≤ j2+m2
   lb = (0, j2-J-m1, j1-J+m2)
   ub = (j1+j2-J, j1-m1, j2+m2)
   for k in maximum(lb):minimum(ub)
      bk = big(k)
      G += (-1)^k *
           binomial(big( j1+j2-J), big(k)) *
           binomial(big( j1-j2+J), big(j1-m1-k)) *
           binomial(big(-j1+j2+J), big(j2+m2-k))
   end

   return T(sqrt(N) * G)
end

function Ctran(l::Int64,m::Int64,μ::Int64)
   if abs(m) ≠ abs(μ)
      return 0
   elseif abs(m) == 0
      return 1
   elseif m < 0 && μ < 0
      return - im * (-1)^m/sqrt(2) # 1/sqrt(2)
   elseif m < 0 && μ > 0
      return im/sqrt(2) # (-1)^m/sqrt(2)
   elseif m > 0 && μ < 0
      return (-1)^m/sqrt(2) # - im * (-1)^m/sqrt(2)
   else
      return 1/sqrt(2) # im/sqrt(2)
   end
end

Ctran(l::Int64) = sparse(Matrix{ComplexF64}([ Ctran(l,m,μ) for m = -l:l, μ = -l:l ])) |> dropzeros

## NOTE: Ctran(L) is the transformation matrix from rSH to cSH. More specifically, 
#        if we write Polynomials4ML rSH as R_{lm} and cSH as Y_{lm} and their corresponding 
#        vectors of order L as R_L and Y_L, respectively. Then R_L = Ctran(L) * Y_L.
#        This suggests that the "D-matrix" for the Polynomials4ML rSH is Ctran(l) * D(l) * Ctran(L)', 
#        where D, the D-matrix for cSH. This inspires the following new CG recursion.
#

ClebschGordan(T=Float64) =
	ClebschGordan{T}(Dict{Tuple{Int,Int,Int,Int,Int,Int}, T}())

_cg_key(j1, m1, j2, m2, J, M) = (j1, m1, j2, m2, J, M)

function (cg::ClebschGordan{T})(j1, m1, j2, m2, J, M) where {T}
	if !cg_conditions(j1,m1, j2,m2, J,M)
		return zero(T)
	end
	key = _cg_key(j1, m1, j2, m2, J, M)
	if haskey(cg.vals, key)
		return cg.vals[key]
	end
	val = clebschgordan(j1, m1, j2, m2, J, M, T)
	cg.vals[key] = val
	return val
end

# ----------------------------------------------------------------------
#     Rot3DCoeffs code: generalized cg coefficients
#
#  Note: in this section kk is a tuple of m-values, it is not
#        related to the k index in the 1-p basis (or radial basis)
# ----------------------------------------------------------------------

dicttype(N::Integer, TP) = dicttype(Val(N), TP)

dicttype(::Val{N}, TP) where {N} =
   Dict{Tuple{SVector{N,Int}, SVector{N,Int}, SVector{N,Int}}, TP}

Rot3DCoeffs(L, T=Float64) = Rot3DCoeffs{L, T}(Dict[], ClebschGordan(T))

Rot3DCoeffs_new(L, T=ComplexF64) = Rot3DCoeffs_new{L, T}(Dict[], ClebschGordan(T))

function get_vals(A::Union{Rot3DCoeffs{L, T},Rot3DCoeffs_new{L, T}}, valN::Val{N}) where {L, T,N}
	# make up an ll, kk, mm and compute a dummy coupling coeff
	ll, mm, kk = SVector(0), SVector(0), SVector(0)
	cc0 = coco_zeros(_ValL(A), T, ll, mm, kk)
	TP = typeof(cc0)
	if length(A.vals) < N
		# create more dictionaries of the correct type
		for n = length(A.vals)+1:N
			push!(A.vals, dicttype(n, TP)())
		end
	end
   return (A.vals[N])::(dicttype(valN, TP))
end

_key(ll::StaticVector{N}, mm::StaticVector{N}, kk::StaticVector{N}) where {N} =
      (SVector{N, Int}(ll), SVector{N, Int}(mm), SVector{N, Int}(kk))

function (A::Union{Rot3DCoeffs{L, T},Rot3DCoeffs_new{L, T}})(ll::StaticVector{N},
                             mm::StaticVector{N},
                             kk::StaticVector{N}) where {L, T, N}
   vals = get_vals(A, Val(N))  # this should infer the type!
   key = _key(ll, mm, kk)
   if haskey(vals, key)
      val  = vals[key]
   else
      val = _compute_val(A, key...)
      vals[key] = val
   end
   return val
end

# the recursion has two steps so we need to define the
# coupling coefficients for N = 1, 2
# TODO: actually this seems false; it is only one recursion step, and a bit
#       or reshuffling should allow us to get rid of the {N = 2} case.

(A::Union{Rot3DCoeffs{L, T},Rot3DCoeffs_new{L, T}})(ll::StaticVector{1},
                 mm::StaticVector{1},
                 kk::StaticVector{1}) where {L, T} =
		coco_init(_ValL(A), T, ll[1], mm[1], kk[1])


function _compute_val(A::Rot3DCoeffs{L, T}, ll::StaticVector{N},
                                         mm::StaticVector{N},
                                         kk::StaticVector{N}) where {L, T, N}
	val = coco_zeros(_ValL(A), T, ll, mm, kk)
	TV = typeof(val)

	tmp = zero(MVector{N-1, Int})

	function _get_pp(aa, ap)
		for i = 1:N-2
			@inbounds tmp[i] = aa[i]
		end
		tmp[N-1] = ap
		return SVector(tmp)
	end

	jmin = maximum( ( abs(ll[N-1]-ll[N]),
				         abs(kk[N-1]+kk[N]),
						   abs(mm[N-1]+mm[N]) ) )
   jmax = ll[N-1]+ll[N]
   for j = jmin:jmax
		cgk = A.cg(ll[N-1], kk[N-1], ll[N], kk[N], j, kk[N-1]+kk[N])
		cgm = A.cg(ll[N-1], mm[N-1], ll[N], mm[N], j, mm[N-1]+mm[N])
		if cgk * cgm  != 0
			llpp = _get_pp(ll, j) # SVector(llp..., j)
			mmpp = _get_pp(mm, mm[N-1]+mm[N]) # SVector(mmp..., mm[N-1]+mm[N])
			kkpp = _get_pp(kk, kk[N-1]+kk[N]) # SVector(kkp..., kk[N-1]+kk[N])
			a = A(llpp, mmpp, kkpp)::TV
			val += cgk * cgm * a
		end
   end
   return val
end

function _compute_val(A::Rot3DCoeffs_new{L, T}, ll::StaticVector{N},
                                         mm::StaticVector{N},
                                         kk::StaticVector{N}) where {L, T, N}
	val = coco_zeros(_ValL(A), T, ll, mm, kk)
	TV = typeof(val)

	tmp = zero(MVector{N-1, Int})

	function _get_pp(aa, ap)
		for i = 1:N-2
			@inbounds tmp[i] = aa[i]
		end
		tmp[N-1] = ap
		return SVector(tmp)
	end
	
	set(m) = unique([m,-m])
	## TODO: the following two functions can be combined, but not for now
	function ntset(m1,k1,m2,k2)
		temp = Iterators.product(set(m1),set(k1),set(m2),set(k2)) |> collect
		return [ temp[i] for i = 1:length(temp) ]
	end
	
	function pqset(n,t)
		temp = Iterators.product(set(n),set(t)) |> collect
		return [ temp[i] for i = 1:length(temp) ]
	end
	
	function const1(m1,k1,m2,k2,n1,t1,n2,t2)
		lmax = maximum([m1,k1,m2,k2,n1,t1,n2,t2])
		return Ctran(lmax,m1,n1) * Ctran(lmax,k1,t1)' * Ctran(lmax,m2,n2) * Ctran(lmax,k2,t2)'
	end
	
	function const2(n,t,p,q)
		lmax = maximum([n,p,t,q])
		return Ctran(lmax,p,n)' * Ctran(lmax,q,t)
	end
	
	jmin = abs(ll[N-1]-ll[N])
	# jmin = maximum( ( abs(ll[N-1]-ll[N]),
	#			         abs(kk[N-1]+kk[N]),
	#					   abs(mm[N-1]+mm[N]) ) )
   	jmax = ll[N-1]+ll[N]
   	for j = jmin:jmax
	   	for (n1,t1,n2,t2) in ntset(mm[N-1],kk[N-1],mm[N],kk[N])
			cgk = A.cg(ll[N-1], t1, ll[N], t2, j, t1+t2)
			cgm = A.cg(ll[N-1], n1, ll[N], n2, j, n1+n2)
			c1 = cgk * cgm * const1(mm[N-1],kk[N-1],mm[N],kk[N],n1,t1,n2,t2)
			if c1 != 0
				for (p,q) in pqset(n1+n2,t1+t2)
					llpp = _get_pp(ll, j) # SVector(llp..., j)
					mmpp = _get_pp(mm, p) # SVector(mmp..., mm[N-1]+mm[N])
					kkpp = _get_pp(kk, q) # SVector(kkp..., kk[N-1]+kk[N])
					a = A(llpp, mmpp, kkpp)::TV
					c2 = const2(n1+n2,t1+t2,p,q)
					if c2 != 0 && abs(p) ≤ j && abs(q) ≤ j
						val += c1 * c2 * a
					end
				end
			end
   		end
	end
   	return val
end

# ----------------------------------------------------------------------
#   construction of a possible set of generalised CG coefficient;
#   numerically via SVD; this could be done analytically which might
#   be more efficient.
# ----------------------------------------------------------------------


function re_basis(A::Union{Rot3DCoeffs{L, T},Rot3DCoeffs_new{L, T}}, ll::SVector) where {L, T}
	TCC = coco_type(_ValL(A), T)
	CC, Mll = compute_Al(A, ll)  # CC::Vector{Vector{...}}
	G = [ sum( coco_dot(CC[a][i], CC[b][i]) for i = 1:length(Mll) )
			for a = 1:length(CC), b = 1:length(CC) ]
	svdC = svd(G)
	rk = rank(Diagonal(svdC.S), rtol = 1e-7)
	# Diagonal(sqrt.(svdC.S[1:rk])) * svdC.U[:, 1:rk]' * CC
	# construct the new basis
	Ured = Diagonal(sqrt.(svdC.S[1:rk])) * svdC.U[:, 1:rk]'
	Ure = Matrix{TCC}(undef, rk, length(Mll))
	for i = 1:rk
		Ure[i, :] = sum(Ured[i, j] * CC[j]  for j = 1:length(CC))
	end
	return Ure, Mll
end


# function barrier
function compute_Al(A::Union{Rot3DCoeffs{L, T},Rot3DCoeffs_new{L, T}}, ll::SVector) where {L, T}
	Mll = collect(_mrange(_ValL(A), ll))
   TP = coco_type(_ValL(A), T)
	if length(Mll) == 0
		return Vector{TP}[], Mll
	end

	TA = typeof(A(ll, Mll[1], Mll[1]))
	return __compute_Al(A, ll, Mll, TP, TA)
end

# TODO: what was TA for? Can we get rid of it via coco_type? 

function __compute_Al(A::Union{Rot3DCoeffs{L, T},Rot3DCoeffs_new{L, T}}, ll, Mll, TP, TA) where {L, T}
	lenMll = length(Mll)
	# each element of CC will be one row of the coupling coefficients
	TCC = coco_type(_ValL(A), T)
	CC = Vector{TCC}[]
	# some utility funcions to allow coco_init to return either a property
	# or a vector of properties
	function __into_cc!(cc, cc0, im)   # cc0: ::AbstractProperty
		@assert length(cc) == 1
		cc[1][im] = cc0
	end
	function __into_cc!(cc, cc0::AbstractVector, im)
		@assert length(cc) == length(cc0)
		for p = 1:length(cc)
			cc[p][im] = cc0[p]
		end
	end

	for (ik, kk) in enumerate(Mll)  # loop over possible basis functions
		# do a dummy calculation to determine how many coefficients we will get
		cc0 = A(ll, Mll[1], kk)::TA
      @assert cc0 isa Number 
      numcc = 1 
      # the assert above replaced the following line; to be replaced with 
      #     the suitable generalisation to L > 0 
		# numcc = (cc0 isa AbstractProperty ? 1 : length(cc0))
		# allocate the right number of vectors to store basis function coeffs
		cc = [ Vector{TCC}(undef, lenMll) for _=1:numcc ]
		for (im, mm) in enumerate(Mll) # loop over possible indices
			if T == Float64
				fil = coco_filter
			elseif T == ComplexF64
				fil = coco_filter_new
			end
			if !fil(_ValL(A), ll, mm, kk)
				cc00 = zeros(TP, length(cc))::TA
				__into_cc!(cc, cc00, im)
			else
				# get all possible coupling coefficients
				cc0 = A(ll, mm, kk)::TA
				__into_cc!(cc, cc0, im)
			end
		end
		# and now push them onto the big stack.
		append!(CC, cc)
	end

	return CC, Mll
end


end
