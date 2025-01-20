module O3

using RepLieGroups: SYYVector
using StaticArrays, SparseArrays
using LinearAlgebra: norm, rank, svd, Diagonal, tr, dot

export ClebschGordan, Rot3DCoeffs, Rot3DCoeffs_real, Rot3DCoeffs_long

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

struct Rot3DCoeffs_real{L, T}
   vals::Vector{Dict}      # val[N] = coeffs for correlation order N
   cg::ClebschGordan{T}
end

_ValL(::Union{Rot3DCoeffs{L},Rot3DCoeffs_real{L}}) where {L} = Val{L}() 

# -------------------  Functions hacked from ACE which will be used to construct CCs for L>0
"""
Index of entries in D matrix (sign included)
"""
struct D_Index
	sign::Int64
	μ::Int64
	m::Int64
end

"""
auxiliary matrix - indices for D matrix
"""
wigner_D_indices(L::Integer) = (   @assert L >= 0;
		[ D_Index(1, i - 1 - L, j - 1 - L) for j = 1:2*L+1, i = 1:2*L+1] )

Base.adjoint(idx::D_Index) = D_Index( (-1)^(idx.μ+idx.m), - idx.μ, - idx.m)

function vec_cou_coe(rotc::Union{Rot3DCoeffs, Rot3DCoeffs_real},
					      l::Integer, m::Integer, μ::Integer,
					      L::Integer, t::Integer)
	@assert 0 < t <= 2L+1
	D = wigner_D_indices(L)'   # Dt = D[:,t]  -->  # D^* ⋅ e^t
	LL = SA[l, L]
	Z = ntuple(i -> begin
			cc = (rotc(LL, SA[μ, D[i, t].m], SA[m, D[i, t].μ]))
			D[i, t].sign * cc
				end, 2*L+1)
	# Z = [ D[i, t].sign * rotc(LL, SA[μ, D[i, t].m], SA[m, D[i, t].μ]) for i = 1:2L+1 ]
	# Z = [ D[i, t].sign * rotc(LL, SA[m, D[i, t].μ], SA[m, D[i, t].μ]) for i = 1:2L+1 ]
	return SVector(Z)
end

function _select_t(L, l, M, K)
	D = wigner_D_indices(L)'
	tret = -1; numt = 0
	for t = 1:2L+1
		prodμt = prod( (D[i, t].μ + M) for i in 1:2L+1)  # avoid more allocations
		prodmt = prod( (D[i, t].m + K) for i in 1:2L+1)
		if prodμt == prodmt == 0
			tret = t; numt += 1
		end
	end
	# We assumed that there is only one coefficient; this will warn us if it fails
	# For Rot3DCoeffs_real, it can have more than one value
	@assert numt == 1
	return tret
end

# ------------------- recursion details for different Ls 

# Val{0} stands for L = 0 so invariants, let's focus on that first. 
# Now it has been extended to general L.
coco_type(::Val{L}, T::Type{<: Number}) where L = L == 0 ? T : SVector{2L+1,T} 

coco_init(::Val{L}, T) where L = L == 0 ? [ complex(T(1));; ] : []

coco_init(::Val{L}, T, l, m, μ) where L = L == 0 ? (
                  l == m == μ == 0 ? complex(T(1)) : complex(T(0))  ) : ( vec_cou_coe(Rot3DCoeffs(0), l, m, μ, L, _select_t(L, l, m, μ)) )

# TODO: This is still wrong - to be modified
coco_init_real(::Val{L}, T, l, m, μ) where L = L == 0 ? (
				  l == m == μ == 0 ? complex(T(1)) : complex(T(0))  ) : ( [ vec_cou_coe(Rot3DCoeffs_real(0), l, m, μ, L, _select_t(L, l, m, μ)[i]) for i = 1: length(_select_t(L, l, m, μ)) ] )

coco_zeros(::Val{L}, T, ll, mm, kk) where L = L == 0 ? complex(T(0)) : @SVector zeros(T,2L+1)

## NOTE: for rSH, we can enforce sum(mm) == 0 but for cSH this is not the case. 
#        We need a new filter, which is "there exist {t_i}_i", such that ∑_i (-1)^(t_i)*mm_i = 0

coco_filter(::Val{L}, ll, mm) where L = iseven(sum(ll)+L) && abs(sum(mm)) <= L

coco_filter(::Val{L}, ll, mm, kk) where L = iseven(sum(ll)+L) && abs(sum(mm)) <= L && abs(sum(kk)) <= L

function mm_filter(mm,L=0)
	# TODO: Extend this to larger L...
	if L ≠ 0
		error("Not implemented yet!")
	end
	set(m) = unique([m,-m])
	mmset = Iterators.product([set(mm[i]) for i = 1:length(mm)]...) |> collect
	for (i,m) in enumerate(mmset)
		if abs(sum(m)) <= L # Was it simply "... <= L" for larger L??
			return true
		end
	end
	return false
end

coco_filter_real(::Val{L}, ll, mm) where L = iseven(sum(ll)+L) && mm_filter(mm,L)

coco_filter_real(::Val{L}, ll, mm, kk) where L = iseven(sum(ll)+L) && mm_filter(mm,L) && mm_filter(kk,L)

coco_dot(u1::Number, u2::Number) = u1' * u2

coco_dot(u1::SVector{L,T}, u2::SVector{L,T}) where {L,T} = dot(u1, u2)

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
Base.length(mr::MRange) = length(mr.cartrg)
Base.sum(cc::CartesianIndex) = sum(cc.I)

"""
Given an l-vector `ll` iterate over all combinations of `mm` vectors  of
the same length such that `sum(mm) == 0` (or such that mm is feasible, for larger L)
"""
_mrange(L::Integer, ll) = _mrange(Val{L}(), ll)

function _mrange(::Val{L}, ll; ll_filter = ll -> iseven(sum(ll)+L), mm_filter = mm -> abs(sum(mm)) <= L) where {L}
   if !ll_filter(ll)
	   return MRange{L, length(ll), Tuple}(ll, ())
   end
   N = length(ll) 
   cartrg = CartesianIndices(ntuple(i -> -ll[i]:ll[i], length(ll)))
   cartrg = cartrg[findall(x -> x==1, mm_filter.(cartrg))]
	return MRange{L, N, typeof(cartrg)}(ll, cartrg)
end

# TODO: should we impose here that (ll, mm) are lexicographically ordered?

function Base.iterate(mr::MRange, idx::Integer=0)
	while true
		idx += 1
		return idx > length(mr.cartrg) ? nothing : (_mvec(mr.cartrg[idx]), idx)
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

# transformation matrix from RSH to CSH for different conventions
function Ctran(L::Int64; convention = :SpheriCart)
	AA = zeros(ComplexF64, 2L+1, 2L+1)
	order_dict = Dict(:SpheriCart => [1,2,3,4], :CondonShortley => [4,3,2,1], :FHIaims => [4,3,1,2])
	for i = -L:L
		val_list = [(-1)^(i), im, (-1)^(i+1)*im, 1] ./ sqrt(2)
		for j in [-i, i]
			AA[i+L+1,j+L+1] = begin
				if i == j == 0
					1
				elseif i > 0 && j > 0
					val_list[order_dict[convention][1]]
				elseif i < 0 && j < 0
					val_list[order_dict[convention][2]]
				elseif i < 0 && j > 0
					val_list[order_dict[convention][3]]
				elseif i > 0 && j < 0
					val_list[order_dict[convention][4]]
				end
			end
		end
	end
	return sparse(AA)
 end

# Ctran(l::Int64) = sparse(Matrix{ComplexF64}([ Ctran(l,m,μ) for m = -l:l, μ = -l:l ])) |> dropzeros
Ctran(l::Int64,m::Int64,μ::Int64; convention = :SpheriCart) = Ctran(l;convention)[l+m+1,l+μ+1]

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

Rot3DCoeffs_real(L, T=ComplexF64) = Rot3DCoeffs_real{L, T}(Dict[], ClebschGordan(T))

function get_vals(A::Union{Rot3DCoeffs{L, T},Rot3DCoeffs_real{L, T}}, valN::Val{N}) where {L, T,N}
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

function (A::Union{Rot3DCoeffs{L, T},Rot3DCoeffs_real{L, T}})(ll::StaticVector{N},
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

(A::Rot3DCoeffs{L, T})(ll::StaticVector{1},
                 mm::StaticVector{1},
                 kk::StaticVector{1}) where {L, T} =
		coco_init(_ValL(A), T, ll[1], mm[1], kk[1])
		
(A::Rot3DCoeffs_real{L, T})(ll::StaticVector{1},
		         mm::StaticVector{1},
		         kk::StaticVector{1}) where {L, T} =
		coco_init_real(_ValL(A), T, ll[1], mm[1], kk[1])


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
			a = TV(A(llpp, mmpp, kkpp))::TV
			val += cgk * cgm * a
		end
   end
   return val
end

function _compute_val(A::Rot3DCoeffs_real{L, T}, ll::StaticVector{N},
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
		lmax = maximum(abs.([m1,k1,m2,k2,n1,t1,n2,t2]))
		return Ctran(lmax,m1,n1) * Ctran(lmax,k1,t1)' * Ctran(lmax,m2,n2) * Ctran(lmax,k2,t2)'
	end
	
	function const2(n,t,p,q)
		lmax = maximum(abs.([n,p,t,q]))
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


function re_basis(A::Union{Rot3DCoeffs{L, T},Rot3DCoeffs_real{L, T}}, ll::SVector) where {L, T}
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
function compute_Al(A::Union{Rot3DCoeffs{L, T},Rot3DCoeffs_real{L, T}}, ll::SVector) where {L, T}
	if typeof(A) <: Rot3DCoeffs
		fil = mm -> abs(sum(mm)) <= L
	elseif typeof(A) <: Rot3DCoeffs_real
		fil = mm_filter
	else 
		error("Not implemented yet")
	end
	Mll = collect(_mrange(_ValL(A), ll; mm_filter = fil))
   TP = coco_type(_ValL(A), T)
	if length(Mll) == 0
		return Vector{TP}[], Mll
	end

	TA = typeof(A(ll, Mll[1], Mll[1]))
	return __compute_Al(A, ll, Mll, TP, TA)
end

# TODO: what was TA for? Can we get rid of it via coco_type? 

function __compute_Al(A::Union{Rot3DCoeffs{L, T},Rot3DCoeffs_real{L, T}}, ll, Mll, TP, TA) where {L, T}	
	if typeof(A) <: Rot3DCoeffs
		fil = coco_filter
	elseif typeof(A) <: Rot3DCoeffs_real
		fil = coco_filter_real
	else 
		error("Not implemented yet")
	end
	
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
	# # NOTE: We won't have this in the current setting???
	# function __into_cc!(cc, cc0::AbstractVector, im)
	# 	@assert length(cc) == length(cc0)
	# 	for p = 1:length(cc)
	# 		cc[p][im] = cc0[p]
	# 	end
	# end

	for (ik, kk) in enumerate(Mll)  # loop over possible basis functions
		# do a dummy calculation to determine how many coefficients we will get
		cc0 = A(ll, Mll[1], kk)::TA
      @assert length(cc0) == 2L+1 
      numcc = 1
      # the assert above replaced the following line; to be replaced with 
      #     the suitable generalisation to L > 0 
		# numcc = (cc0 isa AbstractProperty ? 1 : length(cc0))
		# allocate the right number of vectors to store basis function coeffs
		cc = [ Vector{TCC}(undef, lenMll) for _=1:numcc ]
		for (im, mm) in enumerate(Mll) # loop over possible indices
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


## I will first define Rot3DCoeffs_long at the very end and we then decide how we unify things...

struct Rot3DCoeffs_long{L, T}
   vals::Vector{Dict}      # val[N] = coeffs for correlation order N
   cg::ClebschGordan{T}
end

_ValL(::Rot3DCoeffs_long{L}) where {L} = Val{L}() 

coco_type_long(::Val{L}, T::Type{<: Number}) where L = SYYVector{L, (L+1)^2,T} 

function coco_init_long(::Val{L}, T, l, m, μ) where L
	# TODO: ComplexF64? Float64?
	init = zeros(T,(L+1)^2)
	for ltemp = 0:L
		init[ltemp^2+1:(ltemp+1)^2] = ltemp == 0 ? [coco_init(Val(ltemp),T,l,m,μ)] : 
		try coco_init(Val(ltemp),T,l,m,μ) 
		catch 
			zeros(T,2ltemp+1)
		end
		# NOTE: A very interesting point - we currently do not have an elegant filter for all (sub)L
		#       so we sometimes have asseration error in vec_cou_coeff function. However, this only means that 
		#       the coupling coefficients for those cases should be 0!
	end
	init = NTuple{(L+1)^2,T}(init)
	return SYYVector(init)
end

coco_zeros_long(::Val{L}, T, ll, mm, kk) where L = SYYVector(NTuple{(L+1)^2,T}(zeros((L+1)^2)))

coco_filter_long(::Val{L}, ll, mm) where L = 
		L == 0 ? iseven(sum(ll)) && abs(sum(mm)) <= L : abs(sum(mm)) <= L

coco_filter_long(::Val{L}, ll, mm, kk) where L = 
		L == 0 ? iseven(sum(ll)) && abs(sum(mm)) <= L && abs(sum(kk)) <= L : abs(sum(mm)) <= L && abs(sum(kk)) <= L
		
coco_dot(u1::SYYVector{L,N,T}, u2::SYYVector{L,N,T}) where {L,N,T} = dot(u1.data, u2.data)

Rot3DCoeffs_long(L, T=Float64) = Rot3DCoeffs_long{L, T}(Dict[], ClebschGordan(T))

function get_vals(A::Rot3DCoeffs_long{L, T}, valN::Val{N}) where {L,T,N}
	# make up an ll, kk, mm and compute a dummy coupling coeff
	ll, mm, kk = SVector(0), SVector(0), SVector(0)
	cc0 = coco_zeros_long(_ValL(A), T, ll, mm, kk)
	TP = typeof(cc0)
	if length(A.vals) < N
		# create more dictionaries of the correct type
		for n = length(A.vals)+1:N
			push!(A.vals, dicttype(n, TP)())
		end
	end
   return (A.vals[N])::(dicttype(valN, TP))
end

function (A::Rot3DCoeffs_long{L, T})(ll::StaticVector{N},
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

(A::Rot3DCoeffs_long{L, T})(ll::StaticVector{1},
		         mm::StaticVector{1},
		         kk::StaticVector{1}) where {L, T} =
		coco_init_long(_ValL(A), T, ll[1], mm[1], kk[1])


function _compute_val(A::Rot3DCoeffs_long{L, T}, ll::StaticVector{N},
                                         mm::StaticVector{N},
                                         kk::StaticVector{N}) where {L, T, N}
	val = coco_zeros_long(_ValL(A), T, ll, mm, kk)
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
			a = A(llpp, mmpp, kkpp)# ::TV
			val += cgk * cgm * a
		end
   end
   return val
end

function re_basis(A::Rot3DCoeffs_long{L, T}, ll::SVector) where {L, T}
	TCC = coco_type_long(_ValL(A), T)
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
function compute_Al(A::Rot3DCoeffs_long{L, T}, ll::SVector) where {L, T}
	fil(mm) = abs(sum(mm)) <= L
	Mll = collect(_mrange(_ValL(A), ll; mm_filter = fil))
   TP = coco_type_long(_ValL(A), T)
	if length(Mll) == 0
		return Vector{TP}[], Mll
	end

	TA = typeof(A(ll, Mll[1], Mll[1]))
	return __compute_Al(A, ll, Mll, TP, TA)
end

# TODO: what was TA for? Can we get rid of it via coco_type? 

function __compute_Al(A::Rot3DCoeffs_long{L, T}, ll, Mll, TP, TA) where {L, T}	
	lenMll = length(Mll)
	# each element of CC will be one row of the coupling coefficients
	TCC = coco_type_long(_ValL(A), T)
	CC = Vector{TCC}[]
	# some utility funcions to allow coco_init to return either a property
	# or a vector of properties
	function __into_cc!(cc, cc0, im)   # cc0: ::AbstractProperty
		@assert length(cc) == 1
		cc[1][im] = cc0
	end
	# # NOTE: We won't have this in the current setting???
	# function __into_cc!(cc, cc0::AbstractVector, im)
	# 	@assert length(cc) == length(cc0)
	# 	for p = 1:length(cc)
	# 		cc[p][im] = cc0[p]
	# 	end
	# end

	for (ik, kk) in enumerate(Mll)  # loop over possible basis functions
		# do a dummy calculation to determine how many coefficients we will get
		cc0 = A(ll, Mll[1], kk)# ::TA
      @assert length(cc0) == (L+1)^2 
      numcc = 1
      # the assert above replaced the following line; to be replaced with 
      #     the suitable generalisation to L > 0 
		# numcc = (cc0 isa AbstractProperty ? 1 : length(cc0))
		# allocate the right number of vectors to store basis function coeffs
		cc = [ Vector{TCC}(undef, lenMll) for _=1:numcc ]
		for (im, mm) in enumerate(Mll) # loop over possible indices
			if !coco_filter_long(_ValL(A), ll, mm, kk)
				cc00 = zeros(TP, length(cc))::TA
				__into_cc!(cc, cc00, im)
			else
				# get all possible coupling coefficients
				cc0 = A(ll, mm, kk)# ::TA
				__into_cc!(cc, cc0, im)
			end
		end
		# and now push them onto the big stack.
		append!(CC, cc)
	end

	return CC, Mll
end

## End of the latest long Rot3D implementation

end
