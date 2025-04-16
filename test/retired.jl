# NOTE: In the following, we test the LONG equivariant basis (SYYVector), which we don't have in the new version, 
# and it might not be needed anymore. I leave the test code here for now in case we want that back at some point. 

# @info("Equivariance of coupled cSH based LONG basis")  
# for L = 0:2
#    cgen = Rot3DCoeffs_long(L)
#    maxl = [0, 7, 5, 3, 2]
#    for ν = 2:5
#       @info("Testing equivariance of coupled cSH based LONG basis: L = $L, ν = $ν")
#       for ntest = 1:(200 ÷ ν)
#          local θ
#          ll = rand(0:maxl[ν], ν)
#          if L == 0 
#             if !iseven(sum(ll)+L); continue; end 
#          end
#          ll = SVector(ll...)      
#          Ure, Mll = re_basis(cgen, ll)
#          if size(Ure, 1) == 0; continue; end

#          X = [ (@SVector rand(3)) for i in 1:length(ll) ]
#          θ = rand(3) * 2pi
#          Q = RotZYZ(θ...)

#          B1 = eval_basis(ll, Ure, Mll, X; Real = false)
#          B2 = eval_basis(ll, Ure, Mll, Ref(Q) .* X; Real = false)
#          D = BlockDiagonal([ transpose(WignerD.wignerD(l, θ...)) for l = 0:L] )
#          print_tf(@test norm(B1 - Ref(D) .* B2)<1e-12)
#       end
#       println()
#    end
# end

# @info("Testing equivariance of each 'subblock' of the cSH based LONG basis")  
# Lmax = 4
# cgen = Rot3DCoeffs_long(Lmax)
# maxl = [0, 7, 5, 3, 2]
# for ntest = 1:30
#    local ν, ll, Ure, Mll, X, θ, Q, B1, B2
#    ν = rand(2:5)
#    ll = rand(0:maxl[ν], ν)
#    ll = SVector(ll...)      
#    Ure, Mll = re_basis(cgen, ll)
#    if size(Ure, 1) == 0; continue; end
   
#    X = [ (@SVector rand(3)) for i in 1:length(ll) ]
#    θ = rand(3) * 2pi
#    Q = RotZYZ(θ...)
   
#    B1 = eval_basis(ll, Ure, Mll, X; Real = false)
#    B2 = eval_basis(ll, Ure, Mll, Ref(Q) .* X; Real = false)
   
#    for l = 0:Lmax
#       B1l = [ B1[i][Val(l)] for i = 1:length(B1) ]
#       B2l = [ B2[i][Val(l)] for i = 1:length(B2) ]
#       D = transpose(WignerD.wignerD(l, θ...))
#       print_tf(@test norm(B1l - Ref(D) .* B2l)<1e-12)
#    end
# end

# println()