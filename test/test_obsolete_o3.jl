

using Test, RepLieGroups, StaticArrays, Polynomials4ML
using RepLieGroups.O3: ClebschGordan, Rot3DCoeffs, re_basis, 
            _mrange, MRange
using Polynomials4ML: CYlmBasis, index_y
using Polynomials4ML.Testing: print_tf

##

cg = ClebschGordan()
cgen = Rot3DCoeffs(0)

ll = SVector(1, 2, 1, 2)
Ure, Mll = re_basis(cgen, ll)

