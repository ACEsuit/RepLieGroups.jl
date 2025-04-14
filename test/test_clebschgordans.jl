import PartialWaveFunctions

##

# our original reference implementation of CG coeffs, which we no longer 
# use, but we keep this test to make sure our assumptions are consistent 
# with the PartialWaveFunctions implementation. 

module _CGold

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

end

##

@info("Testing the correctness of PartialWaveFunctions with our assumptions")

Lmax = 6
TOL = 1e-14
nerr = 0

for j1 in 1:Lmax, m1 in -j1:j1
    for j2 in 1:Lmax, m2 in -j2:j2
        for J in 1:Lmax, M in -J:J 
            C1 = _CGold.clebschgordan(j1, m1, j2, m2, J, M)
            C2 = PartialWaveFunctions.clebschgordan(j1, m1, j2, m2, J, M)
            if abs(C1 - C2) > TOL 
                @error("""Clebsch-Gordan error: 
                                j1 = $j1, m1 = $m1, 
                                j2 = $j2, m2 = $m2, 
                                J = $J, M = $M""")
                nerr += 1
            end
        end
    end
end

if nerr > 0 
    @error("There are $nerr errors in the Clebsch-Gordan coefficients test")
else
    @info("All tested Clebsch-Gordan coefficients are correct")
end

@test nerr == 0