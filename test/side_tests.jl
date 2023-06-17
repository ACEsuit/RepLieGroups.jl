##
function fit_recursion_oneterm(l1,m1,μ1,l2,m2,μ2)
   λset = abs(l1-l2):(l1+l2)
   N_samp = 100
   A = zeros(ComplexF64,N_samp,length(λset))
   B = zeros(ComplexF64,N_samp)

   local theta, dset
   for i = 1:N_samp
      theta = rand() * 2pi
      dset = [ Ctran(L) * wignerD(L, 0, 0, theta) * Ctran(L)' for L = 0:Lmax ]

      D(l,m,μ) = dset[l+1][m+l+1,μ+l+1]
      for (j,L) in enumerate(λset)
         if abs(m1+m2) ≤ L && abs(μ1+μ2) ≤ L
            A[i,j] = D(L,m1+m2,μ1+μ2)
         end
      end
      B[i] = D(l1,m1,μ1)*D(l2,m2,μ2)
   end

   C = A\B
   if norm(C - real(C)) >1e-10
      C = real(C)
   end
   
   for i = 1:N_samp
      theta = rand() * 2pi
      dset = [ Ctran(L) * wignerD(L, 0, 0, theta) * Ctran(L)' for L = 0:Lmax ]

      D(l,m,μ) = dset[l+1][m+l+1,μ+l+1]
      for (j,L) in enumerate(λset)
         if abs(m1+m2) ≤ L && abs(μ1+μ2) ≤ L
            A[i,j] = D(L,m1+m2,μ1+μ2)
         end
      end
      B[i] = D(l1,m1,μ1)*D(l2,m2,μ2)
   end
   
   return norm(A * C - B) < 1e-8
end

function fit_recursion(l1,m1,μ1,l2,m2,μ2)
   λset = abs(l1-l2):(l1+l2)
   N_samp = 100
   A = zeros(ComplexF64,N_samp,8*length(λset))
   B = zeros(ComplexF64,N_samp)
   
   local theta, dset

   for i = 1:N_samp
      theta = rand() * 2pi
      dset = [ Ctran(L) * wignerD(L, 0, 0, theta) * Ctran(L)' for L = 0:Lmax ]

      D(l,m,μ) = dset[l+1][m+l+1,μ+l+1]
      for (j,L) in enumerate(λset)
         if abs(m1+m2) ≤ L && abs(μ1+μ2) ≤ L
            A[i,j] = D(L,m1+m2,μ1+μ2)
         end
         if abs(m1+m2) ≤ L && abs(μ1-μ2) ≤ L
            A[i,j+length(λset)] = D(L,m1+m2,μ1-μ2)
         end
         if abs(m1-m2) ≤ L && abs(μ1+μ2) ≤ L
            A[i,j+2length(λset)] = D(L,m1-m2,μ1+μ2)
         end
         if abs(m1-m2) ≤ L && abs(μ1-μ2) ≤ L
            A[i,j+3length(λset)] = D(L,m1-m2,μ1-μ2)
         end
         if abs(-m1+m2) ≤ L && abs(μ1+μ2) ≤ L
            A[i,j+4length(λset)] = D(L,-m1+m2,μ1+μ2)
         end
         if abs(-m1+m2) ≤ L && abs(μ1-μ2) ≤ L
            A[i,j+5length(λset)] = D(L,-m1+m2,μ1-μ2)
         end
         if abs(-m1-m2) ≤ L && abs(μ1+μ2) ≤ L
            A[i,j+6length(λset)] = D(L,-m1-m2,μ1+μ2)
         end
         if abs(-m1-m2) ≤ L && abs(μ1-μ2) ≤ L
            A[i,j+7length(λset)] = D(L,-m1-m2,μ1-μ2)
         end
         # if abs(m1+m2) ≤ L && abs(-μ1+μ2) ≤ L
         #    A[i,j+8length(λset)] = D(L,m1+m2,-μ1+μ2)
         # end
         # if abs(m1+m2) ≤ L && abs(-μ1-μ2) ≤ L
         #    A[i,j+9length(λset)] = D(L,m1+m2,-μ1-μ2)
         # end
         # if abs(m1-m2) ≤ L && abs(-μ1+μ2) ≤ L
         #    A[i,j+10length(λset)] = D(L,m1-m2,-μ1+μ2)
         # end
         # if abs(m1-m2) ≤ L && abs(-μ1-μ2) ≤ L
         #    A[i,j+11length(λset)] = D(L,m1-m2,-μ1-μ2)
         # end
         # if abs(-m1+m2) ≤ L && abs(-μ1+μ2) ≤ L
         #    A[i,j+12length(λset)] = D(L,-m1+m2,-μ1+μ2)
         # end
         # if abs(-m1+m2) ≤ L && abs(-μ1-μ2) ≤ L
         #    A[i,j+13length(λset)] = D(L,-m1+m2,-μ1-μ2)
         # end
         # if abs(-m1-m2) ≤ L && abs(-μ1+μ2) ≤ L
         #    A[i,j+14length(λset)] = D(L,-m1-m2,-μ1+μ2)
         # end
         # if abs(-m1-m2) ≤ L && abs(-μ1-μ2) ≤ L
         #    A[i,j+5length(λset)] = D(L,-m1-m2,-μ1-μ2)
         # end
      end
      B[i] = D(l1,m1,μ1)*D(l2,m2,μ2)
   end

   C = A\B
   C = real(C)
   
   for i = 1:N_samp
      theta = rand() * 2pi
      dset = [ Ctran(L) * wignerD(L, 0, 0, theta) * Ctran(L)' for L = 0:Lmax ]

      D(l,m,μ) = dset[l+1][m+l+1,μ+l+1]
      for (j,L) in enumerate(λset)
         if abs(m1+m2) ≤ L && abs(μ1+μ2) ≤ L
            A[i,j] = D(L,m1+m2,μ1+μ2)
         end
         if abs(m1+m2) ≤ L && abs(μ1-μ2) ≤ L
            A[i,j+length(λset)] = D(L,m1+m2,μ1-μ2)
         end
         if abs(m1-m2) ≤ L && abs(μ1+μ2) ≤ L
            A[i,j+2length(λset)] = D(L,m1-m2,μ1+μ2)
         end
         if abs(m1-m2) ≤ L && abs(μ1-μ2) ≤ L
            A[i,j+3length(λset)] = D(L,m1-m2,μ1-μ2)
         end
         if abs(-m1+m2) ≤ L && abs(μ1+μ2) ≤ L
            A[i,j+4length(λset)] = D(L,-m1+m2,μ1+μ2)
         end
         if abs(-m1+m2) ≤ L && abs(μ1-μ2) ≤ L
            A[i,j+5length(λset)] = D(L,-m1+m2,μ1-μ2)
         end
         if abs(-m1-m2) ≤ L && abs(μ1+μ2) ≤ L
            A[i,j+6length(λset)] = D(L,-m1-m2,μ1+μ2)
         end
         if abs(-m1-m2) ≤ L && abs(μ1-μ2) ≤ L
            A[i,j+7length(λset)] = D(L,-m1-m2,μ1-μ2)
         end
         # if abs(m1+m2) ≤ L && abs(-μ1+μ2) ≤ L
         #    A[i,j+8length(λset)] = D(L,m1+m2,-μ1+μ2)
         # end
         # if abs(m1+m2) ≤ L && abs(-μ1-μ2) ≤ L
         #    A[i,j+9length(λset)] = D(L,m1+m2,-μ1-μ2)
         # end
         # if abs(m1-m2) ≤ L && abs(-μ1+μ2) ≤ L
         #    A[i,j+10length(λset)] = D(L,m1-m2,-μ1+μ2)
         # end
         # if abs(m1-m2) ≤ L && abs(-μ1-μ2) ≤ L
         #    A[i,j+11length(λset)] = D(L,m1-m2,-μ1-μ2)
         # end
         # if abs(-m1+m2) ≤ L && abs(-μ1+μ2) ≤ L
         #    A[i,j+12length(λset)] = D(L,-m1+m2,-μ1+μ2)
         # end
         # if abs(-m1+m2) ≤ L && abs(-μ1-μ2) ≤ L
         #    A[i,j+13length(λset)] = D(L,-m1+m2,-μ1-μ2)
         # end
         # if abs(-m1-m2) ≤ L && abs(-μ1+μ2) ≤ L
         #    A[i,j+14length(λset)] = D(L,-m1-m2,-μ1+μ2)
         # end
         # if abs(-m1-m2) ≤ L && abs(-μ1-μ2) ≤ L
         #    A[i,j+5length(λset)] = D(L,-m1-m2,-μ1-μ2)
         # end
      end
      B[i] = D(l1,m1,μ1)*D(l2,m2,μ2)
   end
   return norm(A * C - B) < 1e-8
end

for i = 1:100
   l1 = rand(0:Lmax)
   m1 = rand(-l1:l1)
   μ1 = rand(-l1:l1)
   l2 = rand(0:Lmax-l1)
   m2 = rand(-l2:l2)
   μ2 = rand(-l2:l2)
   tf = fit_recursion_oneterm(l1,m1,μ1,l2,m2,μ2)
   if !tf
      println("The elements of the new D matrix have no one-term recursion!")
      break
   end
   # println("The elements of the new D matrix have an one-term recursion!")
end

for i = 1:100
   l1 = rand(0:Lmax)
   m1 = rand(-l1:l1)
   μ1 = rand(-l1:l1)
   l2 = rand(0:Lmax-l1)
   m2 = rand(-l2:l2)
   μ2 = rand(-l2:l2)
   tf = fit_recursion(l1,m1,μ1,l2,m2,μ2)
   if !tf
      println("The elements of the new D matrix have no eight-term recursion!")
      break
   end
   # println("The elements of the new D matrix have an eight-term recursion!")
end

@info("Properties of the new D-matrix")
@info("1. Only when abs(m) == abs(μ) can the corresponding element be nonzero")
for i = 1:50
   theta = rand() * 2pi
   dset = [ Ctran(L) * wignerD(L, 0, 0, theta) * Ctran(L)' for L = 0:Lmax ]
   D(l,m,μ) = dset[l+1][m+l+1,μ+l+1]
   # dset = [ wignerD(L, 0, 0, theta) for L = 0:Lmax ]
   
   l = rand(0:Lmax)
   m = rand(-l:l)
   μ = rand(-l:l)
   print_tf(@test abs(D(l,m,μ))<1e-10 || abs(m) == abs(μ))
end
println()

@info("2. D(l,m,μ) = D(l,-μ,-m)")
for i = 1:50
   theta = rand() * 2pi
   dset = [ Ctran(L) * wignerD(L, 0, 0, theta) * Ctran(L)' for L = 0:Lmax ]
   # dset = [ wignerD(L, 0, 0, theta) for L = 0:Lmax ]
   D(l,m,μ) = dset[l+1][m+l+1,μ+l+1]

   l = rand(0:Lmax)
   m = rand(-l:l)
   μ = rand(-l:l)
   print_tf(@test abs(D(l,m,μ) - D(l,-μ,-m)) ≤ 1e-10)
end
println()

@info("3. D(l,m,μ) = -(-1)^{m==μ} * D(l,-m,-μ) - this is the most important part to simplify the recursion")
for i = 1:50
   theta = rand() * 2pi
   dset = [ Ctran(L) * wignerD(L, 0, 0, theta) * Ctran(L)' for L = 0:Lmax ]
   # dset = [ wignerD(L, 0, 0, theta) for L = 0:Lmax ]
   D(l,m,μ) = dset[l+1][m+l+1,μ+l+1]

   l = rand(0:Lmax)
   m = rand(-l:l)
   μ = rand(-l:l)
   print_tf(@test abs(D(l,m,μ)+(-1)^(m==μ)*D(l,-m,-μ)) ≤ 1e-10)
end
println()
