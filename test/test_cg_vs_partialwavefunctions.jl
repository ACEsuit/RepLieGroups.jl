using PartialWaveFunctions
using RepLieGroups.O3: ClebschGordan

@info("Testing the correctness of PartialWaveFunctions")
Lmax = 6
for j1 in 1:Lmax
    for m1 in -j1:j1
        for j2 in 1:Lmax
            for m2 in -j2:j2
                for J in 1:Lmax
                    for M in -J:J 
                        @test abs(clebschgordan(j1, m1, j2, m2, J, M) - PartialWaveFunctions.clebschgordan(j1, m1, j2, m2, J, M) < 1e-14)
                    end
                end
            end
        end
    end
end
