using Statistics

@testset "Joukowsky Airfoil - Analytical vs Numerical" begin
    
    @testset "Lift Coefficient Comparison" begin
        test_angles = [2.0, 4.0, 6.0, 8.0, 10.0]  # degrees
        
        for α_deg in test_angles
            
            airfoil = Airfoil("NACA0002")  
            
            prob = InviscidProblem(airfoil, α_deg)
            sol = solve(prob)
            cl_numerical = sol.cl
            
            α_rad = deg2rad(α_deg)
            cl_analytical = 2π * α_rad
            
            rel_error = abs(cl_numerical - cl_analytical) / abs(cl_analytical)

            @test rel_error < 0.5  
            println("α = $(α_deg)°: Cl_num = $(round(cl_numerical, digits=4)), " *
                    "Cl_theory = $(round(cl_analytical, digits=4)), " *
                    "Error = $(round(rel_error*100, digits=2))%")

        end
    end
    
    @testset "Joukowsky Airfoil - Pressure Distribution" begin
        # Test that pressure distribution satisfies basic physical constraints
        α_deg = 5.0
        airfoil = Airfoil("NACA0012")
        
        prob = InviscidProblem(airfoil, α_deg)
        sol = solve(prob)
        
        @test all(sol.cp .<= 1.0)  
        
   
        
        # Allow some tolerance due to discretization
        @test abs(sol.cp[1] - sol.cp[end]) < 1e-3 
        println("Cp at TE - Upper: $(round(sol.cp[1], digits=3)), " *
               "Lower: $(round(sol.cp[end], digits=3))")
    end
    
    @testset "Joukowsky Airfoil - Circulation and Velocity" begin
        # Test relationship between circulation, velocity, and lift
        α_deg = 8.0
        airfoil = Airfoil("NACA0012")
        
        prob = InviscidProblem(airfoil, α_deg)
        sol = solve(prob)
        
        # Calculate circulation from vortex strength
        # Γ = ∫ γ ds
        dx = diff(airfoil.x)
        dy = diff(airfoil.y)
        ds = hypot.(dx, dy)
        
        # Average vortex strength on each panel
        γ_avg = (sol.strength[1:end-1] .+ sol.strength[2:end]) ./ 2
        circulation = sum(γ_avg .* ds)
        
        # Kutta-Joukowski theorem: L = ρ V∞ Γ
        # For unit density and V∞=1: Cl = 2Γ/c
        # where c is the chord length
        chord = maximum(airfoil.x) - minimum(airfoil.x)
        cl_from_circulation = 2 * circulation / chord
        
        # This should match the computed Cl from the solver
        rel_error = abs(cl_from_circulation - sol.cl) / abs(sol.cl)
        @test rel_error < 0.01  # Should match very closely
        println("Cl from solver: $(round(sol.cl, digits=4)), " *
               "Cl from circulation: $(round(cl_from_circulation, digits=4)), " *
               "Error: $(round(rel_error*100, digits=3))%")
    end
    
    @testset "Zero Angle of Attack - Symmetry" begin
        # At α=0, symmetric airfoil should have symmetric solution
        α_deg = 0.0
        airfoil = Airfoil("NACA0012")
        
        prob = InviscidProblem(airfoil, α_deg)
        sol = solve(prob)
        
        # Lift coefficient should be nearly zero
        @test abs(sol.cl) < 0.001
        
        # Pressure distribution should be symmetric
        n = length(sol.cp)
        mid_idx = leading_edge_index(airfoil)
        
        # Compare upper and lower surface Cp (they should be similar)
        # Note: indexing depends on how airfoil points are ordered
        # Typically goes around from TE -> upper -> LE -> lower -> TE
        cp_upper = sol.cp[1:mid_idx]
        cp_lower = reverse(sol.cp[mid_idx+1:end])
        
        # Trim to same length if needed
        min_len = min(length(cp_upper), length(cp_lower))
        cp_upper = cp_upper[1:min_len]
        cp_lower = cp_lower[1:min_len]
        
        # Check symmetry (allow some numerical tolerance)
        mean_abs_diff = mean(abs.(cp_upper .- cp_lower))
        @test mean_abs_diff < 0.1
        println("Mean absolute Cp difference (upper vs lower): $(round(mean_abs_diff, digits=4))")
    end
    
end
