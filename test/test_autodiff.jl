using Test
using ForwardDiff
using AeroInviscid

@testset "Automatic Differentiation Tests" begin
    
    @testset "Single Airfoil - Angle of Attack Derivatives" begin
        airfoil = Airfoil("NACA0012")
        
        # Define function: α -> cl
        function compute_cl(α)
            prob = InviscidProblem(airfoil, α)
            sol = solve(prob)
            return sol.cl
        end
        
        # Test at α = 5°
        α_test = 5.0
        cl = compute_cl(α_test)
        dcl_dα = ForwardDiff.derivative(compute_cl, α_test)
        
        @test cl ≈ 0.6039436653006978
        @test dcl_dα ≈ 0.12048195767599526
        
        println("  α = $(α_test)°: Cl = $(round(cl, digits=4)), dCl/dα = $(round(dcl_dα, digits=6))")
    end
    
    @testset "Single Airfoil - Integer Angle Support" begin
        airfoil = Airfoil("NACA2412")
        
        # Test with integer angle (should auto-promote)
        prob_int = InviscidProblem(airfoil, 0)
        sol_int = solve(prob_int)
        @test sol_int.cl ≈ 0.6025014201854092
        
    end
    
    @testset "Multielement Airfoil - Angle of Attack Derivatives" begin
        # Create 3-element configuration
        airfoils = [Airfoil("NACA6412"), Airfoil("NACA8412"), Airfoil("NACA4412")]
        chord = [1.0, 0.5, 0.3]
        pitch = [0.0, 30.0, 50.0]
        le_loc = [[0.0, 0.0], [0.9, -0.15], [1.2, -0.45]]
        me = MultielementAirfoil(airfoils, pitch, chord, le_loc)
        
        function compute_cl_multi(α)
            prob = InviscidProblem(me, α)
            sol = solve(prob)
            return sol.cl
        end
        
        α_test = 1.0
        cl = compute_cl_multi(α_test)
        dcl_dα = ForwardDiff.derivative(compute_cl_multi, α_test)

        @test cl ≈ 10.348956265820222
        @test dcl_dα ≈ 0.11606448069013778
        
        println("  Multielement at α = $(α_test)°: Cl = $(round(cl, digits=4)), dCl/dα = $(round(dcl_dα, digits=6))")
    end
    
    @testset "Multielement Airfoil - Element Deflection Derivatives" begin
        # Create 2-element configuration (main + flap)
        main = Airfoil("NACA0012")
        flap = Airfoil("NACA0012")
        
        function compute_cl_flap_deflection(flap_angle)
            me = MultielementAirfoil(
                [main, flap],
                [0.0, flap_angle],  # Flap deflection
                [0.75, 0.25],       # 75% main, 25% flap
                [[0.0, 0.0], [0.8, 0.0]]
            )
            prob = InviscidProblem(me, 0.0)
            sol = solve(prob)
            return sol.cl
        end
        
        δ_test = 10.0
        cl = compute_cl_flap_deflection(δ_test)
        dcl_dδ = ForwardDiff.derivative(compute_cl_flap_deflection, δ_test)

        @test cl ≈ 1.3830313173727955
        @test dcl_dδ ≈ 0.13728630552521887
        
        println("  Flap at δ = $(δ_test)°: Cl = $(round(cl, digits=4)), dCl/dδ = $(round(dcl_dδ, digits=6))")
    end
    
    @testset "Control Surface Deflection - Geometry Derivatives" begin
        airfoil = Airfoil("NACA0012")
        
        function compute_cl_deflected(deflection)
            airfoil_deflected = deflect_control_surface(airfoil, deflection=deflection, x_hinge=0.75)
            prob = InviscidProblem(airfoil_deflected, 0.0)
            sol = solve(prob)
            return sol.cl
        end
        
        δ_test = 5.0
        cl = compute_cl_deflected(δ_test)
        dcl_dδ = ForwardDiff.derivative(compute_cl_deflected, δ_test)

        @test cl ≈ 0.37252080326841713
        @test dcl_dδ ≈ .07439769026924799

        println("  Control surface at δ = $(δ_test)°: Cl = $(round(cl, digits=4)), dCl/dδ = $(round(dcl_dδ, digits=6))")
    end
    
    @testset "Type Promotion - Mixed Types" begin
        airfoil = Airfoil("NACA0012")
        
        # Test that Int and Float64 both work
        prob1 = InviscidProblem(airfoil, 5)      # Int
        prob2 = InviscidProblem(airfoil, 5.0)    # Float64
        
        sol1 = solve(prob1)
        sol2 = solve(prob2)
        
        @test sol1.cl ≈ sol2.cl rtol=1e-10
        
        # Test that Dual numbers work (from ForwardDiff)
        f(α) = begin
            prob = InviscidProblem(airfoil, α)
            sol = solve(prob)
            return sol.cl
        end
        
        dual_result = f(ForwardDiff.Dual(5.0, 1.0))
        @test dual_result isa ForwardDiff.Dual
    end
    
    @testset "Gradient Computation - Multiple Variables" begin
        # Test gradient w.r.t. multiple parameters (α and flap deflection)
        main = Airfoil("NACA0012")
        flap = Airfoil("NACA0012")
        
        function compute_cl_multi_param(x)
            α, δ = x[1], x[2]
            me = MultielementAirfoil(
                [main, flap],
                [0.0, δ],
                [0.75, 0.25],
                [[0.0, 0.0], [0.75, 0.0]]
            )
            prob = InviscidProblem(me, α)
            sol = solve(prob)
            return sol.cl
        end
        
        x_test = [5.0, 10.0]  # α = 5°, δ = 10°
        cl = compute_cl_multi_param(x_test)
        grad = ForwardDiff.gradient(compute_cl_multi_param, x_test)
        
        @test length(grad) == 2
        @test grad[1] ≈ 0.159567299234932
        @test grad[2] ≈ 0.13390070566251278
        
        println("  Multi-param gradient: ∇Cl = [$(round(grad[1], digits=6)), $(round(grad[2], digits=6))]")
    end
    
    @testset "Second Derivatives (Hessian)" begin
        airfoil = Airfoil("NACA0012")
        
        function compute_cl(α)
            prob = InviscidProblem(airfoil, α)
            sol = solve(prob)
            return sol.cl
        end
        
        # First derivative
        dcl_dα = α -> ForwardDiff.derivative(compute_cl, α)
        
        # Second derivative
        α_test = 5.0
        d2cl_dα2 = ForwardDiff.derivative(dcl_dα, α_test)
        
        @test abs(d2cl_dα2) < 0.01  # Second derivative should be small (nearly linear)
        
        println("  Second derivative at α = $(α_test)°: d²Cl/dα² = $(round(d2cl_dα2, digits=8))")
    end
    
    @testset "Numerical vs AD Derivatives" begin
        # Compare ForwardDiff with finite differences
        using FiniteDiff
        
        airfoil = Airfoil("NACA2412")
        
        function compute_cl(α)
            prob = InviscidProblem(airfoil, α)
            sol = solve(prob)
            return sol.cl
        end
        
        α_test = 3.0
        
        # AD derivative
        dcl_dα_ad = ForwardDiff.derivative(compute_cl, α_test)
        
        # Finite difference derivative
        dcl_dα_fd = FiniteDiff.finite_difference_derivative(compute_cl, α_test)
        
        @test dcl_dα_ad ≈ dcl_dα_fd rtol=1e-6
        
        println("  AD vs FD: $(round(dcl_dα_ad, digits=8)) vs $(round(dcl_dα_fd, digits=8))")
    end
    
end
