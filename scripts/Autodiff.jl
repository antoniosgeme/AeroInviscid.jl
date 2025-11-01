using ForwardDiff
using AeroInviscid
using Plots
using LinearAlgebra


airfoil = Airfoil("NACA0012")
function compute_cl(α)
    ip = InviscidProblem(airfoil, α)
    sol = solve(ip)
    return sol.cl
end

dcl_dα = ForwardDiff.derivative(compute_cl, 5.0)


# Do the same for multielement airfoil
airfoils = [Airfoil("NACA$i") for i in ["6412","8412","4412"]]
chord = [1.0, 0.5, 0.3]
pitch = [0.0, 30.0, 50.0]
le_loc = [[0.0,0.0],[0.9,-0.15],[1.2,-0.45]]    
me = MultielementAirfoil(airfoils, pitch, chord, le_loc)

function compute_cl_multielement(α)
    ip = InviscidProblem(me, α)
    sol = solve(ip)
    return sol.cl
end

dcl_dα_multielement = ForwardDiff.derivative(compute_cl_multielement, 1.0)


function compute_cl_flap(flap_deflection)
    airfoil = Airfoil("NACA0012")
    airfoil_deflected = deflect_control_surface(airfoil, deflection=flap_deflection)
    ip = InviscidProblem(airfoil_deflected, 0.0)
    sol = solve(ip)
    return sol.cl
end

plot(compute_cl_flap,0:0.1:10)

airfoil = Airfoil("NACA0012")
airfoil_deflected = deflect_control_surface(airfoil, deflection=45)
plot(airfoil_deflected)

plot(compute_cl_flap,0:0.1:10)
dcl_dflap = ForwardDiff.derivative(compute_cl_flap, 20)



function compute_cp_from(δn)
    airfoil = Airfoil("NACA0012")
    n = normals(airfoil; centers=false)
    x_new = airfoil.x .+ δn .* n[:,1]
    y_new = airfoil.y .+ δn .* n[:,2]
    airfoil = Airfoil(x_new, y_new)
    ip = InviscidProblem(airfoil, 4.0)  # Now consistent with Cp_target
    sol = solve(ip)
    return sol.cp
end

function objective(δn)
    Cp = compute_cp_from(δn)
    ΔCp = Cp - Cp_target
    return sum(ΔCp .^ 2)
end

airfoil = Airfoil("NACA0012")
sol = solve(InviscidProblem(airfoil, 4.0))
Cp_target = sol.cp

# a guasisan bump on the upper surface
Cp_target[1:leading_edge_index(airfoil)] = Cp_target[1:leading_edge_index(airfoil)] .- 0.1 .* exp.(-((airfoil.x[1:leading_edge_index(airfoil)] .- 0.5) ./ 0.1).^2)

plot(airfoil.x, Cp_target, label="Target Cp", xlabel="x", ylabel="Cp", yflip=true)

function compute_cp_from_xy(xy)
    airfoil_design = Airfoil(xy[1:N], xy[N+1:end])
    ip = InviscidProblem(airfoil_design, 4.0)
    sol = solve(ip)
    return sol.cp
end

N = length(airfoil.x)
ForwardDiff.jacobian(residuals, vcat(airfoil.x, airfoil.y))

function residuals(xy)
    cp = compute_cp_from_xy(xy)
    return cp - Cp_target
end


# Continue the newton solve here
xy0 = vcat(airfoil.x, airfoil.y)

max_iter = 100
xy = copy(xy0)
for iter in 1:max_iter
    r = residuals(xy)
    J = ForwardDiff.jacobian(residuals, xy)
    
    δ = J \ r
    println("Trying step norm: ", norm(δ))
    xy -= δ*0.05
    println("Iter $iter: ||r|| = ", norm(r))
    
    if norm(r) < 1e-6
        println("Converged!")
        break
    end
end

# ========== BAREBONES NEWTON'S METHOD ==========

# Residual function (what we want to make zero)
function residuals(δn_small)
    # Embed in full space
    δn_full = zeros(eltype(δn_small), length(airfoil.x))
    δn_full[1:length(δn_small)] = δn_small
    Cp = compute_cp_from(δn_full)
    return Cp - Cp_target
end

# Basic Newton's method with damping
function newton_method(x0, max_iter=10)
    x = copy(x0)
    
    for i in 1:max_iter
        # Compute residuals and Jacobian
        r = residuals(x)
        J = ForwardDiff.jacobian(residuals, x)
        
        # Newton step with damping
    
        step = J \ r
        
        # Try damping factors to prevent divergence
        current_residual = norm(r)
        best_x = x
        best_residual = current_residual
        
        for α in [1.0, 0.5, 0.1, 0.01]
            x_trial = x - α * step
            r_trial = residuals(x_trial)
            residual_trial = norm(r_trial)
            
            if residual_trial < best_residual
                best_x = x_trial
                best_residual = residual_trial
            end
        end
        
        x = best_x
        println("Iter $i: ||r|| = $best_residual")
        
        # Check convergence
        if best_residual < 1e-6
            println("Converged!")
            break
        end
    end
    
    return x
end

# Run Newton's method with 5 variables
using AeroInviscid
using ForwardDiff
println("\n=== Barebones Newton's Method ===")
airfoil = Airfoil("NACA0012")
sol = solve(InviscidProblem(airfoil, 4.0))
Cp_target = sol.cp

# a guasisan bump on the upper surface
Cp_target[1:leading_edge_index(airfoil)] = Cp_target[1:leading_edge_index(airfoil)] .- 0.1 .* exp.(-((airfoil.x[1:leading_edge_index(airfoil)] .- 0.5) ./ 0.1).^2)
x0 = zeros(length(airfoil.x))  # Start with 5 variables
x_opt = newton_method(x0)

# Embed result in full space and visualize
δn_full = zeros(length(airfoil.x))
δn_full[1:5] = x_opt

println("\nFinal result:")
println("Max perturbation: ", maximum(abs.(δn_full)))
println("Final objective: ", objective(δn_full))

# Plot result
airfoil_orig = Airfoil("NACA0012")
n = normals(airfoil_orig; centers=false)
x_new = airfoil_orig.x .+ δn_full .* n[:,1]
y_new = airfoil_orig.y .+ δn_full .* n[:,2]
airfoil_opt = Airfoil(x_new, y_new)

p1 = plot(airfoil_orig.x, airfoil_orig.y, label="Original", aspect_ratio=:equal)
plot!(p1, airfoil_opt.x, airfoil_opt.y, label="Newton Optimized", linewidth=2)

Cp_opt = compute_cp_from(δn_full)
p2 = plot(airfoil_orig.x, Cp_target, label="Target", yflip=true, linewidth=2)
plot!(p2, airfoil_orig.x, Cp_opt, label="Newton Result", linewidth=2)

plot(p1, p2, layout=(2,1), size=(800,600))
