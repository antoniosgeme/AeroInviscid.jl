function vortex_panel_solver(airfoil::Airfoil,aoa)
    x_vort = airfoil.coordinates[1:end,1]
    y_vort = airfoil.coordinates[1:end,2]
    x_col = ( x_vort[1:end-1] + x_vort[2:end]) / 2
    y_col = ( y_vort[1:end-1] + y_vort[2:end]) / 2
    num_vorts = length(x_vort)
    num_cols = length(x_col)

    θ = atan.( diff(y_vort) , diff(x_vort) )
    n̂ = [sin.(θ) -cos.(θ)] 
    uᵧ,vᵧ = induced_velocity_vortex_sheet(x_col,y_col,x_vort,y_vort,ones(size(x_vort)))

    A = zeros(num_cols+1,num_vorts)
    A[1:end-1,:] = uᵧ .* n̂[:,1] + vᵧ .* n̂[:,2]
    A[num_cols+1,1] = 1
    A[num_cols+1,num_vorts] = 1 

    RHS = Vector{typeof(aoa)}(undef,num_vorts)
    RHS[1:end-1] = -( cosd(aoa) .* n̂[:,1] +  sind(aoa)  .* n̂[:,2])

    if trailing_edge_thickness(airfoil) < 1e-5
        A[end-1,:] .= 0
        A[end-1,1] = 1
        A[end-1,2] = -2
        A[end-1,3] = 1
        A[end-1,end-2] = -1
        A[end-1,end-1] = 2
        A[end-1,end] = -1
        RHS[end-1] = 0
    end 

    γ = A\RHS
    return γ
end 




"""
    induced_velocity_vortex_sheet(x,y,xᵧ,yᵧ,γ)

Computes the induced velocity at points (x,y) due to a vortex sheet defined by (xᵧ,yᵧ)
and strength γ. The sheet is discretized by linear vorticity panels. The ends of the sheet 
are connected by a constant stength source/vorticty panel. This is done to make sure the flow 
leaves each end tangentially (to satisfy the kutta condition) 
""" 
function induced_velocity_vortex_sheet(x,y,xᵧ,yᵧ,γ)
    num_vorts = length(xᵧ)
    num_cols = length(x)
    θ = atan.( diff(yᵧ) , diff(xᵧ) );
    d = hypot.( diff(xᵧ) , diff(yᵧ) )
    t_hat = [cos.(θ) sin.(θ)]
    θ_te = atan( yᵧ[1] - yᵧ[end] , xᵧ[1] - xᵧ[end] )
    d_te = hypot( xᵧ[1] - xᵧ[end] , yᵧ[1] - yᵧ[end] )
    t_te = [cos(θ_te) sin(θ_te)]
    b_te = (t_hat[1,:] - t_hat[end,:]) / norm( t_hat[1,:]-t_hat[end,:])
    dot_prod = abs(t_te[1]*b_te[1] + t_te[2]*b_te[2])
    cross_prod = abs(t_te[1]*b_te[2] - t_te[2]*b_te[1])
    uₐ = zeros(num_cols,num_vorts)
    vₐ = zeros(num_cols,num_vorts)
    uᵦ = zeros(num_cols,num_vorts)
    vᵦ = zeros(num_cols,num_vorts)
    u = zeros(num_cols,num_vorts)
    v = zeros(num_cols,num_vorts)
    for i = 1:num_cols
        for j = 1:num_vorts-1
            (xp, yp) = panel_coordinates(x[i], y[i], xᵧ[j], yᵧ[j], θ[j])
            (upₐ, upᵦ, vpₐ, vpᵦ) = induced_velocity_linear_vortex.(γ[j], γ[j+1], xp, yp, d[j])
            (uₐ[i,j], vₐ[i,j]) = global_coordinates.(upₐ,vpₐ,0,0,θ[j])
            (uᵦ[i,j], vᵦ[i,j]) = global_coordinates.(upᵦ,vpᵦ,0,0,θ[j])
        end 
        (xp, yp) = panel_coordinates(x[i], y[i], xᵧ[end], yᵧ[end], θ_te);
        (upₐ, upᵦ, vpₐ, vpᵦ) = induced_velocity_te_panel(-γ[end], γ[1], xp, yp, d_te,cross_prod,dot_prod);
        (uₐ[i,num_vorts], vₐ[i,num_vorts]) = global_coordinates.(upₐ,vpₐ,0,0,θ_te)
        (uᵦ[i,num_vorts], vᵦ[i,num_vorts]) = global_coordinates.(upᵦ,vpᵦ,0,0,θ_te)
    end

    u[:,2:end-1] = uᵦ[:,1:end-2] + uₐ[:,2:end-1]
    v[:,2:end-1] = vᵦ[:,1:end-2] + vₐ[:,2:end-1]
    u[:,1] = uₐ[:,1] + uᵦ[:,end]
    v[:,1] = vₐ[:,1] + vᵦ[:,end]
    u[:,end] = uᵦ[:,end-1] + uₐ[:,end]
    v[:,end] = vᵦ[:,end-1] + vₐ[:,end]
    return u,v
end 




"""
    induced_velocity_linear_vortex(γₐ, γᵦ, x, y, d)

Computes the induced velocity at point (x,y) due to a vorticity panel 
that has endpoints (0,0) and (d,0) and strength which varies linearly 
from γₐ at (0,0) to γᵦ at (d,0). 
"""
function induced_velocity_linear_vortex(γₐ, γᵦ, x, y, d)
    rₐ = hypot(x, y) 
    rᵦ = hypot(x-d, y)
    θₐ = atan(y, x)
    θᵦ = atan(y, x - d)
    lnr = log(rᵦ / rₐ)
    dθ = θᵦ - θₐ
    uᵦ = γᵦ / (2π*d) * ( y  * lnr + x * dθ  )
    uₐ = γₐ / (2π*d) * ( -y  * lnr + (d - x) * dθ ) 
    vᵦ = γᵦ / (2π*d) * ( x * lnr  + d - dθ * y)
    vₐ = γₐ / (2π*d) * ( (d - x) * lnr - d + dθ * y)
    return uₐ, uᵦ, vₐ, vᵦ
end

"""
    induced_velocity_te_panel(γₐ, γᵦ, x, y, d)

This function computes the induced velocity at point (x,y) due to a 
constant strength source panel as well as a constant strength vorticity
panel, which has endpoints (0,0) and (d,0). The str
"""
function induced_velocity_te_panel(γₐ,γᵦ,x,y,d,cross_prod,dot_prod)
    rₐ = hypot(x, y) 
    rᵦ = hypot(x-d, y)
    θₐ = atan(y, x)
    θᵦ = atan(y, x - d)
    lnr = log(rₐ / rᵦ)
    dθ = θᵦ - θₐ
    uₐ = γₐ/(4π) * lnr * cross_prod + γₐ/(4π) * dθ * dot_prod
    vₐ = γₐ/(4π) * dθ * cross_prod - γₐ/(2π) * lnr * dot_prod
    uᵦ = γᵦ/(4π) * lnr * cross_prod - γᵦ/(4π) * dθ * dot_prod
    vᵦ = γᵦ/(4π) * dθ * cross_prod + γᵦ/(2π) * lnr * dot_prod
    return uₐ,uᵦ,vₐ,vᵦ
end