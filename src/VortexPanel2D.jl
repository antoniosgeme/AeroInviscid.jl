function solve(prob::InviscidProblem{A,LinearVortex}) where A<:Airfoil
    airfoil = prob.geometry
    α       = prob.alpha

    γ  = linear_vortex_solver(airfoil, α)       # Vector of vortex strengths
    cp = @. 1 - γ^2                             # @. adds the dots ⇒ 1 .- γ .^ 2

    # crude thin-airfoil estimate: CL = 2∫γ dx
    d = hypot.(diff(airfoil.x),diff(airfoil.y))
    cl = 2 * sum((γ[1:end-1] + γ[2:end])/2 .* d)

    return InviscidSolution(airfoil, α, LinearVortex, γ, cp, cl)
end

function solve(prob::InviscidProblem{MultielementAirfoil,LinearVortex})
    multielement = prob.geometry

    # big concatenated gamma, one block per element
    γ  = linear_vortex_solver(multielement, prob.alpha)
    cp = @. 1 - γ^2   # pressure coefficient

    cl = 0.0
    offset = 1
    for airfoil in multielement.airfoils
        Nv = length(airfoil.x)

        # pull out the γ’s for *this* airfoil
        γ_seg = γ[offset:offset+Nv-1]

        # element-local panel lengths
        d = hypot.(diff(airfoil.x), diff(airfoil.y))

        # 2 * ∑ (γᵢ + γᵢ₊₁)/2 * Δsᵢ
        cl += 2 * sum(((γ_seg[1:end-1] .+ γ_seg[2:end]) ./ 2) .* d)

        offset += Nv
    end

    return InviscidSolution(multielement,  prob.alpha, LinearVortex, γ, cp, cl)
end


function linear_vortex_solver(airfoil::Airfoil,α::Real)
    xᵥ = airfoil.x
    yᵥ = airfoil.y
    xc = ( xᵥ[1:end-1] + xᵥ[2:end]) / 2
    yc = ( yᵥ[1:end-1] + yᵥ[2:end]) / 2
    Nv = length(xᵥ)
    Nc = Nv - 1

    θ = atan.( diff(yᵥ) , diff(xᵥ) )
    n̂ = [sin.(θ) -cos.(θ)] 
    uᵧ,vᵧ = induced_velocity_vortex_sheet(xc, yc, xᵥ, yᵥ, ones(size(xᵥ)))

    A = zeros(Nc+1,Nv)
    A[1:end-1,:] = uᵧ .* n̂[:,1] + vᵧ .* n̂[:,2]
    A[Nc+1,1] = 1
    A[Nc+1,Nv] = 1 

    RHS = Vector{Real}(undef,Nv)
    RHS[1:end-1] = -( cosd(α) .* n̂[:,1] +  sind(α)  .* n̂[:,2])

    if trailing_edge_thickness(airfoil) < 1e-4
        A[end-1,:] .= 0
        A[end-1,1] = 1
        A[end-1,2] = -2
        A[end-1,3] = 1
        A[end-1,end-2] = -1
        A[end-1,end-1] = 2
        A[end-1,end] = -1
        RHS[end-1] = 0
    end 
    
    γ = A \ RHS
    return γ
end 



function linear_vortex_solver(multielement::MultielementAirfoil,α::Real)
    airfoils = multielement.airfoils
    pitch = multielement.pitch
    chord = multielement.chord
    le_loc = multielement.le_loc
    num_airfoils = length(multielement.airfoils)
    num_vort = sum([length(airfoil.x) for airfoil in airfoils]) 
    num_col = num_vort-num_airfoils
    x_col = zeros(num_col)
    y_col = zeros(num_col)
    x_vort = zeros(num_vort)
    y_vort = zeros(num_vort)
    d = zeros(num_col)
    θ = zeros(num_col)
    array_size = [length(airfoil.x) for airfoil in airfoils]
    start_vort_idx = cumsum(array_size)-array_size.+1
    end_vort_idx = start_vort_idx + array_size .- 1
    array_size = [length(airfoil.x)-1 for airfoil in airfoils]
    start_col_idx = cumsum(array_size)-array_size.+1
    end_col_idx = start_col_idx + array_size .- 1

    # Create geometry from input information
    for (i,airfoil) in enumerate(airfoils)
        x = airfoil.x
        y = airfoil.y
        xy_vort = shift_scale_rotate.(x,y,le_loc[i][1],le_loc[i][2],pitch[i]+α,chord[i])     
        x_vort[start_vort_idx[i]:end_vort_idx[i]] = [xy[1] for xy in xy_vort]
        y_vort[start_vort_idx[i]:end_vort_idx[i]] = [xy[2] for xy in xy_vort]            
        x_col[start_col_idx[i]:end_col_idx[i]] = (x_vort[start_vort_idx[i]:end_vort_idx[i]-1]+x_vort[start_vort_idx[i]+1:end_vort_idx[i]])/2
        y_col[start_col_idx[i]:end_col_idx[i]] = (y_vort[start_vort_idx[i]:end_vort_idx[i]-1]+y_vort[start_vort_idx[i]+1:end_vort_idx[i]])/2
        dx = diff(x_vort[start_vort_idx[i]:end_vort_idx[i]])
        dy = diff(y_vort[start_vort_idx[i]:end_vort_idx[i]])
        d[start_col_idx[i]:end_col_idx[i]] = hypot.(dx, dy)
        θ[start_col_idx[i]:end_col_idx[i]] = atan.(dy, dx)
    end 

    n_hat = [sin.(θ) -cos.(θ)] 
    A = zeros(num_vort,num_vort) 
    for i = 1:num_airfoils # Vortex
        v1 = start_vort_idx[i]
        v2 = end_vort_idx[i]
        for j = 1:num_airfoils # Collocation
            c1 = start_col_idx[j]
            c2 = end_col_idx[j]
            u,v = induced_velocity_vortex_sheet(x_col[c1:c2],y_col[c1:c2],x_vort[v1:v2],y_vort[v1:v2],ones(length(v1:v2)))
            A[c1:c2,v1:v2] = A[c1:c2,v1:v2] + u .* n_hat[c1:c2,1] + v .* n_hat[c1:c2,2]
        end 
        A[end-num_airfoils+i,start_vort_idx[i]] = 1
        A[end-num_airfoils+i,end_vort_idx[i]] = 1
    end 
    
    RHS = zeros(num_vort)
    RHS[start_col_idx[1]:end_col_idx[end]] = - n_hat[start_col_idx[1]:end_col_idx[end],1] 

    for i = 1:num_airfoils
        if trailing_edge_thickness(airfoils[i]) < 1e-5
            A[end_col_idx[i],:] .= 0
            A[end_col_idx[i],start_vort_idx[i]] = 1
            A[end_col_idx[i],start_vort_idx[i]+1] = -2
            A[end_col_idx[i],start_vort_idx[i]+2] = 1
            A[end_col_idx[i],end_vort_idx[i]-2] = -1
            A[end_col_idx[i],end_vort_idx[i]-1] = 2
            A[end_col_idx[i],end_vort_idx[i]] = -1
            RHS[end_col_idx[i]] = 0
        end 
    end 
    γ = A\RHS

    #x = [x_vort[start_vort_idx[i]:end_vort_idx[i]] for i = 1:num_airfoils]
    #y = [y_vort[start_vort_idx[i]:end_vort_idx[i]] for i = 1:num_airfoils]
    #γ = [γ[start_vort_idx[i]:end_vort_idx[i]] for i = 1:num_airfoils]

    return γ
end 


"""
    induced_velocity_vortex_sheet(x,y,xᵧ,yᵧ,γ)

Computes the induced velocity at points (x,y) due to a vortex sheet defined by (xᵧ,yᵧ)
and strength γ. The sheet is discretized by linear vorticity panels. The ends of the sheet 
are connected by a constant stength source/vorticty panel. This is done to make sure the flow 
leaves each end tangentially 
""" 
function induced_velocity_vortex_sheet(x,y,xᵧ,yᵧ,γ)
    Nv = length(xᵧ)
    Nc = length(x)
    θ = atan.( diff(yᵧ) , diff(xᵧ) );
    d = hypot.( diff(xᵧ) , diff(yᵧ) )
    t_hat = [cos.(θ) sin.(θ)]
    θ_te = atan( yᵧ[1] - yᵧ[end] , xᵧ[1] - xᵧ[end] )
    d_te = hypot( xᵧ[1] - xᵧ[end] , yᵧ[1] - yᵧ[end] )
    t_te = [cos(θ_te) sin(θ_te)]
    b_te = (t_hat[1,:] - t_hat[end,:]) / norm( t_hat[1,:]-t_hat[end,:])
    dot_prod = abs(t_te[1]*b_te[1] + t_te[2]*b_te[2])
    cross_prod = abs(t_te[1]*b_te[2] - t_te[2]*b_te[1])
    uₐ = zeros(Nc,Nv)
    vₐ = zeros(Nc,Nv)
    uᵦ = zeros(Nc,Nv)
    vᵦ = zeros(Nc,Nv)
    u = zeros(Nc,Nv)
    v = zeros(Nc,Nv)
    for i = 1:Nc
        for j = 1:Nv-1
            (xp, yp) = panel_coordinates(x[i], y[i], xᵧ[j], yᵧ[j], θ[j])
            (upₐ, upᵦ, vpₐ, vpᵦ) = induced_velocity_linear_vortex.(γ[j], γ[j+1], xp, yp, d[j])
            (uₐ[i,j], vₐ[i,j]) = global_coordinates.(upₐ,vpₐ,0,0,θ[j])
            (uᵦ[i,j], vᵦ[i,j]) = global_coordinates.(upᵦ,vpᵦ,0,0,θ[j])
        end 
        (xp, yp) = panel_coordinates(x[i], y[i], xᵧ[end], yᵧ[end], θ_te);
        (upₐ, upᵦ, vpₐ, vpᵦ) = induced_velocity_te_panel(-γ[end], γ[1], xp, yp, d_te,cross_prod,dot_prod);
        (uₐ[i,Nv], vₐ[i,Nv]) = global_coordinates.(upₐ,vpₐ,0,0,θ_te)
        (uᵦ[i,Nv], vᵦ[i,Nv]) = global_coordinates.(upᵦ,vpᵦ,0,0,θ_te)
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


# 1) Scalar dispatch: computes (U, V) at a single point (x, y)
function induced_velocity(
    sol::InviscidSolution{A,LinearVortex},
    x::Real, y::Real,
) where A<:Airfoil
    # pull out panels
    x_sheet, y_sheet = sol.geometry.x, sol.geometry.y
    γ_sheet          = sol.strength

    # induced from sheet
    u_vals, v_vals = induced_velocity_vortex_sheet(
        x, y,
        x_sheet, y_sheet,
        γ_sheet
    )
    U = sum(u_vals) + cosd(sol.alpha)
    V = sum(v_vals) + sind(sol.alpha)

    return U, V
end

# 2) Vector dispatch: just build two matrices by calling the scalar version
function induced_velocity(
    sol::InviscidSolution{A,LinearVortex},
    xs::AbstractVector, ys::AbstractVector,
) where A<:Airfoil
    Nx, Ny = length(xs), length(ys)
    U = Array{Float64}(undef, Nx, Ny)
    V = Array{Float64}(undef, Nx, Ny)

    for i in 1:Nx, j in 1:Ny
        U[i,j], V[i,j] = induced_velocity(sol, xs[i], ys[j])
    end

    return U, V
end


function induced_velocity(sol::InviscidSolution{MultielementAirfoil,LinearVortex}, xs::AbstractVector, ys::AbstractVector) 
    Nx, Ny = length(xs), length(ys)
    U = zeros(Float64, Nx, Ny)
    V = zeros(Float64, Nx, Ny)

    # panel geometry & strengths
    ranges = segment_ranges(sol.geometry)
    for (n,airfoil) in enumerate(sol.geometry.airfoils)
        rng = ranges[n]
        
        xy_vort = shift_scale_rotate.(airfoil.x,airfoil.y,
                                        sol.geometry.le_loc[n][1],sol.geometry.le_loc[n][2],
                                        sol.geometry.pitch[n],sol.geometry.chord[n]) 

        x_sheet  = [xy[1] for xy in xy_vort]
        y_sheet  = [xy[2] for xy in xy_vort]
        γ_sheet  = sol.strength[rng]

        # build the field
        for i in 1:Nx, j in 1:Ny
            u_vals, v_vals = induced_velocity_vortex_sheet(
                xs[i], ys[j],
                x_sheet, y_sheet, γ_sheet
            )
            U[i, j] += sum(u_vals)
            V[i, j] += sum(v_vals)
        end

    end 

    # add freestream component
    U .+= cosd(sol.alpha)
    V .+= sind(sol.alpha)

    return U, V
end

function segment_ranges(multielement::MultielementAirfoil)
    offset = 1
    lengths = [length(airfoil.x) for airfoil in multielement.airfoils]
    ends = offset .+ cumsum(lengths) .- 1
    starts = offset .+ [0; cumsum(lengths[1:end-1])]
    return [starts[i]:ends[i] for i in eachindex(lengths)]
end