
"""
    panel_coordinates(x,y,xₚ,yₚ,θₚ)

Converts vector (x,y) to the panel coordinates of a panel that 
is has an endpoint at xₚ,xₚ and orientation θₚ, measured counterclockwise 
from the positve x-axis 
""" 
function panel_coordinates(x::Real,y::Real,xₚ::Real,yₚ::Real,θₚ::Real)
    xpanel = cos(θₚ) * (x - xₚ) + sin(θₚ) * (y - yₚ)
    ypanel = -sin(θₚ) * (x - xₚ) + cos(θₚ) * (y - yₚ)
    return xpanel,ypanel
end 

"""
    global_coordinates(x,y,xₚ,yₚ,θₚ)

Converts vector (x,y) to the global coordinates of a panel that 
is has an endpoint at xₚ,xₚ and orientation θₚ, measured counterclockwise 
from the positve x-axis, in the global coordinate system
""" 
function global_coordinates(x::Real,y::Real,xₚ::Real,yₚ::Real,θₚ::Real)
    xₙ = cos(θₚ) * x - sin(θₚ) * y + xₚ
    yₙ = sin(θₚ) * x + cos(θₚ) * y + yₚ
    return xₙ,yₙ
end 

function shift_scale_rotate(x::Real,y::Real,x₀::Real,y₀::Real,θ::Real,chord::Real)
    xₙ = cosd(θ) * x*chord + sind(θ) * y*chord + x₀
    yₙ = -sind(θ) * x*chord + cosd(θ) * y*chord + y₀
    return xₙ,yₙ
end 


"""
    element_coordinates(me::MultielementAirfoil)

Compute the global (x,y) coordinates of each airfoil in `me` after
  1. Scaling by chord[i],
  2. Rotating by pitch[i],
  3. Translating to le_loc[i].

Returns a Vector of Nx2 Float64 matrices.
If `deg=true`, interprets pitch angles in degrees.
"""
function element_coordinates(me::MultielementAirfoil{T}) where T
    N = length(me.airfoils)
    coords = Vector{Matrix{Float64}}(undef, N)

    for i in 1:N
        foil = me.airfoils[i]
        θ = deg2rad(me.pitch[i])

        # 1) scale
        xs = foil.x .* me.chord[i]
        ys = foil.y .* me.chord[i]

        # 2) rotate
        xrot =  cos(θ) .* xs .+ sin(θ) .* ys
        yrot =  -sin(θ) .* xs .+ cos(θ) .* ys

        # 3) translate
        le = me.le_loc[i]
        xglob = xrot .+ le[1]
        yglob = yrot .+ le[2]

        coords[i] = hcat(xglob, yglob)
    end

    return coords
end

"""
    element_coordinates(af::Airfoil)

Return a Vector of one Nx2 Float64 matrix for a single airfoil geometry,
so callers can iterate uniformly over elements regardless of single or
multi-element configurations.
"""
function element_coordinates(af::Airfoil{T}) where T
    return [hcat(af.x, af.y)]
end


# Mask inside airfoil shapes (no flow inside body)
function mask_inside_polygon(xp::AbstractVector, yp::AbstractVector, X, Y)
    n = length(xp)
    @assert n == length(yp) && n ≥ 3
    inside = falses(size(X))

    x1 = xp; y1 = yp
    x2 = circshift(xp, -1); y2 = circshift(yp, -1)

    @inbounds for i in 1:n
        ycross = ((y1[i] .<= Y) .& (Y .< y2[i])) .| ((y2[i] .<= Y) .& (Y .< y1[i]))
        xints  = (x2[i] - x1[i]) .* (Y .- y1[i]) ./ (y2[i] - y1[i]) .+ x1[i]
        inside .= xor.(inside, ycross .& (X .< xints))
    end
    return inside
end