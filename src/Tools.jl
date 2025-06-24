
"""
    panel_coordinates(x,y,xₚ,yₚ,θₚ)

Converts vector (x,y) to the panel coordinates of a panel that 
is has an endpoint at xₚ,xₚ and orientation θₚ, measured counterclockwise 
from the positve x-axis 
""" 
function  panel_coordinates(x,y,xₚ,yₚ,θₚ)
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
function global_coordinates(x,y,xₚ,yₚ,θₚ)
    xₙ = cos(θₚ) * x - sin(θₚ) * y + xₚ
    yₙ = sin(θₚ) * x + cos(θₚ) * y + yₚ
    return xₙ,yₙ
end 

function shift_scale_rotate(x,y,x₀,y₀,θ,chord)
    xₙ = cosd(θ) * x*chord + sind(θ) * y*chord + x₀
    yₙ = -sind(θ) * x*chord + cosd(θ) * y*chord + y₀
    return xₙ,yₙ
end 


"""
    element_coordinates(me::MultielementAirfoil; deg=false)

Compute the global (x,y) coordinates of each airfoil in `me` after
  1. Scaling by chord[i],
  2. Rotating by pitch[i],
  3. Translating to le_loc[i].

Returns a Vector of Nx2 Float64 matrices.
If `deg=true`, interprets pitch angles in degrees.
"""
function element_coordinates(me::MultielementAirfoil)
    N = length(me.airfoils)
    coords = Vector{Matrix{Float64}}(undef, N)

    for i in 1:N
        foil = me.airfoils[i]
        θ = deg2rad(me.pitch[i])

        # 1) scale
        xs = foil.xinitial .* me.chord[i]
        ys = foil.yinitial .* me.chord[i]

        # 2) rotate
        cθ = cos(θ)
        sθ = sin(θ)
        xrot =  cθ .* xs .- sθ .* ys
        yrot =  sθ .* xs .+ cθ .* ys

        # 3) translate
        le = me.le_loc[i]
        xglob = xrot .+ le[1]
        yglob = yrot .+ le[2]

        coords[i] = hcat(xglob, yglob)
    end

    return coords
end
