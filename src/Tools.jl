
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
