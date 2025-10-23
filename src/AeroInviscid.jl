module AeroInviscid

using LinearAlgebra
using Reexport
@reexport using AeroGeometry

export InviscidProblem, solve, LinearVortex, MultielementAirfoil, 
induced_velocity, streamlines, element_coordinates, shift_scale_rotate, 
streamlines_from_grid, InviscidSolution

include("Containers.jl")

include("Tools.jl")

include("VortexPanel2D.jl")

include("PlotViz.jl")


end # module AeroInviscid
