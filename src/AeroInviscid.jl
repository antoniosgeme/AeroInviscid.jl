module AeroInviscid

using LinearAlgebra
using Reexport
@reexport using AeroGeometry

include("Tools.jl")

include("VortexPanel2D.jl")


end # module AeroInviscid
