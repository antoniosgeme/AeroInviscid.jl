abstract type FlowSolution   end
abstract type FlowSingularity end
abstract type LinearVortex <: FlowSingularity end

struct InviscidProblem{G<:Geometry,S<:FlowSingularity}
    geometry    :: G
    alpha       :: Float64
end

"""
Convenience constructors so you can pass the *type* of the singularity:

    InviscidProblem(geom, LinearVortex)
    InviscidProblem(geom, LinearVortex, α)
    InviscidProblem(geom, α, LinearVortex)
"""
InviscidProblem(geom::G, ::Type{S},α) where {G<:Geometry,S<:FlowSingularity} =
    InviscidProblem{G,S}(geom, α)

InviscidProblem(geom::G, α, ::Type{S}) where {G<:Geometry,S<:FlowSingularity} =
    InviscidProblem{G,S}(geom, α)

InviscidProblem(geom::G,α) where G<:Geometry =
    InviscidProblem{G,LinearVortex}(geom, α)

# ────────────────────────────────────────────────
# Solution container 
# ────────────────────────────────────────────────
struct InviscidSolution{G<:Geometry,S<:FlowSingularity} <: FlowSolution
    geometry    :: G
    alpha       :: Real
    strength    :: Vector{Float64}
    cp          :: Vector{Float64}
    cl          :: Float64
end

"""
    InviscidSolution(geom, S::Type, σ, cp, cl)

Small helper so you can write  
`InviscidSolution(airfoil, LinearVortex, σ, cp, cl)`
"""
InviscidSolution(geom::G, α::Real, ::Type{S},
                 σ::Vector{<:Real}, cp::Vector{<:Real}, cl::Real) where
                {G<:Geometry,S<:FlowSingularity} =
    InviscidSolution{G,S}(geom, α, σ, cp, cl)



struct MultielementAirfoil <: Geometry
    airfoils::Vector{Airfoil}
    pitch::Vector{<:Real}
    chord::Vector{<:Real}
    le_loc::Vector{Vector{Real}}
end 