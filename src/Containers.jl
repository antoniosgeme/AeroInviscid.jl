abstract type FlowSolution   end
abstract type FlowSingularity end
abstract type LinearVortex <: FlowSingularity end

struct InviscidProblem{G<:AeroComponent,S<:FlowSingularity}
    geometry    :: G
    alpha       :: Float64
end

"""
Convenience constructors so you can pass the *type* of the singularity:

    InviscidProblem(geom, LinearVortex)
    InviscidProblem(geom, LinearVortex, α)
    InviscidProblem(geom, α, LinearVortex)
"""
InviscidProblem(geom::G, ::Type{S},α) where {G<:AeroComponent,S<:FlowSingularity} =
    InviscidProblem{G,S}(geom, α)

InviscidProblem(geom::G, α, ::Type{S}) where {G<:AeroComponent,S<:FlowSingularity} =
    InviscidProblem{G,S}(geom, α)

InviscidProblem(geom::G,α) where G<:AeroComponent =
    InviscidProblem{G,LinearVortex}(geom, α)

# ────────────────────────────────────────────────
# Solution container 
# ────────────────────────────────────────────────
struct InviscidSolution{G<:AeroComponent,S<:FlowSingularity} <: FlowSolution
    geometry    :: G
    alpha       :: Float64
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
                {G<:AeroComponent,S<:FlowSingularity} =
    InviscidSolution{G,S}(geom, α, σ, cp, cl)



struct MultielementAirfoil{T} <: AeroComponent
    airfoils::Vector{Airfoil{T}}
    pitch::Vector{T}
    chord::Vector{T}
    le_loc::Vector{Vector{T}}
end