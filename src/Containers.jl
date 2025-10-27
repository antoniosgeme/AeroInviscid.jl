abstract type FlowSolution   end
abstract type FlowSingularity end
abstract type LinearVortex <: FlowSingularity end

struct InviscidProblem{G<:AeroComponent,S<:FlowSingularity,T<:Real}
    geometry    :: G
    alpha       :: T
end

"""
Convenience constructors so you can pass the *type* of the singularity:

    InviscidProblem(geom, LinearVortex)
    InviscidProblem(geom, LinearVortex, α)
    InviscidProblem(geom, α, LinearVortex)
    InviscidProblem(geom; alpha=α)
    InviscidProblem(geom, LinearVortex; alpha=α)
"""
InviscidProblem(geom::G, ::Type{S}, α::T) where {G<:AeroComponent,S<:FlowSingularity,T<:Real} =
    InviscidProblem{G,S,T}(geom, α)

InviscidProblem(geom::G, α::T, ::Type{S}) where {G<:AeroComponent,S<:FlowSingularity,T<:Real} =
    InviscidProblem{G,S,T}(geom, α)

InviscidProblem(geom::G, α::T) where {G<:AeroComponent,T<:Real} =
    InviscidProblem{G,LinearVortex,T}(geom, α)

# Keyword argument constructors
InviscidProblem(geom::G; alpha::T) where {G<:AeroComponent,T<:Real} =
    InviscidProblem{G,LinearVortex,T}(geom, alpha)

InviscidProblem(geom::G, ::Type{S}; alpha::T) where {G<:AeroComponent,S<:FlowSingularity,T<:Real} =
    InviscidProblem{G,S,T}(geom, alpha)

# ────────────────────────────────────────────────
# Solution container 
# ────────────────────────────────────────────────
struct InviscidSolution{G<:AeroComponent,S<:FlowSingularity,T<:Real} <: FlowSolution
    geometry    :: G
    alpha       :: T
    strength    :: Vector{T}
    cp          :: Vector{T}
    cl          :: T
end

"""
    InviscidSolution(geom, S::Type, σ, cp, cl)

Small helper so you can write  defined
`InviscidSolution(airfoil, LinearVortex, σ, cp, cl)`
"""
InviscidSolution(geom::G, α::T, ::Type{S},
                 σ::Vector{T}, cp::Vector{T}, cl::T) where
                {G<:AeroComponent,S<:FlowSingularity,T<:Real} =
    InviscidSolution{G,S,T}(geom, α, σ, cp, cl)

# Flexible constructor that promotes types
InviscidSolution(geom::G, α::T1, ::Type{S},
                 σ::Vector{T2}, cp::Vector{T3}, cl::T4) where
                {G<:AeroComponent,S<:FlowSingularity,T1<:Real,T2<:Real,T3<:Real,T4<:Real} = begin
    T = promote_type(T1, T2, T3, T4)
    InviscidSolution{G,S,T}(geom, convert(T, α), convert(Vector{T}, σ), convert(Vector{T}, cp), convert(T, cl))
end



struct MultielementAirfoil{T} <: AeroComponent
    airfoils::Vector{Airfoil{T}}
    pitch::Vector{T}
    chord::Vector{T}
    le_loc::Vector{Vector{T}}
end

# Flexible constructor that promotes types for AD compatibility
function MultielementAirfoil(airfoils::Vector{Airfoil{T1}}, pitch::Vector{T2}, 
                              chord::Vector{T3}, le_loc::Vector{Vector{T4}}) where {T1,T2,T3,T4}
    T = promote_type(T1, T2, T3, T4)
    airfoils_promoted = [Airfoil(a.name, convert(Vector{T}, a.x), convert(Vector{T}, a.y)) for a in airfoils]
    pitch_promoted = convert(Vector{T}, pitch)
    chord_promoted = convert(Vector{T}, chord)
    le_loc_promoted = [convert(Vector{T}, loc) for loc in le_loc]
    return MultielementAirfoil{T}(airfoils_promoted, pitch_promoted, chord_promoted, le_loc_promoted)
end