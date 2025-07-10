# Assume unit freestream velocity and chord length

using AeroGeometry
using AeroInviscid
using Plots
using LinearAlgebra




"""
    space_geom(dx0, L, Np)

Generate `Np` geometrically‐spaced points on [0, L] with first interval `dx0`.
Returns `x` of length `Np`, with x[1]=0 and x[end]≈L.
"""
function space_geom(dx0::Real, L::Real, Np::Int)
    @assert Np > 1 "Need at least two points for spacing."
    N = Np - 1
    d = L / dx0

    # the root‐finding target: (r^N - 1)/(r - 1) = d
    f(r) = (r^N - 1) / (r - 1) - d

    # bracket the root: we know f(1+ε)<0, so start above 1
    r_lo = 1.0 + eps(Float64)
    r_hi = 1.5
    while f(r_hi) < 0
        r_hi *= 2
    end

    # bisection to ~1e-12 in r
    for _ in 1:100
        r_mid = (r_lo + r_hi) / 2
        if f(r_mid) > 0
            r_hi = r_mid
        else
            r_lo = r_mid
        end
        abs(r_hi - r_lo) < 1e-12 && break
    end
    r = (r_lo + r_hi) / 2

    # now build x: x[1]=0, then cumulative sums of dx0 * r^k for k=0..N-1
    ratios = r .^ (0:N-1)
    x = [0.0; cumsum(dx0 .* ratios)]
    return x
end


# ── wake container ────────────────────────────────────────────────────────────
struct InviscidWake{T}
    x   ::Matrix{T}   # 2×N coordinates
    s   ::Vector{T}   # arclength (continues airfoil s)
    t   ::Matrix{T}   # 2×N unit tangents
    ue  ::Vector{T}   # inviscid edge speed along wake (same α)
end

"""
    build_wake(sol; V∞ = 1.0) -> WakeResult

Predictor-corrector streamline march that grows a discrete wake behind the
trailing edge using the inviscid γ distribution in `sol.strength`.


"""
function build_wake(sol::InviscidSolution{A,El}; wakelen=10) where {A<:Airfoil,El<:LinearVortex}
    airfoil = sol.geometry
    X = coordinates(airfoil)
    S = surface_coordinates(airfoil)
    N = size(X,1)
    α = sol.alpha
    γ = sol.strength

    # --- choose number and spacing of wake points ----------------------------
    Nw  = ceil(Int, N/10 + 10*wakelen)
    ds1 = abs(((S[2]-S[1]) + (S[end]-S[end-1]))/2)
    sv  = space_geom(ds1, wakelen, Nw)

    Xw = zeros(Nw,2)         # coordinates
    tw  = similar(Xw)        # tangents

    # --- first wake node: tiny step behind trailing edge ---------------------
    xy1, xyN = X[1,:], X[end,:]
    xy_te    = 0.5 .* (xy1 + xyN)
    t_te     = xy1 .- xyN
    @assert t_te[1] > 0 "Airfoil must be CCW so wake points downstream (+x)."
    Xw[1,:] = xy_te .+ 1e-5 .* t_te
    sw       = S[end] .+ sv             # cumulative s for the wake nodes

    # --- streamline march ----------------------------------------------------
    for i in 1:Nw-1
        u,v = induced_velocity(sol,Xw[i,1],Xw[i,2])
        vt1  = [u,v] / norm([u,v])              
        tw[i,:] = vt1
        Xw[i+1,:] = Xw[i,:] .+ (sv[i+1]-sv[i]) .* vt1  # predictor

        u,v = induced_velocity(sol,Xw[i,1],Xw[i,2])
        vt2  = [u,v] / norm([u,v])              
        tw[i+1,:] = vt2
        Xw[i+1,:] = Xw[i,:] .+ (sv[i+1]-sv[i]) .* 0.5 .* (vt1 + vt2) # corrector
    end
    tw[end,:] .= tw[end-1,:]            # last tangent = previous one

    # --- edge speed along wake (same α) --------------------------------------
    ue = similar(sw)
    for i in eachindex(sw)
        u,v = induced_velocity(sol,Xw[i,1],Xw[i,2])
        ue[i] = dot([u,v], tw[i,:])
    end

    return InviscidWake(Xw, sw, tw, ue)
end

"""
    stagpoint_find(sol::InviscidSolution{A,LinearVortex},wake::InviscidWake)

Compute the leading‐edge stagnation point from an inviscid panel solution.

# Returns
A named tuple with fields
- `sstag::Float64`          : arclength location of the stagnation point  
- `sstag_g::NTuple{2,Float64}` : ∂sₛₜₐg/∂[γ₁, γ₂] on the two bracketing panels  
- `Istag::Tuple{Int,Int}`   : the two node‐indices before/after the zero crossing  
- `xstag::Tuple{Float64,Float64}` : the (x,y) coordinates of that point  
- `sgnue::Vector{Int}`      : +1/–1 “upper vs lower” sign on each panel  
- `xi::Vector{Float64}`     : distance along foil (and wake) from sₛₜₐg  
"""
function stagpoint_find(sol::InviscidSolution{A,LinearVortex},wake::InviscidWake) where A<:Airfoil
    γ = sol.strength                     # γ vector, length N
    s = surface_coordinates(sol.geometry)                   # arclength coords, length N
    N = length(γ)

    # find first panel with γ<0
    j = findfirst(<(0), γ)
    @assert j !== nothing || j == N || j == 1 "no stagnation point found"
    i1, i2 = j-1, j                      # the bracketing indices
    γ1, γ2 = γ[i1], γ[i2]
    s1, s2 = s[i1], s[i2]

    # interpolation weights
    den = γ2 - γ1
    w1, w2 = γ2/den, -γ1/den

    # stagnation‐point arclength & position
    sstag = w1*s1 + w2*s2
    xn, yn = sol.geometry.x, sol.geometry.y   # node coords
    xstag = (w1*xn[i1] + w2*xn[i2],
             w1*yn[i1] + w2*yn[i2])

    # derivative wrt [γ₁, γ₂]
    stg = γ2*(s1 - s2)/den^2
    sstag_g = ( stg, -stg )

    # sign switch: –1 on panels 1…i1, +1 thereafter
    sgnue = vcat( fill(1, i1), fill(-1, N - i1) )

    # distance from stagnation along foil, then wake (if any)
    xi_foil = abs.(s .- sstag)
    xi = vcat(xi_foil, wake.s .- sstag)

    return (
      sstag,
      sstag_g,
      i1, i2,
      xstag,
      sgnue,
      xi
    )
end

"""
    TE_info(airfoil::Airfoil)

Compute trailing‐edge info for an airfoil whose node coords are in `X` (2×N, ordered CW).

# Returns
- `t::Vector{Float64}`    : bisector vector (unit)
- `hTE::Float64`          : TE gap (displacement thickness)
- `dtdx::Float64`         : thickness slope ≈ sin(Δθ) between lower/upper tangents
- `tcp::Float64`          : |t × p|, for TE source strength
- `tdp::Float64`          : t·p, for TE vortex strength
"""
function TE_info(airfoil::Airfoil)
    # lower tangent (first→second node)
    X = coordinates(airfoil)
    t1 = X[1,:] .- X[2,:]
    t1 /= norm(t1)

    # upper tangent (last→second-last node)
    t2 = X[end,:] .- X[end-1,:]
    t2 /= norm(t2)

    # bisector (average tangent)
    t = t1 .+ t2
    t /= norm(t)

    # connector from lower→upper TE points
    s = X[end,:] .- X[1,:]

    # gap = cross(s, t) in 2D = det([s t])
    hTE = abs(-s[1]*t[2] + s[2]*t[1])

    # approx ∂t/∂x ≈ sin(θ₂−θ₁)
    dtdx = t1[1]*t2[2] - t2[1]*t1[2]

    # unit TE panel vector
    p = s / norm(s)

    # scalar cross‐product and dot‐product
    tcp = abs(t[1]*p[2] - t[2]*p[1])
    tdp = dot(t, p)

    return t, hTE, dtdx, tcp, tdp
end

"""
    wake_gap(sol::InviscidSolution{A,LinearVortex},wake::InviscidWake,xi::Vector) where A<:Airfoil

Compute the cubic extrapolation of the trailing‐edge boundary‐layer thickness
into the wake, per Drela (1989, IBL for Blunt TEs).

# Returns
- `wgap::Vector{Float64}` : wake‐gap δ* at each of the `sol.wake.N` wake points
"""
function wake_gap(sol::InviscidSolution{A,LinearVortex},wake::InviscidWake,xi::Vector) where A<:Airfoil
    # 1) trailing‐edge info
    # TE_info(xf) ⇒ (t_vec, hTE, dtdx, tcp, tdp)
    _, hTE, dtdx, _, _ = TE_info(sol.geometry)

    # 2) clip the slope
    flen = 2.5
    dtdx = clamp(dtdx, -3/flen, 3/flen)

    # 3) wake‐length scale
    Lw = flen * hTE

    # 4) assemble
    Nf = length(sol.strength)        # number of foil panels
    Nw = length(wake.s)          # number of wake panels
    wgap = zeros(Float64, Nw)

    # base‐arclength at start of wake
    s0 = xi[Nf]

    for i in 1:Nw
        # normalized distance into wake
        ξb = (xi[Nf + i] - s0) / Lw
        if ξb ≤ 1
            # cubic profile: hTE * [1 + (2 + flen*dtdx)*ξb] * (1−ξb)^2
            wgap[i] = hTE * (1 + (2 + flen*dtdx)*ξb) * (1 - ξb)^2
        end
    end

    return wgap
end





airfoil = Airfoil("NACA6409")
prob = InviscidProblem(airfoil,5)
sol = solve(prob)
plot(sol)

wake = build_wake(sol)

plot(airfoil)
scatter!(wake.x[:,1],wake.x[:,2])


sstag,sstag_g,i1, i2,xstag, sgnue, xi = stagpoint_find(sol,wake)

t, hTE, dtdx, tcp, tdp = TE_info(airfoil)

wgap = wake_gap(sol,wake,xi)