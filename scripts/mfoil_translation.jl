using AeroGeometry
using LinearAlgebra
using Printf: Format, format





include("mfoil_types.jl")
using .MfoilTypes

# Simple verbosity-gated printer with dynamic format strings
function vprint(param, lvl, fmt::AbstractString, args...)
    if param.verb >= lvl
        buf = IOBuffer()
        format(buf, Format(fmt), args...)
        print(String(take!(buf)))
    end
    return nothing
end


function init_M(alpha::Real=0.0)
    # Build Mfoil
    snaca = "2412"
    foil = Airfoil("NACA"*snaca)
    M = Mfoil()
    # set geometry name
    M.geom.name = "NACA $snaca"
    # generate raw geometry points and assign
    coords = coordinates(foil)
    coords = collect(transpose(coords) )
    M.geom.xpoint = coords
    M.geom.npoint = size(coords, 2)

    # Simple panel mesh: copy raw points
    M.foil.x = M.geom.xpoint
    M.foil.N = size(M.foil.x, 2)
    # compute arclength s
    d = sqrt.(diff(M.foil.x[1, :]).^2 .+ diff(M.foil.x[2, :]).^2)
    M.foil.s = vcat(0.0, cumsum(d))
    # compute tangents t = dx/ds
    tangs = @. (M.foil.x[:, 2:end] - M.foil.x[:, 1:end-1]) / d'
    M.foil.t = hcat(tangs, tangs[:, end])

    # Wake empty by default
    M.wake.N = 0; 
    M.wake.x = zeros(2,0); 
    M.wake.s = Float64[]; 
    M.wake.t = zeros(2,0)

    # Operating conditions
    M.oper.Vinf = 1.0
    M.oper.alpha = alpha
    M.oper.rho   = 1.0
    M.oper.Re    = 1e5
    M.oper.Ma    = 0.0
    M.oper.viscous = false   # inviscid default
    M.oper.givencl = false

    return M 
end 

function enforce_CW_and_TE!(M)
    X = M.foil.x
    N = size(X,2)

    # Ensure clockwise
    if signed_area(X) > 0                # CCW -> flip to CW
        X = reverse(X; dims=2)
    end

    # Ensure column 1 is the lower TE and column N is the upper TE
    # (If your generator doesn't guarantee this, rotate columns.)
    # Heuristic: pick the two closest points (the TE pair), then
    # make the lower one be col 1 and the other be col N.
    d = sum((X[:,1] .- X[:,end]).^2)
    # If these are not your TE points, adjust your generator to put
    # TE endpoints at columns 1 and N consistently.

    if X[2,1] > X[2,end]     # first is above second -> swap TE ends by rotation
        X = X[:, [size(X,2); 1:size(X,2)-1]]  # rotate so previous last becomes first
    end

    M.foil.x = X
end

"""
    init_thermo!(M::Mfoil)

Initialize thermodynamic parameters in `M.param` from `M.oper` & `M.geom`.
"""
function init_thermo!(M::Mfoil)
    # gas constants
    g   = M.param.gam
    gmi = g - 1

    # freestream
    ρ∞   = M.oper.rho
    V∞   = M.oper.Vinf
    M.param.Vinf  = V∞
    M.param.muinf = ρ∞ * V∞ * M.geom.chord / M.oper.Re

    # Mach number
    M∞ = M.oper.Ma
    M.param.Minf = M∞

    # Sutherland’s reference temperature ratio
    finf = 1.0
    if M∞ > 0
        M.param.KTb = sqrt(1 - M∞^2)
        M.param.KTl = M∞^2 / (1 + M.param.KTb)^2

        # stagnation enthalpy
        M.param.H0 = (1 + 0.5*gmi*M∞^2) * V∞^2 / (gmi*M∞^2)

        # freestream/stag temperature ratio
        Tr = 1 - 0.5*V∞^2 / M.param.H0

        # Sutherland’s law
        finf = Tr^1.5 * (1 + M.param.Tsrat) / (Tr + M.param.Tsrat)

        # sonic cp line
        M.param.cps = 2/(g*M∞^2) * (((1 + 0.5*gmi*M∞^2)/(1 + 0.5*gmi))^(g/gmi) - 1)
    end

    # stagnation viscosity & density
    M.param.mu0  = M.param.muinf / finf
    M.param.rho0 = ρ∞ * (1 + 0.5*gmi*M∞^2)^(1/gmi)

    return nothing
end

function TE_info(X::AbstractMatrix{<:Real})
    # lower tangent: from node 2 to node 1
    t1 = X[:, 1] .- X[:, 2]
    t1 /= norm(t1)

    # upper tangent: from node end−1 to node end
    t2 = X[:, end] .- X[:, end-1]
    t2 /= norm(t2)

    # bisector of t1 and t2
    t = 0.5 .* (t1 .+ t2)
    t /= norm(t)

    # connector vector from lower to upper TE
    s = X[:, end] .- X[:, 1]

    # signed gap = s × t  (in 2D, cross‐product scalar)
    hTE = -s[1]*t[2] + s[2]*t[1]

    # thickness slope ≈ sin(angle(t1,t2))
    dtdx = t1[1]*t2[2] - t2[1]*t1[2]

    # unit vector along TE panel
    p = s / norm(s)

    # cross and dot for TE panel strength
    tcp = abs(t[1]*p[2] - t[2]*p[1])
    tdp = dot(t, p)

    return t, hTE, dtdx, tcp, tdp
end

function panel_linvortex_stream(Xj::AbstractMatrix{<:Real}, xi::AbstractVector{<:Real})
    # panel endpoints
    xj1, zj1 = Xj[1,1], Xj[2,1]
    xj2, zj2 = Xj[1,2], Xj[2,2]

    # panel‐aligned tangent and normal
    dx, dz = xj2 - xj1, zj2 - zj1
    d = hypot(dx, dz)
    t = (dx, dz) ./ d
    n = (-t[2], t[1])

    # control point relative to first endpoint
    xz = (xi[1] - xj1, xi[2] - zj1)
    x = xz[1]*t[1] + xz[2]*t[2]
    z = xz[1]*n[1] + xz[2]*n[2]

    # distances & angles
    r1 = hypot(x, z)
    r2 = hypot(x - d, z)
    θ1 = atan(z, x)
    θ2 = atan(z, x - d)

    # avoid log(0)
    ϵ = 1e-10
    logr1 = r1 < ϵ ? 0.0 : log(r1)
    logr2 = r2 < ϵ ? 0.0 : log(r2)

    # streamfunction components
    P1 = (0.5/π)*( z*(θ2 - θ1) - d + x*logr1 - (x - d)*logr2 )
    P2 = x*P1 + (0.5/π)*( 0.5*r2^2*logr2 - 0.5*r1^2*logr1 - r2^2/4 + r1^2/4 )

    # influence coefficients
    a = P1 - P2/d
    b = P2/d

    return a, b
end

function panel_constsource_stream(Xj::AbstractMatrix{<:Real}, xi::AbstractVector{<:Real})
    # panel endpoints
    xj1, zj1 = Xj[1,1], Xj[2,1]
    xj2, zj2 = Xj[1,2], Xj[2,2]

    # panel‐aligned tangent and normal
    dx, dz = xj2 - xj1, zj2 - zj1
    d = hypot(dx, dz)
    t = (dx, dz) ./ d
    n = (-t[2], t[1])

    # control point relative to first endpoint
    xz = (xi[1] - xj1, xi[2] - zj1)
    x = xz[1]*t[1] + xz[2]*t[2]
    z = xz[1]*n[1] + xz[2]*n[2]

    # distances & angles
    r1 = hypot(x, z)
    r2 = hypot(x - d, z)
    θ1 = atan(z, x)
    θ2 = atan(z, x - d)

    # handle singularities
    ϵ = 1e-9
    if r1 < ϵ
        logr1 = 0.0
        θ1 = π
        θ2 = π
    else
        logr1 = log(r1)
    end
    if r2 < ϵ
        logr2 = 0.0
        θ1 = 0.0
        θ2 = 0.0
    else
        logr2 = log(r2)
    end

    # base streamfunction
    P = ( x*(θ1 - θ2) + d*θ2 + z*logr1 - z*logr2 ) / (2π)

    # correction for branch cut
    dP = d
    if θ1 + θ2 > π
        P -= 0.25 * dP
    else
        P += 0.75 * dP
    end

    return P
end

function build_gamma!(M::MfoilTypes.Mfoil, α::Real)
    N = M.foil.N
    A   = zeros(N+1, N+1)
    rhs = zeros(N+1, 2)

    # get trailing‐edge info
    _, hTE, _, tcp, tdp = TE_info(M.foil.x)
    nogap = abs(hTE) < 1e-10 * M.geom.chord

    # optional verbose print
    if M.param.verb ≥ 1
        println("\n <<< Solving inviscid problem >>> \n")
    end

    # assemble A and RHS for 0° and 90°
    for i in 1:N
        xi = M.foil.x[:, i]
        for j in 1:N-1
            aij, bij = panel_linvortex_stream(M.foil.x[:, [j, j+1]], xi)
            A[i, j  ] += aij
            A[i, j+1] += bij
            A[i, N+1] = -1
        end
        # freestream‐only contribution
        rhs[i, 1] = -xi[2]
        rhs[i, 2] =  xi[1]

        # TE source contribution
        a = panel_constsource_stream(M.foil.x[:, [N,1]], xi)
        A[i, 1] -= 0.5*tcp * a
        A[i, N] += 0.5*tcp * a

        # TE vortex‐panel contribution
        a2, b2 = panel_linvortex_stream(M.foil.x[:, [N,1]], xi)
        A[i, 1] -= -0.5*tdp * (a2 + b2)
        A[i, N] += -0.5*tdp * (a2 + b2)
    end

    # sharp TE extrapolation if no gap
    if nogap
        A[N, :] .= 0
        A[N, [1,2,3, N-2, N-1, N]] .= (1, -2, 1, -1, 2, -1)
    end

    # Kutta condition
    A[N+1, 1] = 1
    A[N+1, N] = 1

    # solve
    M.isol.AIC     = A
    g               = A \ rhs
    M.isol.gamref  = g[1:end-1, :]                    # discard the streamfunction DOF
    M.isol.gam     = M.isol.gamref[:,1] .* cosd(α) .+  # γ(α) = γ₀ cos α + γ₉₀ sin α
                       M.isol.gamref[:,2] .* sind(α)

    return nothing
end

function get_ueinv(M::Mfoil)
    @assert !isempty(M.isol.gam) "No inviscid solution"

    # freestream direction cosines (degrees)
    α  = M.oper.alpha
    cs = [cosd(α); sind(α)]

    # airfoil edge velocity: signed gamma projection
    # M.isol.sgnue :: Vector{Int}, M.isol.gamref :: Matrix{Float64} (N×2)
    uea = M.isol.sgnue .* (M.isol.gamref * cs)

    # wake edge velocity (if viscous + wake panels exist)
    uew = Float64[]
    if M.oper.viscous && M.wake.N > 0
        uew = M.isol.uewiref * cs
        # enforce continuity: upper surface → wake
        uew[1] = uea[end]
    end

    return vcat(uea, uew)
end

function build_param(M::Mfoil, is::Int)
    p = deepcopy(M.param)
    p.wake = (is == 3)
    p.turb = p.wake       # wake is always turbulent
    p.simi = false        # similarity station only set later
    return p
end

function station_param(M::Mfoil, p::Param, i::Int)
    p.turb = M.vsol.turb[i]
    p.simi = i in M.isol.Istag
    return p
end


function get_uk(u, param::Param)
    if param.Minf > 0
        ℓ    = param.KTl
        Vinf = param.Vinf
        den   = 1 .- ℓ .* (u ./ Vinf).^2
        den_u = -2 .* ℓ .* u ./ (Vinf^2)
        uk    = (1 - ℓ) .* u ./ den
        uk_u  = (1 - ℓ) ./ den .- (uk ./ den) .* den_u
        return uk, uk_u
    else
        # incompressible: no correction
        return u, one.(u)
    end
end

# ---------------------------
# Utility: upwind weighting
# ---------------------------
function get_upw(U1::AbstractVector{<:Real}, U2::AbstractVector{<:Real}, param)
    Hk1, Hk1_U1 = get_Hk(U1, param)
    Hk2, Hk2_U2 = get_Hk(U2, param)
    Z = zeros(length(Hk1_U1))
    Hut = 1.0
    C = param.wake ? 1.0 : 5.0
    Huc = C*Hut/Hk2^2
    Huc_U = vcat(Z, -2*Huc/Hk2 .* Hk2_U2)
    aa  = (Hk2-1)/(Hk1-1)
    sga = sign(aa)
    la  = log(sga*aa)
    la_U = vcat( -1/(Hk1-1) .* Hk1_U1,  1/(Hk2-1) .* Hk2_U2 )
    Hls  = la^2
    Hls_U = 2*la .* la_U
    if Hls > 15
        Hls  = 15.0
        Hls_U .= 0.0
    end
    upw   = 1 - 0.5*exp(-Hls*Huc)
    upw_U = -0.5*exp(-Hls*Huc) .* ( -Hls_U*Huc .- Hls .* Huc_U )
    return upw, upw_U
end

function upwind(upw::Real, upw_U::AbstractVector{<:Real},
                f1::Real, f1_U1::AbstractVector{<:Real},
                f2::Real, f2_U2::AbstractVector{<:Real})
    f   = (1-upw)*f1 + upw*f2
    # derivative layout [d/dU1, d/dU2] (8 entries total for 4+4)
    f_U = (-upw_U).*f1 .+ upw_U.*f2 .+ vcat((1-upw).*f1_U1, upw.*f2_U2)
    return f, f_U
end

# ---------------------------
# Smooth limiter
# ---------------------------
function slimit_Hkc(Hkc0::Real)
    Hl, Hh = 0.01, 0.05
    if Hkc0 < Hh
        rn = (Hkc0 - Hl)/(Hh - Hl)
        rn_H = 1/(Hh - Hl)
        if rn < 0
            rn   = 0.0
            rn_H = 0.0
        end
        rf    = 3*rn^2 - 2*rn^3
        rf_rn = 6*rn - 6*rn^2
        Hkc   = Hl + rf*(Hh - Hl)
        rd    = rf_rn*rn_H*(Hh - Hl)
    else
        Hkc, rd = Hkc0, 1.0
    end
    return Hkc, rd
end

# ---------------------------
# Equilibrium ue' term
# ---------------------------
function get_uq(ds::Real, ds_U::AbstractVector{<:Real},
                cf::Real, cf_U::AbstractVector{<:Real},
                Hk::Real, Hk_U::AbstractVector{<:Real},
                Ret::Real, Ret_U::AbstractVector{<:Real},
                param)
    β  = param.GB
    A  = param.GA
    C  = param.GC
    if param.wake
        A *= param.Dlr
        C  = 0.0
    end
    # hard lower bounds on Hk (per original)
    if param.wake && (Hk < 1.00005); Hk = 1.00005; Hk_U .= 0.0; end
    if !param.wake && (Hk < 1.05);   Hk = 1.05;    Hk_U .= 0.0; end

    Hkc   = Hk - 1 - C/Ret
    Hkc_U = Hk_U .+ (C/Ret^2) .* Ret_U
    # optional smoothing: (Hkc, rd) = slimit_Hkc(Hkc); Hkc_U .= rd .* Hkc_U
    if Hkc < 0.01
        Hkc = 0.01
        Hkc_U .= 0.0
    end

    ut   = 0.5*cf - (Hkc/(A*Hk))^2
    ut_U = 0.5 .* cf_U .- 2*(Hkc/(A*Hk)) .* ( Hkc_U/(A*Hk) .- (Hkc/(A*Hk^2)).*Hk_U )

    uq   = ut/(β*ds)
    uq_U = ut_U/(β*ds) .- (uq/ds) .* ds_U
    return uq, uq_U
end

# ---------------------------
# Shear stress at transition
# ---------------------------
function get_cttr(U::AbstractVector{<:Real}, param)
    # Transition just before wake: force wake=false locally
    p = deepcopy(param); p.wake = false
    cteq, cteq_U = get_cteq(U, p)
    Hk, Hk_U = get_Hk(U, p)
    if Hk < 1.05
        Hk = 1.05; Hk_U .= 0.0
    end
    C, E = p.CtauC, p.CtauE
    c    = C*exp(-E/(Hk-1))
    c_U  = c*E/(Hk-1)^2 .* Hk_U
    cttr   = c*cteq
    cttr_U = c_U .* cteq .+ c .* cteq_U
    return cttr, cttr_U
end

function get_cteq(U::AbstractVector{<:Real}, param)
    CC = 0.5/(param.GA^2*param.GB)
    C  = param.GC
    Hk, Hk_U = get_Hk(U, param)
    Hs, Hs_U = get_Hs(U, param)
    H,  H_U  = get_H(U)
    Ret, Ret_U = get_Ret(U, param)

    if param.wake
        if Hk < 1.00005; Hk = 1.00005; Hk_U .= 0.0; end
        Hkc   = Hk - 1
        Hkc_U = Hk_U
    else
        if Hk < 1.05; Hk = 1.05; Hk_U .= 0.0; end
        Hkc   = Hk - 1 - C/Ret
        Hkc_U = Hk_U .+ (C/Ret^2) .* Ret_U
        # optional smoother: (Hkc, rd) = slimit_Hkc(Hkc); Hkc_U .= rd .* Hkc_U
        if Hkc < 0.01; Hkc = 0.01; Hkc_U .= 0.0; end
    end

    num   = CC*Hs*(Hk-1)*Hkc^2
    num_U = CC .* ( Hs_U*(Hk-1)*Hkc^2 .+ Hs .* Hk_U .* Hkc^2 .+ Hs*(Hk-1)*2Hkc .* Hkc_U )
    den   = (1 - get_Us(U, param)[1]) * H * Hk^2
    Us, Us_U = get_Us(U, param)
    den_U = (-Us_U).*H*Hk^2 .+ (1-Us).*H_U*Hk^2 .+ (1-Us).*H.*(2Hk).*Hk_U

    cteq   = sqrt(num/den)
    cteq_U = 0.5/cteq .* ( num_U/den .- (num/den^2).*den_U )
    return cteq, cteq_U
end

# ---------------------------
# Shape parameters
# ---------------------------
function get_H(U::AbstractVector{<:Real})
    th, ds = U[1], U[2]
    H = ds/th
    H_U = [-H/th, 1/th, 0.0, 0.0]
    return H, H_U
end

function get_Hw(U::AbstractVector{<:Real}, wgap::Real)
    th = U[1]
    Hw = wgap/th
    Hw_U = [-Hw/th, 0.0, 0.0, 0.0]
    return Hw, Hw_U
end

function get_Hk(U::AbstractVector{<:Real}, param)
    H, H_U = get_H(U)
    if param.Minf > 0
        M2, M2_U = get_Mach2(U, param)
        den = 1 + 0.113*M2
        Hk  = (H - 0.29*M2)/den
        den_M2 = 0.113
        Hk_U = (H_U .- 0.29 .* M2_U) ./ den .- (Hk/den)*den_M2 .* M2_U
        return Hk, Hk_U
    else
        return H, H_U
    end
end

function get_Hs(U::AbstractVector{<:Real}, param)
    Hk, Hk_U = get_Hk(U, param)
    # lower bounds for Hk
    if param.wake && (Hk < 1.00005); Hk = 1.00005; Hk_U .= 0.0; end
    if !param.wake && (Hk < 1.05);   Hk = 1.05;    Hk_U .= 0.0; end

    if param.turb
        Hsmin, dHsinf = 1.5, 0.015
        Ret, Ret_U = get_Ret(U, param)

        Ho, Ho_U = 4.0, zeros(4)
        if Ret > 400
            Ho   = 3 + 400/Ret
            Ho_U = -400/Ret^2 .* Ret_U
        end
        Reb, Reb_U = Ret, Ret_U
        if Ret < 200
            Reb   = 200.0
            Reb_U .= 0.0
        end
        if Hk < Ho
            Hr   = (Ho - Hk)/(Ho - 1)
            Hr_U = (Ho_U - Hk_U)/(Ho - 1) .- (Ho - Hk)/(Ho - 1)^2 .* Ho_U
            aa   = (2 - Hsmin - 4/Reb)*Hr^2
            aa_U = (4/Reb^2 .* Reb_U)*Hr^2 .+ (2 - Hsmin - 4/Reb)*2Hr .* Hr_U
            Hs   = Hsmin + 4/Reb + aa * 1.5/(Hk + 0.5)
            Hs_U = -4/Reb^2 .* Reb_U .+ aa_U * 1.5/(Hk + 0.5) .- aa*1.5/(Hk + 0.5)^2 .* Hk_U
        else
            lrb   = log(Reb); lrb_U = (1/Reb) .* Reb_U
            aa    = Hk - Ho + 4/lrb
            aa_U  = Hk_U - Ho_U .- 4/lrb^2 .* lrb_U
            bb    = 0.007*lrb/aa^2 + dHsinf/Hk
            bb_U  = 0.007*(lrb_U/aa^2 - 2lrb/aa^3 .* aa_U) .- dHsinf/Hk^2 .* Hk_U
            Hs    = Hsmin + 4/Reb + (Hk - Ho)^2 * bb
            Hs_U  = -4/Reb^2 .* Reb_U .+ 2*(Hk - Ho).*(Hk_U - Ho_U).*bb .+ (Hk - Ho)^2 .* bb_U
        end
        # slight Mach correction
        M2, M2_U = get_Mach2(U, param)
        den = 1 + 0.014*M2
        den_M2 = 0.014
        Hs  = (Hs + 0.028*M2)/den
        Hs_U = (Hs_U .+ 0.028 .* M2_U)./den .- (Hs/den)*den_M2 .* M2_U
    else
        a = Hk - 4.35
        if Hk < 4.35
            num   = 0.0111*a^2 - 0.0278*a^3
            Hs    = num/(Hk + 1) + 1.528 - 0.0002*(a*Hk)^2
            Hs_Hk = (0.0111*2a - 0.0278*3a^2)/(Hk + 1) - num/(Hk + 1)^2 - 0.0002*2a*Hk*(Hk + a)
        else
            Hs    = 0.015*a^2/Hk + 1.528
            Hs_Hk = 0.015*2a/Hk - 0.015*a^2/Hk^2
        end
        Hs_U = Hs_Hk .* Hk_U
    end
    return Hs, Hs_U
end

function get_Hss(U::AbstractVector{<:Real}, param)
    M2, M2_U = get_Mach2(U, param)
    Hk, Hk_U = get_Hk(U, param)
    num   = 0.064/(Hk - 0.8) + 0.251
    num_U = -0.064/(Hk - 0.8)^2 .* Hk_U
    Hss   = M2*num
    Hss_U = M2_U .* num .+ M2 .* num_U
    return Hss, Hss_U
end

# ---------------------------
# Compressibility helpers
# ---------------------------
function get_cp(u::Real, param)
    Vinf = param.Vinf
    cp   = 1 - (u/Vinf)^2
    cp_u = -2u / Vinf^2
    if param.Minf > 0
        l, b = param.KTl, param.KTb
        den = b + 0.5*l*(1+b)*cp
        den_cp = 0.5*l*(1+b)
        cp   = cp/den
        cp_u = cp_u * (1 - cp*den_cp) / den
    end
    return cp, cp_u
end

function get_cp(u::AbstractVector{<:Real}, param)
    cp  = similar(u, Float64)
    cpu = similar(u, Float64)
    for i in eachindex(u)
        cp[i], cpu[i] = get_cp(u[i], param)
    end
    return cp, cpu
end

function get_uk(u::Real, param)
    if param.Minf > 0
        l, Vinf = param.KTl, param.Vinf
        den   = 1 - l*(u/Vinf)^2
        den_u = -2l*u/Vinf^2
        uk    = u*(1-l)/den
        uk_u  = (1-l)/den - (uk/den)*den_u
        return uk, uk_u
    else
        return u, 1.0
    end
end

function get_Mach2(U::AbstractVector{<:Real}, param)
    if param.Minf > 0
        H0, g = param.H0, param.gam
        uk, uk_u = get_uk(U[4], param)
        c2    = (g - 1) * (H0 - 0.5*uk^2)
        c2_uk = (g - 1) * (-uk)
        M2    = uk^2 / c2
        M2_uk = 2uk/c2 - (M2/c2)*c2_uk
        M2_U  = [0.0, 0.0, 0.0, M2_uk*uk_u]
        return M2, M2_U
    else
        return 0.0, zeros(4)
    end
end

# ---------------------------
# Thickness metrics
# ---------------------------
function get_de(U::AbstractVector{<:Real}, param)
    Hk, Hk_U = get_Hk(U, param)
    aa   = 3.15 + 1.72/(Hk - 1)
    aa_U = -1.72/(Hk - 1)^2 .* Hk_U
    de   = U[1]*aa + U[2]
    de_U = [aa, 1.0, 0.0, 0.0] .+ U[1] .* aa_U
    dmx  = 12.0
    if de > dmx*U[1]
        de   = dmx*U[1]
        de_U = [dmx, 0.0, 0.0, 0.0]
    end
    return de, de_U
end

# ---------------------------
# Thermo: density, Re_θ
# ---------------------------
function get_rho(U::AbstractVector{<:Real}, param)
    if param.Minf > 0
        M2, M2_U = get_Mach2(U, param)
        H0, gmi = param.H0, param.gam - 1
        den = 1 + 0.5*gmi*M2
        den_M2 = 0.5*gmi
        rho = param.rho0 / den^(1/gmi)
        rho_U = (-1/gmi) * rho/den * den_M2 .* M2_U
        return rho, rho_U
    else
        return param.rho0, zeros(4)
    end
end

function get_Ret(U::AbstractVector{<:Real}, param)
    if param.Minf > 0
        M2, M2_U = get_Mach2(U, param)
        uk, uk_u = get_uk(U[4], param)
        H0, gmi, Ts = param.H0, param.gam - 1, param.Tsrat
        Tr    = 1 - 0.5*uk^2/H0
        Tr_uk = -uk/H0
        f     = Tr^1.5 * (1 + Ts) / (Tr + Ts)      # Sutherland ratio
        f_Tr  = 1.5*f/Tr - f/(Tr + Ts)
        mu    = param.mu0 * f
        mu_uk = param.mu0 * f_Tr * Tr_uk
        den   = 1 + 0.5*gmi*M2
        den_M2 = 0.5*gmi
        rho   = param.rho0 / den^(1/gmi)
        rho_U = (-1/gmi) * rho/den * den_M2 .* M2_U

        Ret   = rho*uk*U[1]/mu
        Ret_U = rho_U*uk*U[1]/mu .+ (rho*U[1]/mu - Ret/mu*mu_uk) .* [0.0, 0.0, 0.0, uk_u] .+ (rho*uk/mu) .* [1.0, 0.0, 0.0, 0.0]
        return Ret, Ret_U
    else
        Ret   = param.rho0 * U[1]*U[4] / param.mu0
        Ret_U = [U[4], 0.0, 0.0, U[1]] / param.mu0
        return Ret, Ret_U
    end
end

# ---------------------------
# Skin friction & combos
# ---------------------------
function get_cf(U::AbstractVector{<:Real}, param)
    if param.wake
        return 0.0, zeros(4)
    end
    Hk,  Hk_U  = get_Hk(U, param)
    Ret, Ret_U = get_Ret(U, param)

    if param.turb
        M2, M2_U = get_Mach2(U, param)
        Fc   = sqrt(1 + 0.5*(param.gam - 1)*M2)
        Fc_U = (0.5/Fc) * 0.5*(param.gam - 1) .* M2_U
        aa   = -1.33*Hk
        aa_U = -1.33 .* Hk_U
        if aa < -17
            aa_new = -20 + 3*exp((aa + 17)/3)
            aa_U   = ((aa_new + 20)/3) .* aa_U
            aa     = aa_new
        end
        bb   = log(Ret/Fc)
        bb_U = Ret_U/Ret .- Fc_U/Fc
        if bb < 3
            bb   = 3.0
            bb_U .= 0.0
        end
        bb   /= log(10)
        bb_U ./= log(10)

        cc   = -1.74 - 0.31*Hk
        cc_U = -0.31 .* Hk_U
        dd   = tanh(4.0 - Hk/0.875)
        dd_U = (1 - dd^2) .* (-Hk_U/0.875)

        cf0   = 0.3*exp(aa) * bb^cc
        cf0_U = cf0 .* aa_U .+ 0.3*exp(aa) * cc * bb^(cc-1) .* bb_U .+ cf0*log(bb) .* cc_U
        cf    = (cf0 + 1.1e-4*(dd - 1)) / Fc
        cf_U  = (cf0_U .+ 1.1e-4 .* dd_U) ./ Fc .- (cf/Fc) .* Fc_U
        return cf, cf_U
    else
        if Hk < 5.5
            num    = 0.0727*(5.5 - Hk)^3/(Hk + 1) - 0.07
            num_Hk = 0.0727*(3*(5.5 - Hk)^2/(Hk + 1)*(-1) - (5.5 - Hk)^3/(Hk + 1)^2)
        else
            num    = 0.015*(1 - 1/(Hk - 4.5))^2 - 0.07
            num_Hk = 0.015*2*(1 - 1/(Hk - 4.5))/((Hk - 4.5)^2)
        end
        cf   = num/Ret
        cf_U = (num_Hk/Ret) .* Hk_U .- (num/Ret^2) .* Ret_U
        return cf, cf_U
    end
end

function get_cfxt(U::AbstractVector{<:Real}, x::Real, param)
    cf, cf_U = get_cf(U, param)
    th = U[1]
    cfxt   = cf*x/th
    cfxt_U = (cf_U .* x/th); cfxt_U[1] -= cfxt/th
    cfxt_x = cf/th
    return cfxt, cfxt_U, cfxt_x
end

function get_cfutstag(U::AbstractVector{<:Real}, param)
    U0 = collect(U); U0[4] = 0.0
    Hk, Hk_U = get_Hk(U0, param)
    if Hk < 5.5
        num    = 0.0727*(5.5 - Hk)^3/(Hk + 1) - 0.07
        num_Hk = 0.0727*(3*(5.5 - Hk)^2/(Hk + 1)*(-1) - (5.5 - Hk)^3/(Hk + 1)^2)
    else
        num    = 0.015*(1 - 1/(Hk - 4.5))^2 - 0.07
        num_Hk = 0.015*2*(1 - 1/(Hk - 4.5))/((Hk - 4.5)^2)
    end
    ν = param.mu0/param.rho0
    F   = ν*num
    F_U = ν*num_Hk .* Hk_U
    return F, F_U
end

function get_cdutstag(U::AbstractVector{<:Real}, param)
    U0 = collect(U); U0[4] = 0.0
    Hk, Hk_U = get_Hk(U0, param)
    if Hk < 4
        num    = 0.00205*(4 - Hk)^5.5 + 0.207
        num_Hk = 0.00205*5.5*(4 - Hk)^4.5*(-1)
    else
        Hk1    = Hk - 4
        num    = -0.0016*Hk1^2/(1 + 0.02*Hk1^2) + 0.207
        num_Hk = -0.0016*(2*Hk1/(1 + 0.02*Hk1^2) - (Hk1^2/(1 + 0.02*Hk1^2)^2)*0.02*2*Hk1)
    end
    ν = param.mu0/param.rho0
    D   = ν*num
    D_U = ν*num_Hk .* Hk_U
    return D, D_U
end

# ---------------------------
# Dissipation cDi and pieces
# ---------------------------
function get_cDi(U::AbstractVector{<:Real}, param)
    if param.turb
        cDi  = 0.0
        cDi_U = zeros(4)

        if !param.wake
            c0, c0_U = get_cDi_turbwall(U, param)
            cDi  += c0;  cDi_U .+= c0_U
            cDil, cDil_U = get_cDi_lam(U, param)       # for max check
        else
            cDil, cDil_U = get_cDi_lamwake(U, param)   # for max check
        end

        c1, c1_U = get_cDi_outer(U, param)
        cDi  += c1;  cDi_U .+= c1_U

        c2, c2_U = get_cDi_lamstress(U, param)
        cDi  += c2;  cDi_U .+= c2_U

        if cDil > cDi
            cDi, cDi_U = cDil, cDil_U
        end

        if param.wake
            cDi  *= 2
            cDi_U .*= 2
        end
        return cDi, cDi_U
    else
        return get_cDi_lam(U, param)
    end
end

function get_cDi_turbwall(U::AbstractVector{<:Real}, param)
    if param.wake
        return 0.0, zeros(4)
    end
    cf,  cf_U  = get_cf(U, param)
    Hk,  Hk_U  = get_Hk(U, param)
    Hs,  Hs_U  = get_Hs(U, param)
    Us,  Us_U  = get_Us(U, param)
    Ret, Ret_U = get_Ret(U, param)

    lr   = log(Ret); lr_U = Ret_U/Ret
    Hmin   = 1 + 2.1/lr
    Hmin_U = -2.1/lr^2 .* lr_U

    aa   = tanh((Hk - 1)/(Hmin - 1))
    fac  = 0.5 + 0.5*aa
    fac_U = 0.5*(1 - aa^2) .* ( Hk_U/(Hmin - 1) .- (Hk - 1)/(Hmin - 1)^2 .* Hmin_U )

    cDi   = 0.5*cf*Us*(2/Hs)*fac
    cDi_U = cf_U.*(Us/Hs*0.5*2) .+ cf.*(Us_U/Hs*0.5*2) .- cDi/Hs .* Hs_U .+ cf*Us*(2/Hs) .* fac_U .* 0.5
    return cDi, cDi_U
end

function get_cDi_lam(U::AbstractVector{<:Real}, param)
    Hk,  Hk_U  = get_Hk(U, param)
    Ret, Ret_U = get_Ret(U, param)
    if Hk < 4
        num    = 0.00205*(4 - Hk)^5.5 + 0.207
        num_Hk = 0.00205*5.5*(4 - Hk)^4.5*(-1)
    else
        Hk1    = Hk - 4
        num    = -0.0016*Hk1^2/(1 + 0.02*Hk1^2) + 0.207
        num_Hk = -0.0016*(2*Hk1/(1 + 0.02*Hk1^2) - (Hk1^2/(1 + 0.02*Hk1^2)^2)*0.02*2*Hk1)
    end
    cDi   = num/Ret
    cDi_U = (num_Hk/Ret) .* Hk_U .- (num/Ret^2) .* Ret_U
    return cDi, cDi_U
end

function get_cDi_lamwake(U::AbstractVector{<:Real}, param)
    p = deepcopy(param); p.turb = false
    Hk, Hk_U = get_Hk(U, p)
    Hs, Hs_U = get_Hs(U, p)
    Ret, Ret_U = get_Ret(U, p)
    HsRet   = Hs*Ret
    HsRet_U = Hs_U .* Ret .+ Hs .* Ret_U

    num    = 2*1.1*(1 - 1/Hk)^2*(1/Hk)
    num_Hk = 2*1.1*( 2*(1 - 1/Hk)*(1/Hk^2)*(1/Hk) + (1 - 1/Hk)^2*(-1/Hk^2) )
    cDi    = num / HsRet
    cDi_U  = (num_Hk .* Hk_U) ./ HsRet .- (num/HsRet^2) .* HsRet_U
    return cDi, cDi_U
end

function get_cDi_outer(U::AbstractVector{<:Real}, param)
    if !param.turb
        return 0.0, zeros(4)
    end
    Hs, Hs_U = get_Hs(U, param)
    Us, Us_U = get_Us(U, param)
    ct   = U[3]^2
    ct_U = [0.0, 0.0, 2U[3], 0.0]
    cDi   = ct*(0.995 - Us)*2/Hs
    cDi_U = ct_U*(0.995 - Us)*2/Hs .+ ct*(-Us_U)*2/Hs .- ct*(0.995 - Us)*2/Hs^2 .* Hs_U
    return cDi, cDi_U
end

function get_cDi_lamstress(U::AbstractVector{<:Real}, param)
    Hs, Hs_U = get_Hs(U, param)
    Us, Us_U = get_Us(U, param)
    Ret, Ret_U = get_Ret(U, param)
    HsRet   = Hs*Ret
    HsRet_U = Hs_U .* Ret .+ Hs .* Ret_U

    num    = 0.15*(0.995 - Us)^2*2
    num_Us = 0.15*2*(0.995 - Us)*(-1)*2
    cDi    = num / HsRet
    cDi_U  = (num_Us .* Us_U) ./ HsRet .- (num/HsRet^2) .* HsRet_U
    return cDi, cDi_U
end

# ---------------------------
# Slip velocity
# ---------------------------
function get_Us(U::AbstractVector{<:Real}, param)
    Hs, Hs_U = get_Hs(U, param)
    Hk, Hk_U = get_Hk(U, param)
    H,  H_U  = get_H(U)

    if param.wake && (Hk < 1.00005); Hk = 1.00005; Hk_U .= 0.0; end
    if !param.wake && (Hk < 1.05);   Hk = 1.05;    Hk_U .= 0.0; end

    β  = param.GB
    bi = 1/β
    Us   = 0.5*Hs*(1 - bi*(Hk - 1)/H)
    Us_U = 0.5 .* Hs_U .* (1 - bi*(Hk - 1)/H) .+
           0.5 .* Hs   .* ( -bi .* Hk_U ./ H .+ bi*(Hk - 1)/H^2 .* H_U )

    if !param.wake && (Us > 0.95)
        Us = 0.98; Us_U .= 0.0
    end
    if param.wake && (Us > 0.99995)
        Us = 0.99995; Us_U .= 0.0
    end
    return Us, Us_U
end

# ---------------------------
# Amplification rate (e^n)
# ---------------------------
function get_damp(U::AbstractVector{<:Real}, param)
    Hk, Hk_U   = get_Hk(U, param)
    Ret, Ret_U = get_Ret(U, param)
    th = U[1]

    if Hk < 1.05
        Hk = 1.05; Hk_U .= 0.0
    end

    Hmi   = 1/(Hk - 1)
    Hmi_U = -Hmi^2 .* Hk_U
    aa    = 2.492 * Hmi^0.43
    aa_U  = 0.43*aa/Hmi .* Hmi_U
    bb    = tanh(14Hmi - 9.24)
    bb_U  = (1 - bb^2) .* 14 .* Hmi_U
    lrc   = aa + 0.7*(bb + 1)
    lrc_U = aa_U .+ 0.7 .* bb_U

    lten = log(10)
    lr   = log(Ret)/lten
    lr_U = (1/Ret) .* Ret_U / lten

    dl = 0.1
    damp   = 0.0
    damp_U = zeros(4)

    if lr >= lrc - dl
        rn   = (lr - (lrc - dl)) / (2dl)
        rn_U = (lr_U .- lrc_U) / (2dl)
        if rn >= 1
            rf   = 1.0
            rf_U = zeros(4)
        else
            rf   = 3rn^2 - 2rn^3
            rf_U = (6rn - 6rn^2) .* rn_U
        end
        ar   = 3.87*Hmi - 2.52
        ar_U = 3.87 .* Hmi_U
        ex   = exp(-ar^2)
        ex_U = ex .* (-2ar) .* ar_U
        da   = 0.028*(Hk - 1) - 0.0345*ex
        da_U = 0.028 .* Hk_U .- 0.0345 .* ex_U
        af   = -0.05 + 2.7Hmi - 5.5Hmi^2 + 3Hmi^3 + 0.1*exp(-20Hmi)
        af_U = (2.7 - 11Hmi + 9Hmi^2 - exp(-20Hmi)*1.0) .* Hmi_U
        damp   = rf*af*da/th
        damp_U = (rf_U.*af.*da .+ rf.*af_U.*da .+ rf.*af.*da_U)/th .- (damp/th) .* [1.0, 0.0, 0.0, 0.0]
    end

    # extra amplification near ncrit
    ncrit = param.ncrit
    Cea = 5.0
    nx   = Cea*(U[3] - ncrit)
    nx_U = Cea .* [0.0, 0.0, 1.0, 0.0]
    eex   = 1 + tanh(nx)
    eex_U = (1 - tanh(nx)^2) .* nx_U

    ed   = eex * 0.001 / th
    ed_U = eex_U * 0.001 / th .- (ed/th) .* [1.0, 0.0, 0.0, 0.0]

    damp   += ed
    damp_U .+= ed_U
    return damp, damp_U
end


function calc_force!(M)
    chord = M.geom.chord
    xref  = M.geom.xref         # tuple (x,z)
    Vinf  = M.param.Vinf
    rho   = M.oper.rho
    α     = M.oper.alpha
    qinf  = 0.5*rho*Vinf^2
    N     = M.foil.N

    # edge speed and pressure coefficients on all stations available
    ue = M.oper.viscous ? vec(M.glob.U[4, :]) : get_ueinv(M)
    cp, cp_ue = get_cp(ue, M.param)
    M.post.cp  = cp[1:N]                      # store airfoil section cp only
    M.post.cpi = get_cp(get_ueinv(M), M.param)[1]  # inviscid cp

    # panel-wise pressure integration for cl, cm, cdpi
    cl = 0.0
    cl_ue = zeros(Float64, N)
    cl_alpha = 0.0
    cm = 0.0
    cdpi = 0.0

    for i0 in 2:N+1
        i  = (i0 <= N) ? i0 : 1
        ip = (i == 1) ? N : (i-1)

        x1 = M.foil.x[:, ip]
        x2 = M.foil.x[:, i]
        dxv = x2 - x1
        dx1 = x1 .- collect(xref)
        dx2 = x2 .- collect(xref)

        # (x - xref) ⋅ (n ds) algebra reduces to these dot combos
        dx1nds = dxv[1]*dx1[1] + dxv[2]*dx1[2]
        dx2nds = dxv[1]*dx2[1] + dxv[2]*dx2[2]

        # components in lift/drag directions (node ordering is CW)
        dx = -dxv[1]*cosd(α) - dxv[2]*sind(α)
        dz =  dxv[2]*cosd(α) - dxv[1]*sind(α)

        cp1 = cp[ip]; cp2 = cp[i]; cpbar = 0.5*(cp1 + cp2)

        cl += dx * cpbar
        idx = (ip, i)
        cl_ue[[ip, i]] .+= dx * 0.5 .* cp_ue[[ip, i]]

        cl_alpha += cpbar * (sind(α)*dxv[1] - cosd(α)*dxv[2]) * (pi/180)

        cm   += cp1*dx1nds/3 + cp1*dx2nds/6 + cp2*dx1nds/6 + cp2*dx2nds/3
        cdpi += dz * cpbar
    end

    M.post.cl       = cl/chord
    M.post.cl_ue    = cl_ue
    M.post.cl_alpha = cl_alpha
    M.post.cm       = cm/chord^2
    M.post.cdpi     = cdpi/chord

    # --- viscous contributions ---
    cd  = 0.0
    cdf = 0.0
    if M.oper.viscous
        # Squire–Young extrapolation at end of wake
        iw = M.vsol.Is[3][end]                  # last wake station
        U  = M.glob.U[:, iw]
        H  = U[2]/U[1]
        uk, _ = get_uk(U[4], M.param)
        cd = 2.0 * U[1] * (uk/Vinf)^((5 + H)/2)

        # skin-friction drag by trapezoidal marching on each surface
        Df = 0.0
        for is in 1:2
            Is = M.vsol.Is[is]
            param = build_param(M, is)
            param = station_param(M, param, Is[1])

            cf1 = 0.0
            ue1 = 0.0
            rho1 = rho
            x1 = collect(M.isol.xstag)

            for k in eachindex(Is)
                i = Is[k]
                param = station_param(M, param, i)

                cf2, _ = get_cf(M.glob.U[:, i], param)
                ue2, _ = get_uk(M.glob.U[4, i], param)
                rho2, _ = get_rho(M.glob.U[:, i], param)

                x2 = M.foil.x[:, i]
                dxv = x2 - x1
                dx = dxv[1]*cosd(α) + dxv[2]*sind(α)

                Df += 0.25 * (rho1*cf1*ue1^2 + rho2*cf2*ue2^2) * dx

                cf1 = cf2; ue1 = ue2; rho1 = rho2; x1 = x2
            end
        end
        cdf = Df / (qinf*chord)
    end

    M.post.cd  = cd
    M.post.cdf = cdf
    M.post.cdp = cd - cdf

    vprint(M.param, 1,
    "  alpha=%.2fdeg, cl=%.6f, cm=%.6f, cdpi=%.6f, cd=%.6f, cdf=%.6f, cdp=%.6f\n",
    M.oper.alpha, M.post.cl, M.post.cm, M.post.cdpi, M.post.cd, M.post.cdf, M.post.cdp)
    return nothing
end 


function solve_inviscid!(M)
    @assert M.foil.N>0 "No panels"
    M.oper.viscous = false
    init_thermo!(M)
    M.isol.sgnue = ones(Int,M.foil.N)
    build_gamma!(M, M.oper.alpha)
    if (M.oper.givencl)
         cltrim_inviscid!(M)
    end
    calc_force!(M)
    M.glob.conv = true
end

function sanity_inviscid(M)
    airfoil = Airfoil("NACA2412")
    prob = InviscidProblem(airfoil, 0)
    sol = solve(prob)
    @assert !isempty(M.isol.gam) "No inviscid solution stored"
    γ = M.isol.gam
    kutta_res = abs(γ[1] + γ[end])

    cl_thin = sol.cl
    cl_num  = M.post.cl
    rel_err = (cl_num - cl_thin) / (abs(cl_thin) > 0 ? cl_thin : 1)

    @printf "Kutta residual |γ1+γN| = %.3e\n" kutta_res
    @printf "c_l (num) = %.5f,  c_l my solver = %.5f,  rel err = %.2f%%\n" cl_num cl_thin 100*rel_err
    return nothing
end

M = init_M(5)
enforce_CW_and_TE!(M)
solve_inviscid!(M)
sanity_inviscid(M)