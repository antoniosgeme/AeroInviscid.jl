using AeroGeometry
using LinearAlgebra
using Printf
using Printf: format, Format
using SparseArrays


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
    snaca = "0012"
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

function upwind(upw, upw_U,
                f1, f1_U1,
                f2, f2_U2)
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

function cltrim_inviscid!(M)
    for i in 1:15
        α = M.oper.alpha
        calc_force!(M)                          # updates M.post.cl, cl_alpha, cl_ue, etc.
        R = M.post.cl - M.oper.cltgt
        if abs(R) < 1e-10
            break
        end

        # d/dα of [cosd α; sind α] = [-sind α; cosd α] * π/180
        sc = [-sind(α); cosd(α)] * (π/180)

        # total derivative: dcl/dα = cl_α + (∂cl/∂ue)^T * (d ue/dα)
        # ue(α) on the airfoil nodes is (gamref * [cosd α; sind α]) with the same sign
        # As in your MATLAB, we use M.isol.gamref directly:
        cl_a = M.post.cl_alpha + dot(M.post.cl_ue, M.isol.gamref * sc)

        dalpha = -R / cl_a
        M.oper.alpha = α + clamp(dalpha, -2.0, 2.0)
    end

    # final update of γ(α)
    M.isol.gam = M.isol.gamref * [cosd(M.oper.alpha); sind(M.oper.alpha)]
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

function signed_area(X)
    N = size(X,2)
    s = 0.0
    for i in 1:N
        j = i == N ? 1 : i+1
        s += X[1,i]*X[2,j] - X[1,j]*X[2,i]
    end
    return 0.5*s
end

function sanity_inviscid(M)
    airfoil = Airfoil("NACA2412")
    prob = InviscidProblem(airfoil, M.oper.alpha)
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

"""
    space_geom(dx0, L, Np) -> Vector{Float64}

Geometrically spaces `Np` points on [0, L], where the first interval is `dx0`.
Returns a vector `[x₀, x₁, …, x_{Np-1}]` with `x₀=0` and `x_{end}=L`.
"""
function space_geom(dx0::Real, L::Real, Np::Integer)
    @assert Np > 1 "Need at least two points for spacing."
    N = Np - 1
    N == 1 && return Float64[0.0, float(L)]

    d = L / dx0
    # Initial guess for r using the same cubic-based idea as the MATLAB code
    a = N*(N-1)*(N-2)/6
    b = N*(N-1)/2
    c = N - d
    r = 1.0
    if a != 0
        disc = max(b*b - 4*a*c, 0.0)
        r = 1 + (-b + sqrt(disc)) / (2a)
    else
        # Fallback for small N where a==0 (N ≤ 2)
        if N > 1
            r = 1 + 2*(d - N)/(N*(N - 1))  # linearized sum guess near r≈1
            if !isfinite(r) || r <= 0
                r = 1.0
            end
        end
    end

    # Newton iterations to solve: r^N - 1 = d*(r - 1)
    for _ in 1:20
        R  = r^N - 1 - d*(r - 1)
        Rp = N*r^(N-1) - d
        if Rp == 0
            break
        end
        dr = -R / Rp
        r += dr
        if abs(dr) < 1e-12
            break
        end
    end
    if !isfinite(r) || r <= 0
        r = 1.0  # fallback to uniform if Newton went weird
    end

    x = zeros(Float64, Np)
    x[1] = 0.0
    if abs(r - 1) < 1e-12
        # uniform spacing
        dx = L / N
        @inbounds for i in 2:Np
            x[i] = (i - 1) * dx
        end
    else
        # geometric spacing: Δx_k = dx0 * r^(k-1)
        s = 0.0
        @inbounds for k in 1:N
            s += dx0 * r^(k - 1)
            x[k + 1] = s
        end
        x[end] = float(L)  # enforce exact endpoint
    end

    return x
end


function panel_linvortex_velocity(
    Xj::AbstractMatrix{<:Real},
    xi::AbstractVector{<:Real},
    vdir::Union{Nothing,AbstractVector{<:Real}} = nothing,
    onmid::Bool = false,
)
    @assert size(Xj,1) == 2 && size(Xj,2) == 2 "Xj must be 2×2"
    @assert length(xi) == 2 "xi must be length-2"
    if vdir !== nothing
        @assert length(vdir) == 2 "vdir must be length-2"
    end

    # panel endpoints
    xj1, zj1 = Xj[1,1], Xj[2,1]
    xj2, zj2 = Xj[1,2], Xj[2,2]

    # tangent/normal (panel-aligned basis)
    tx, tz = xj2 - xj1, zj2 - zj1
    d = hypot(tx, tz)
    @assert d > 0 "panel has zero length"
    tx, tz = tx/d, tz/d
    nx, nz = -tz, tx

    # control point in panel-aligned coordinates
    xrel, zrel = xi[1] - xj1, xi[2] - zj1
    x = xrel*tx + zrel*tz       # along-tangent
    z = xrel*nx + zrel*nz       # along-normal

    # distances/angles
    r1 = hypot(x, z)            # from left edge
    r2 = hypot(x - d, z)        # from right edge
    θ1 = atan(z, x)             # atan2(z, x)
    θ2 = atan(z, x - d)

    # velocity in panel frame due to unit end strengths
    if onmid
        ug1 = 1/2 - 1/4;  ug2 = 1/4
        wg1 = -1/(2π);    wg2 = 1/(2π)
    else
        temp1 = (θ2 - θ1) / (2π)
        temp2 = (2*z*log(r1/r2) - 2*x*(θ2 - θ1)) / (4π*d)
        ug1 =  temp1 + temp2
        ug2 =       - temp2

        temp1 =  log(r2/r1) / (2π)
        temp2 = (x*log(r1/r2) - d + z*(θ2 - θ1)) / (2π*d)
        wg1 =  temp1 + temp2
        wg2 =       - temp2
    end

    # rotate back to global (x,z)
    ax, az = ug1*tx + wg1*nx, ug1*tz + wg1*nz
    bx, bz = ug2*tx + wg2*nx, ug2*tz + wg2*nz

    if vdir === nothing
        return (Float64[ax, az], Float64[bx, bz])
    else
        return (ax*vdir[1] + az*vdir[2],
                bx*vdir[1] + bz*vdir[2])
    end
end

function panel_constsource_velocity(
    Xj::AbstractMatrix{<:Real},
    xi::AbstractVector{<:Real},
    vdir::Union{Nothing,AbstractVector{<:Real}} = nothing,
)
    @assert size(Xj,1) == 2 && size(Xj,2) == 2 "Xj must be 2×2"
    @assert length(xi) == 2 "xi must be length-2"
    if vdir !== nothing
        @assert length(vdir) == 2 "vdir must be length-2"
    end

    # panel endpoints
    xj1, zj1 = Xj[1,1], Xj[2,1]
    xj2, zj2 = Xj[1,2], Xj[2,2]

    # tangent/normal (panel-aligned basis)
    tx, tz = xj2 - xj1, zj2 - zj1
    d = hypot(tx, tz)
    @assert d > 0 "panel has zero length"
    tx, tz = tx/d, tz/d
    nx, nz = -tz, tx

    # control point in panel-aligned coordinates
    xrel, zrel = xi[1] - xj1, xi[2] - zj1
    x = xrel*tx + zrel*tz       # along-tangent
    z = xrel*nx + zrel*nz       # along-normal

    # distances/angles
    r1 = hypot(x, z)            # from left edge
    r2 = hypot(x - d, z)        # from right edge
    θ1 = atan(z, x)
    θ2 = atan(z, x - d)

    # tiny-distance guards (mirror MATLAB behavior)
    ep = 1e-9
    if r1 < ep
        logr1 = 0.0
        θ1 = π
        θ2 = π
    else
        logr1 = log(r1)
    end
    if r2 < ep
        logr2 = 0.0
        θ1 = 0.0
        θ2 = 0.0
    else
        logr2 = log(r2)
    end

    # velocity in panel frame
    u = (0.5/π) * (logr1 - logr2)
    w = (0.5/π) * (θ2 - θ1)

    # rotate back to global
    ax = u*tx + w*nx
    az = u*tz + w*nz

    if vdir === nothing
        return Float64[ax, az]
    else
        return ax*vdir[1] + az*vdir[2]
    end
end


function inviscid_velocity(
    X::AbstractMatrix{<:Real},
    G::AbstractVector{<:Real},
    Vinf::Real,
    alpha::Real,
    x::AbstractVector{<:Real};
    return_jacobian::Bool = false,
)
    @assert size(X,1) == 2 "X must be 2×N (rows: x,z; cols: nodes)"
    @assert length(G) == size(X,2) "G must have one value per node (length N)"
    @assert length(x) == 2 "x must be a 2-vector"

    N = size(X,2)
    V   = zeros(Float64, 2)
    V_G = return_jacobian ? zeros(Float64, 2, N) : nothing

    # TE info
    _, _, _, tcp, tdp = TE_info(X)

    # panel contributions (airfoil panels 1..N-1)
    for j in 1:(N-1)
        a, b = panel_linvortex_velocity(X[:, [j, j+1]], x, nothing, false)
        # accumulate velocity
        @inbounds V .+= a .* G[j] .+ b .* G[j+1]
        if return_jacobian
            @inbounds V_G[:, j]   .+= a
            @inbounds V_G[:, j+1] .+= b
        end
    end

    # TE source panel (between node N -> 1)
    a = panel_constsource_velocity(X[:, [N, 1]], x, nothing)
    f1 = a .* (-0.5*tcp)
    f2 = a .* ( 0.5*tcp)
    @inbounds V .+= f1 .* G[1] .+ f2 .* G[N]
    if return_jacobian
        @inbounds V_G[:, 1] .+= f1
        @inbounds V_G[:, N] .+= f2
    end

    # TE vortex panel (between node N -> 1)
    aV, bV = panel_linvortex_velocity(X[:, [N, 1]], x, nothing, false)
    f1 = (aV .+ bV) .* ( 0.5*tdp)
    f2 = (aV .+ bV) .* (-0.5*tdp)
    @inbounds V .+= f1 .* G[1] .+ f2 .* G[N]
    if return_jacobian
        @inbounds V_G[:, 1] .+= f1
        @inbounds V_G[:, N] .+= f2
    end

    # freestream contribution
    cα = cos(alpha * π / 180)
    sα = sin(alpha * π / 180)
    V .+= Vinf .* [cα, sα]

    return return_jacobian ? (V, V_G) : V
end

function build_wake!(M)
    @assert !isempty(M.isol.gam) "No inviscid solution"

    N    = M.foil.N
    Vinf = M.oper.Vinf

    # Number of wake points (same heuristic as MATLAB)
    Nw = ceil(Int, N/10 + 10*M.geom.wakelen)

    # Ensure we have foil arclengths S
    S = M.foil.s
    if length(S) != N
        # fallback: compute cumulative arclength along foil nodes
        S = zeros(Float64, N)
        for i in 2:N
            S[i] = S[i-1] + norm(M.foil.x[:, i] .- M.foil.x[:, i-1])
        end
    end

    # first nominal wake panel size (average of first & last airfoil panel)
    ds1 = 0.5 * ((S[2]-S[1]) + (S[end]-S[end-1]))

    # geometrically spaced distances along wake (0..L_wake)
    sv = space_geom(ds1, M.geom.wakelen * M.geom.chord, Nw)  # length Nw

    xyw = zeros(Float64, 2, Nw)  # wake points
    tw  = similar(xyw)           # wake tangents

    # TE midpoint & wake initial direction
    xy1 = M.foil.x[:, 1]
    xyN = M.foil.x[:, N]
    xyte = 0.5 .* (xy1 .+ xyN)
    n = xyN .- xy1
    t = [n[2]; -n[1]]                 # tangent leaving TE toward +x if CCW
    @assert t[1] > 0 "Wrong wake direction; ensure airfoil points are CCW"

    # first wake point—nudge just behind TE
    xyw[:, 1] = xyte .+ 1e-5 * M.geom.chord .* (t / norm(t))

    # s-values for wake are continuation of foil s
    sw = S[end] .+ sv

    # March the wake with predictor–corrector
    for i in 1:(Nw-1)
        # predictor
        v1 = inviscid_velocity(M.foil.x, M.isol.gam, Vinf, M.oper.alpha, xyw[:, i])
        v1 ./= norm(v1);   tw[:, i] = v1
        xyw[:, i+1] = xyw[:, i] .+ (sv[i+1] - sv[i]) .* v1

        # corrector
        v2 = inviscid_velocity(M.foil.x, M.isol.gam, Vinf, M.oper.alpha, xyw[:, i+1])
        v2 ./= norm(v2);   tw[:, i+1] = v2
        xyw[:, i+1] = xyw[:, i] .+ (sv[i+1] - sv[i]) .* 0.5 .* (v1 .+ v2)
    end

    # Inviscid UE along wake and reference (0°, 90°)
    uewi     = zeros(Float64, Nw)
    uewiref  = zeros(Float64, Nw, 2)
    for i in 1:Nw
        v = inviscid_velocity(M.foil.x, M.isol.gam,            Vinf, M.oper.alpha, xyw[:, i]);  uewi[i]      = dot(v, tw[:, i])
        v = inviscid_velocity(M.foil.x, M.isol.gamref[:, 1],   Vinf, 0.0,          xyw[:, i]);  uewiref[i,1] = dot(v, tw[:, i])
        v = inviscid_velocity(M.foil.x, M.isol.gamref[:, 2],   Vinf, 90.0,         xyw[:, i]);  uewiref[i,2] = dot(v, tw[:, i])
    end

    # Write back to M
    M.wake.N = Nw
    M.wake.x = xyw
    M.wake.s = sw
    M.wake.t = tw
    M.isol.uewi    = uewi
    M.isol.uewiref  = uewiref

    return nothing
end


function stagpoint_find!(M)
    @assert !isempty(M.isol.gam) "No inviscid solution"

    N = M.foil.N
    γ = M.isol.gam
    S = M.foil.s

    # first index where gamma > 0 (upper surface region for CW ordering)
    J = findall(g -> g > 0, γ)
    @assert !isempty(J) "no stagnation point"
    j  = first(J)
    i1 = (j == 1) ? N : j - 1
    i2 = j

    G = (γ[i1], γ[i2])
    Ss = (S[i1], S[i2])
    den = G[2] - G[1]
    @assert abs(den) > eps() "degenerate gamma difference at stagnation"

    # linear interpolation weights in s using zero crossing of gamma
    w1 = G[2] / den
    w2 = -G[1] / den
    sst = w1*Ss[1] + w2*Ss[2]

    # stash indices and s location
    M.isol.Istag  = (i1, i2)
    M.isol.sstag  = sst

    # x-location (linear blend of the two neighboring nodes)
    xstag_vec = M.foil.x[:, [i1, i2]] * [w1; w2]
    M.isol.xstag = (xstag_vec[1], xstag_vec[2])

    # derivative of sstag w.r.t. the two gammas
    st_g1 = G[2] * (Ss[1] - Ss[2]) / (den^2)
    M.isol.sstag_g = (st_g1, -st_g1)

    # sign convention for ue on airfoil nodes
    sgnue = fill(-1, N)
    for k in J
        sgnue[k] = 1
    end
    M.isol.sgnue = sgnue

    # distance-from-stagnation coordinate (airfoil + wake continuation)
    xi_air  = abs.(S .- sst)
    xi_wake = (M.wake.N > 0) ? (M.wake.s .- sst) : Float64[]
    M.isol.xi = vcat(xi_air, xi_wake)

    return nothing
end

function identify_surfaces!(M)
    @assert M.isol.Istag != (0, 0) "Stagnation point not set. Call `stagpoint_find!` first."

    iL, iU = M.isol.Istag
    N  = M.foil.N
    Nw = M.wake.N

    Is_lower = collect(iL:-1:1)
    Is_upper = collect(iU:N)
    Is_wake  = Nw > 0 ? collect(N+1:N+Nw) : Int[]

    M.vsol.Is = [Is_lower, Is_upper, Is_wake]
    return nothing
end

function set_wake_gap!(M)
    # TE geometry info
    _, hTE, dtdx, _, _ = TE_info(M.foil.x)

    # clip TE thickness slope
    flen = 2.5
    dtdx = clamp(dtdx, -3.0/flen, 3.0/flen)
    Lw = flen * hTE

    Nw = M.wake.N
    wgap = zeros(Float64, Nw)

    # nothing to do if no wake or no TE gap
    if Nw == 0 || abs(Lw) <= eps(Float64)
        M.vsol.wgap = wgap
        return nothing
    end

    # xi distances (airfoil then wake); TE is at index M.foil.N
    ξ = M.isol.xi
    iTE = M.foil.N

    for i in 1:Nw
        xib = (ξ[iTE + i] - ξ[iTE]) / Lw
        if xib <= 1.0
            wgap[i] = hTE * (1 + (2 + flen*dtdx)*xib) * (1 - xib)^2
        else
            wgap[i] = 0.0
        end
    end

    M.vsol.wgap = wgap
    return nothing
end

function calc_ue_m!(M)
    @assert !isempty(M.isol.gam) "No inviscid solution"
    N  = M.foil.N
    Nw = M.wake.N
    @assert Nw > 0 "No wake has been constructed"

    # -------------------------
    # Cgam = d(ue_wake)/d(gamma)  [Nw × N]
    # -------------------------
    Cgam = zeros(Float64, Nw, N)
    for i in 1:Nw
        # sensitivities of induced velocity at wake point i
        _, V_G = inviscid_velocity(M.foil.x, M.isol.gam, 0.0, 0.0, M.wake.x[:, i]; return_jacobian=true)
        ti = M.wake.t[:, i]
        # project the 2D velocity sensitivity onto the local tangent
        Cgam[i, :] = V_G[1, :].*ti[1] .+ V_G[2, :].*ti[2]
    end

    # ----------------------------------------------
    # B = d(psi_surface)/d(source)  [(N+1) × (N+Nw-2)]
    # (airfoil panels: constant source; wake: piecewise-linear)
    # ----------------------------------------------
    nsrc = N + Nw - 2  # number of panels (N-1 airfoil + Nw-1 wake)
    B = zeros(Float64, N+1, nsrc)

    for i in 1:N
        xi = M.foil.x[:, i]

        # airfoil panels (constant source)
        for j in 1:N-1
            B[i, j] = panel_constsource_stream(M.foil.x[:, [j, j+1]], xi)
        end

        # wake panels (two-piece linear source per panel)
        for j in 1:Nw-1
            # build "left / mid / right" points for wake panel j
            Xj = M.wake.x[:, [j, j+1]]
            Xm = 0.5 .* (Xj[:, 1] .+ Xj[:, 2])
            X3 = similar(M.wake.x, 2, 3)
            X3[:, 1] = Xj[:, 1]   # left
            X3[:, 2] = Xm         # mid
            X3[:, 3] = Xj[:, 2]   # right
            if j == Nw-1
                # ghost extension for the last right half
                X3[:, 3] .= 2 .* X3[:, 3] .- X3[:, 2]
            end

            # left half (linear)
            a, b = panel_linsource_stream(X3[:, [1, 2]], xi)
            if j > 1
                B[i, N-1 + j    ] += 0.5*a + b
                B[i, N-1 + j - 1] += 0.5*a
            else
                B[i, N-1 + j    ] += b
            end

            # right half (linear)
            a, b = panel_linsource_stream(X3[:, [2, 3]], xi)
            B[i, N-1 + j] += a + 0.5*b
            if j < Nw-1
                B[i, N-1 + j + 1] += 0.5*b
            else
                B[i, N-1 + j]    += 0.5*b
            end
        end
    end

    # ----------------------------------------------
    # Bp = -AIC^{-1} * B  =>  d(gamma)/d(source)  [N × nsrc]
    # (last row of the solve is the surface psi DOF—drop it)
    # ----------------------------------------------
    Bp_full = -(M.isol.AIC \ B)
    Bp = Bp_full[1:end-1, :]

    # ----------------------------------------------
    # Csig = d(ue_wake)/d(source)  [Nw × nsrc]
    # ----------------------------------------------
    Csig = zeros(Float64, Nw, nsrc)

    for i in 1:Nw
        xi = M.wake.x[:, i]
        ti = M.wake.t[:, i]

        # airfoil panels: avoid the two panels that meet at TE for i=1
        jstart = (i == 1) ? 2 : 1
        jend   = (i == 1) ? (N-2) : (N-1)
        for j in jstart:jend
            Csig[i, j] = panel_constsource_velocity(M.foil.x[:, [j, j+1]], xi, ti)
        end

        # wake panels: piecewise linear (split into halves around each wake node)
        for j in 1:Nw
            I1 = max(j-1, 1)
            I2 = j
            I3 = min(j+1, Nw)
            Xj = M.wake.x[:, [I1, I2, I3]]
            # convert to (left-midpoint, node, right-midpoint)
            Xj[:, 1] .= 0.5 .* (Xj[:, 1] .+ Xj[:, 2])
            Xj[:, 3] .= 0.5 .* (Xj[:, 2] .+ Xj[:, 3])
            if j == Nw
                # ghost extension on last
                Xj[:, 3] .= 2 .* Xj[:, 2] .- Xj[:, 1]
            end

            d1 = hypot(Xj[1,2]-Xj[1,1], Xj[2,2]-Xj[2,1])
            d2 = hypot(Xj[1,3]-Xj[1,2], Xj[2,3]-Xj[2,2])

            if i == j
                # singular/self terms: use the closed-form limits
                if j == 1
                    dl = norm(M.foil.x[:, 2] .- M.foil.x[:, 1])     # lower surface panel length
                    du = norm(M.foil.x[:, N] .- M.foil.x[:, N-1])   # upper surface panel length
                    Csig[i, 1      ] += (0.5/π) * (log(dl/d2) + 1.0)
                    Csig[i, N-1    ] += (0.5/π) * (log(du/d2) + 1.0)
                    Csig[i, N-1 + 1] += -(0.5/π)
                elseif j == Nw
                    # last point has no self-effect (ghost extension)
                    Csig[i, N-1 + j - 1] += 0.0
                else
                    aa = (0.25/π) * log(d1/d2)
                    Csig[i, N-1 + j - 1] += aa + 0.5/π
                    Csig[i, N-1 + j    ] += aa - 0.5/π
                end
            else
                # regular off-diagonal wake contributions
                if j == 1
                    a, b = panel_linsource_velocity(Xj[:, [2, 3]], xi, ti)
                    Csig[i, N-1 + 1] += b             # right half of first wake panel
                    Csig[i, 1      ] += a             # lower airfoil panel
                    Csig[i, N-1    ] += a             # upper airfoil panel
                elseif j == Nw
                    a = panel_constsource_velocity(Xj[:, [1, 3]], xi, ti) # ghost full panel
                    Csig[i, N + Nw - 2] += a
                else
                    a1, b1 = panel_linsource_velocity(Xj[:, [1, 2]], xi, ti) # left half
                    a2, b2 = panel_linsource_velocity(Xj[:, [2, 3]], xi, ti) # right half
                    Csig[i, N-1 + j - 1] += a1 + 0.5*b1
                    Csig[i, N-1 + j    ] += 0.5*a2 + b2
                end
            end
        end
    end

    # ----------------------------------------------
    # Combine to get ue_sigma at airfoil+wake nodes
    # Dw = d(ue_wake)/d(source) = Cgam*Bp + Csig
    # enforce ue continuity at first wake node
    # ----------------------------------------------
    Dw = Cgam * Bp + Csig
    Dw[1, :] .= Bp[end, :]

    M.vsol.ue_sigma = vcat(Bp, Dw)

    # Use sgnue to convert signed ue_sigma → unsigned ue_m, and set sigma_m.
    # (We’ll implement this next in `rebuild_ue_m!`.)
    rebuild_ue_m!(M)

    return nothing
end

function rebuild_ue_m!(M::MfoilTypes.Mfoil)
    @assert size(M.vsol.ue_sigma, 1) > 0 "Need ue_sigma to build ue_m"

    N  = M.foil.N
    Nw = M.wake.N

    # Dp = d(source)/d(mass)  [(N+Nw-2) x (N+Nw)]
    Dp = zeros(Float64, N + Nw - 2, N + Nw)

    # Airfoil constant-source panels: rows 1..(N-1)
    for i in 1:(N-1)
        ds = M.foil.s[i+1] - M.foil.s[i]
        @assert ds != 0.0 "Zero panel spacing on foil (s[i+1] == s[i])"
        # elementwise multiply by ±1 sign on each endpoint
        Dp[i, i]   =  M.isol.sgnue[i]   * (-1.0 / ds)
        Dp[i, i+1] =  M.isol.sgnue[i+1] * ( 1.0 / ds)
    end

    # Wake two-piece-linear sources: rows N..(N+Nw-2)
    for i in 1:max(Nw-1, 0)
        ds = M.wake.s[i+1] - M.wake.s[i]
        @assert ds != 0.0 "Zero panel spacing in wake (s[i+1] == s[i])"
        row = (N - 1) + i
        Dp[row, N + i]     = -1.0 / ds
        Dp[row, N + i + 1] =  1.0 / ds
    end

    M.vsol.sigma_m = Dp

    # sign of ue at all points (airfoil sign pattern + +1 on wake)
    sgue = vcat(Float64.(M.isol.sgnue), ones(Float64, Nw))        # (N+Nw)
    sgue_col = reshape(sgue, :, 1)                                # (N+Nw)×1

    # ue_m = diag(sgue) * ue_sigma * sigma_m
    # Do the left diag-mult via broadcasting instead of building a diagonal matrix
    UEscaled = M.vsol.ue_sigma .* sgue_col                        # (N+Nw)×(N+Nw-2)
    M.vsol.ue_m = UEscaled * M.vsol.sigma_m                       # (N+Nw)×(N+Nw)

    return nothing
end

function panel_linsource_stream(Xj::AbstractMatrix{<:Real},
                                xi::AbstractVector{<:Real})
    # panel endpoints
    xj1, zj1 = Xj[1,1], Xj[2,1]
    xj2, zj2 = Xj[1,2], Xj[2,2]

    # panel-aligned basis
    t = [xj2 - xj1, zj2 - zj1]
    d = sqrt(t[1]^2 + t[2]^2)
    t ./= d
    n = [-t[2], t[1]]

    # control point in panel coords
    xz = [xi[1] - xj1, xi[2] - zj1]
    x  = xz[1]*t[1] + xz[2]*t[2]
    z  = xz[1]*n[1] + xz[2]*n[2]

    # distances & angles
    r1 = hypot(x, z)
    r2 = hypot(x - d, z)
    θ1 = atan(z, x)
    θ2 = atan(z, x - d)

    # branch cut at θ=0
    if θ1 < 0; θ1 += 2π; end
    if θ2 < 0; θ2 += 2π; end

    # near-singularity handling
    ep = 1e-9
    if r1 < ep
        logr1 = 0.0
        θ1 = π; θ2 = π
    else
        logr1 = log(r1)
    end
    if r2 < ep
        logr2 = 0.0
        θ1 = 0.0; θ2 = 0.0
    else
        logr2 = log(r2)
    end

    # streamfunction pieces
    P1 = (0.5/π) * ( x*(θ1 - θ2) + θ2*d + z*logr1 - z*logr2 )
    P2 = x*P1 + (0.5/π) * ( 0.5*r2^2*θ2 - 0.5*r1^2*θ1 - 0.5*z*d )

    # influence coeffs for endpoint source strengths (s1,s2)
    a = P1 - P2/d
    b = P2/d
    return a, b
end

# Linear source panel — velocity influence (a,b)
# If vdir === nothing -> return 2-vectors; else return scalars dotted with vdir
function panel_linsource_velocity(Xj::AbstractMatrix{<:Real},
                                  xi::AbstractVector{<:Real},
                                  vdir::Union{AbstractVector{<:Real},Nothing})
    # panel endpoints
    xj1, zj1 = Xj[1,1], Xj[2,1]
    xj2, zj2 = Xj[1,2], Xj[2,2]

    # panel-aligned basis
    t = [xj2 - xj1, zj2 - zj1]
    d = sqrt(t[1]^2 + t[2]^2)
    t ./= d
    n = [-t[2], t[1]]

    # control point in panel coords
    xz = [xi[1] - xj1, xi[2] - zj1]
    x  = xz[1]*t[1] + xz[2]*t[2]
    z  = xz[1]*n[1] + xz[2]*n[2]

    # distances & angles
    r1 = hypot(x, z)
    r2 = hypot(x - d, z)
    θ1 = atan(z, x)
    θ2 = atan(z, x - d)

    # velocity components in panel coords (per endpoint source)
    tmp1 = log(r1/r2)/(2π)
    tmp2 = (x*log(r1/r2) - d + z*(θ2 - θ1)) / (2π*d)
    ug1 =  tmp1 - tmp2
    ug2 =           tmp2

    tmp1 = (θ2 - θ1)/(2π)
    tmp2 = (-z*log(r1/r2) + x*(θ2 - θ1)) / (2π*d)
    wg1 =  tmp1 - tmp2
    wg2 =           tmp2

    # transform back to global coords
    a_vec = [ug1*t[1] + wg1*n[1], ug1*t[2] + wg1*n[2]]
    b_vec = [ug2*t[1] + wg2*n[1], ug2*t[2] + wg2*n[2]]

    if vdir === nothing
        return a_vec, b_vec
    else
        return dot(a_vec, vdir), dot(b_vec, vdir)
    end
end


function thwaites_init(K::Real, nu::Real)
    @assert K > 0 "K (stag-point constant) must be positive."
    th = sqrt(0.45 * nu / (6 * K))  # momentum thickness θ
    ds = 2.2 * th                    # displacement thickness δ*
    return th, ds
end

function residual_station(param, x::AbstractVector{<:Real}, U::AbstractMatrix{<:Real}, Aux::AbstractMatrix{<:Real})
    # Copy U and remove wake gap from δ* entry for ALL calcs below
    Uloc = copy(U)
    Uloc[2, :] .-= Aux[1, :]

    # split states
    U1 = Uloc[:, 1]
    U2 = Uloc[:, 2]
    Um = 0.5 .* (U1 .+ U2)

    th = Uloc[1, :]
    ds = Uloc[2, :]
    sa = Uloc[3, :]

    # compressibility-corrected speeds (scalar → scalar derivative)
    uk1, uk1_u = get_uk(U1[4], param)
    uk2, uk2_u = get_uk(U2[4], param)

    # logs
    thlog    = log(th[2] / th[1])
    thlog_U  = [-1/th[1], 0, 0, 0,  1/th[2], 0, 0, 0]

    uelog    = log(uk2 / uk1)
    uelog_U  = [0, 0, 0, -uk1_u/uk1,  0, 0, 0, uk2_u/uk2]

    xlog     = log(x[2] / x[1])
    xlog_x   = [-1/x[1], 1/x[2]]

    dx       = x[2] - x[1]
    dx_x     = [-1.0, 1.0]

    # upwind factor (scalar and 1×8 derivative)
    upw, upw_U = get_upw(U1, U2, param)

    # shape parameter H
    H1, H1_U1 = get_H(Uloc[:, 1])
    H2, H2_U2 = get_H(Uloc[:, 2])
    H         = 0.5 * (H1 + H2)
    H_U       = 0.5 .* vcat(H1_U1, H2_U2)

    # H* (KE shape)
    Hs1, Hs1_U1 = get_Hs(U1, param)
    Hs2, Hs2_U2 = get_Hs(U2, param)
    Hs, Hs_U    = upwind(0.5, 0.0, Hs1, Hs1_U1, Hs2, Hs2_U2)

    # log change in H*
    Hslog    = log(Hs2 / Hs1)
    Hslog_U  = vcat((-1/Hs1) .* Hs1_U1, (1/Hs2) .* Hs2_U2)

    # similarity station special case
    thlog_s    = thlog
    thlog_U_s  = thlog_U
    Hslog_s    = Hslog
    Hslog_U_s  = Hslog_U
    uelog_s    = uelog
    uelog_U_s  = uelog_U
    xlog_s     = xlog
    xlog_x_s   = xlog_x
    dx_s       = dx
    dx_x_s     = dx_x

    if param.simi
        thlog_s   = 0.0;  thlog_U_s .= 0.0
        Hslog_s   = 0.0;  Hslog_U_s .= 0.0
        uelog_s   = 1.0;  uelog_U_s .= 0.0
        xlog_s    = 1.0;  xlog_x_s   .= 0.0
        dx_s      = 0.5 * (x[1] + x[2])
        dx_x_s    = [0.5, 0.5]
    end

    # wake gap shape Hw
    Hw1, Hw1_U1 = get_Hw(Uloc[:, 1], Aux[1, 1])
    Hw2, Hw2_U2 = get_Hw(Uloc[:, 2], Aux[1, 2])
    Hw          = 0.5 * (Hw1 + Hw2)
    Hw_U        = 0.5 .* vcat(Hw1_U1, Hw2_U2)

    # shear-lag / amplification equation
    if param.turb
        # log change of sqrt(ctau)
        salog   = log(sa[2] / sa[1])
        salog_U = [0, 0, -1/sa[1], 0,  0, 0, 1/sa[2], 0]

        # BL thickness measure de (averaged)
        de1, de1_U1 = get_de(U1, param)
        de2, de2_U2 = get_de(U2, param)
        de,  de_U   = upwind(0.5, 0.0, de1, de1_U1, de2, de2_U2)

        # normalized slip velocity Us (averaged)
        Us1, Us1_U1 = get_Us(U1, param)
        Us2, Us2_U2 = get_Us(U2, param)
        Us,  Us_U   = upwind(0.5, 0.0, Us1, Us1_U1, Us2, Us2_U2)

        # Hk (upwinded)
        Hk1, Hk1_U1 = get_Hk(U1, param)
        Hk2, Hk2_U2 = get_Hk(U2, param)
        Hk,  Hk_U   = upwind(upw, upw_U, Hk1, Hk1_U1, Hk2, Hk2_U2)

        # Re_theta (averaged)
        Ret1, Ret1_U1 = get_Ret(U1, param)
        Ret2, Ret2_U2 = get_Ret(U2, param)
        Ret,  Ret_U   = upwind(0.5, 0.0, Ret1, Ret1_U1, Ret2, Ret2_U2)

        # cf (upwinded)
        cf1, cf1_U1 = get_cf(U1, param)
        cf2, cf2_U2 = get_cf(U2, param)
        cf,  cf_U   = upwind(upw, upw_U, cf1, cf1_U1, cf2, cf2_U2)

        # ds averaged (remember: wake-gap already removed)
        dsa   = 0.5 * (ds[1] + ds[2])
        dsa_U = 0.5 .* [0,1,0,0,  0,1,0,0]

        # equilibrium (1/ue) due/dx
        uq, uq_U = get_uq(dsa, dsa_U, cf, cf_U, Hk, Hk_U, Ret, Ret_U, param)

        # cteq (upwinded)
        cteq1, cteq1_U1 = get_cteq(U1, param)
        cteq2, cteq2_U2 = get_cteq(U2, param)
        cteq,  cteq_U   = upwind(upw, upw_U, cteq1, cteq1_U1, cteq2, cteq2_U2)

        # sqrt(ctau) state (upwinded from sa)
        saa, saa_U = upwind(upw, upw_U, sa[1], [0,0,1,0], sa[2], [0,0,1,0])

        # lag coefficient
        Klag  = param.SlagK
        beta  = param.GB
        Clag  = Klag / beta * 1/(1 + Us)
        Clag_U = -Clag/(1 + Us) .* Us_U

        # extra dissipation in wake
        ald = param.wake ? param.Dlr : 1.0

        # shear-lag residual
        Rlag   = Clag*(cteq - ald*saa) * dx_s - 2*de*salog + 2*de*(uq*dx_s - uelog_s)*param.Cuq
        Rlag_U = Clag_U*(cteq - ald*saa)*dx_s .+ Clag*(cteq_U .- ald .* saa_U)*dx_s .-
                 2 .* de_U .* salog .- 2 .* de .* salog_U .+
                 2 .* de_U .* (uq*dx_s - uelog_s) .* param.Cuq .+ 2 .* de .* (uq_U .* dx_s .- uelog_U_s) .* param.Cuq
        Rlag_x = Clag*(cteq - ald*saa) .* dx_x_s .+ 2 .* de .* uq .* dx_x_s
    else
        # laminar: amplification equation
        if param.simi
            Rlag   = sa[1] + sa[2]
            Rlag_U = [0,0,1,0,  0,0,1,0]
            Rlag_x = [0.0, 0.0]
        else
            damp1, damp1_U1 = get_damp(U1, param)
            damp2, damp2_U2 = get_damp(U2, param)
            damp,  damp_U   = upwind(0.5, 0.0, damp1, damp1_U1, damp2, damp2_U2)
            Rlag   = sa[2] - sa[1] - damp*dx
            Rlag_U = [0,0,-1,0,  0,0,1,0] .- damp_U .* dx
            Rlag_x = .-damp .* dx_x
        end
    end

    # M^2 (averaged)
    Ms1, Ms1_U1 = get_Mach2(U1, param)
    Ms2, Ms2_U2 = get_Mach2(U2, param)
    Ms,  Ms_U   = upwind(0.5, 0.0, Ms1, Ms1_U1, Ms2, Ms2_U2)

    # cf*x/θ (symm avg with midpoint correction)
    cfxt1, cfxt1_U1, cfxt1_x1 = get_cfxt(U1, x[1], param)
    cfxt2, cfxt2_U2, cfxt2_x2 = get_cfxt(U2, x[2], param)
    cfxtm, cfxtm_Um, cfxtm_xm = get_cfxt(Um, 0.5*(x[1] + x[2]), param)

    cfxt   = 0.25*cfxt1 + 0.5*cfxtm + 0.25*cfxt2
    cfxt_U = 0.25 .* vcat(cfxt1_U1 .+ cfxtm_Um, cfxtm_Um .+ cfxt2_U2)
    cfxt_x = 0.25 .* [cfxt1_x1 + cfxtm_xm, cfxtm_xm + cfxt2_x2]

    # momentum residual
    Rmom   = thlog_s + (2 + H + Hw - Ms) * uelog_s - 0.5 * xlog_s * cfxt
    Rmom_U = thlog_U_s .+ (H_U .+ Hw_U .- Ms_U) .* uelog_s .+ (2 + H + Hw - Ms) .* uelog_U_s .- 0.5 * xlog_s .* cfxt_U
    Rmom_x = .-0.5 .* xlog_x_s .* cfxt .- 0.5 .* xlog_s .* cfxt_x

    # cDi*x/θ (upwinded)
    cDixt1, cDixt1_U1, cDixt1_x1 = get_cDixt(U1, x[1], param)
    cDixt2, cDixt2_U2, cDixt2_x2 = get_cDixt(U2, x[2], param)
    cDixt,  cDixt_U               = upwind(upw, upw_U, cDixt1, cDixt1_U1, cDixt2, cDixt2_U2)
    cDixt_x = [(1 - upw) * cDixt1_x1,  upw * cDixt2_x2]

    # cf*x/θ (upwinded)
    cfxtu,  cfxtu_U  = upwind(upw, upw_U, cfxt1, cfxt1_U1, cfxt2, cfxt2_U2)
    cfxtu_x          = [(1 - upw) * cfxt1_x1,  upw * cfxt2_x2]

    # Hss (averaged)
    Hss1, Hss1_U1 = get_Hss(U1, param)
    Hss2, Hss2_U2 = get_Hss(U2, param)
    Hss,  Hss_U   = upwind(0.5, 0.0, Hss1, Hss1_U1, Hss2, Hss2_U2)

    # shape-parameter residual
    Rshape   = Hslog_s + (2*Hss/Hs + 1 - H - Hw) * uelog_s + xlog_s * (0.5*cfxtu - cDixt)
    Rshape_U = Hslog_U_s .+ (2 .* Hss_U ./ Hs .- 2 .* Hss ./ (Hs^2) .* Hs_U .- H_U .- Hw_U) .* uelog_s .+
               (2*Hss/Hs + 1 - H - Hw) .* uelog_U_s .+
               xlog_s .* (0.5 .* cfxtu_U .- cDixt_U)
    Rshape_x = xlog_x_s .* (0.5*cfxtu - cDixt) .+ xlog_s .* (0.5 .* cfxtu_x .- cDixt_x)

    # pack outputs with consistent shapes
    R = [Rmom; Rshape; Rlag]

    # Make each Jacobian row explicitly 1×8, then stack → 3×8
    R_U = vcat(
        reshape(Rmom_U,   1, :),
        reshape(Rshape_U, 1, :),
        reshape(Rlag_U,   1, :)
    )

    # Same idea for the x-derivatives: each 1×2, then stack → 3×2
    R_x = vcat(
        reshape(Rmom_x,   1, :),
        reshape(Rshape_x, 1, :),
        reshape(Rlag_x,   1, :)
    )

    return R, R_U, R_x
end

function wake_init(M, ue::Real)
    # first wake index, and the corresponding current state
    @assert !isempty(M.vsol.Is[3]) "wake_init: no wake indices found; did you run build_wake/identify_surfaces?"
    iw = M.vsol.Is[3][1]
    @assert size(M.glob.U, 1) ≥ 4 "wake_init: M.glob.U must have 4 rows (th, ds, sa, ue)."

    Uw = copy(@view M.glob.U[:, iw])  # 4×1 vector

    # construct the wake system residuals
    R, _, _ = wake_sys(M, M.param)    # expect R to be a length-3 vector
    @assert length(R) == 3 "wake_init: wake_sys must return a 3-vector residual R."

    # apply the update: solve the tiny system by a single subtraction, as in MATLAB
    Uw[1:3] .-= R
    Uw[4]    =  ue

    return Uw
end

function wake_sys(M, param)
    # Indices at the trailing edge (lower and upper) and first wake node
    il = M.vsol.Is[1][end]    # lower-surface TE index
    iu = M.vsol.Is[2][end]    # upper-surface TE index
    iw = M.vsol.Is[3][1]      # first wake index

    Ul = M.glob.U[:, il]      # state at lower TE
    Uu = M.glob.U[:, iu]      # state at upper TE
    Uw = M.glob.U[:, iw]      # state at first wake point

    # TE gap
    _, hTE, _, _, _ = TE_info(M.foil.x)

    # Compute turbulent root shear stress at TE on each side.
    # If already turbulent, take it directly from the state; otherwise take transition value.
    # We compute with a *local* param so we don't mutate the caller's struct.
    p = deepcopy(param)
    p.turb = true
    p.wake = false

    if M.vsol.turb[il]
        ctl     = Ul[3]
        ctl_Ul  = [0.0, 0.0, 1.0, 0.0]  # d ctl / d Ul
    else
        ctl, ctl_Ul = get_cttr(Ul, p)   # 1×4 row derivative
    end

    if M.vsol.turb[iu]
        ctu     = Uu[3]
        ctu_Uu  = [0.0, 0.0, 1.0, 0.0]
    else
        ctu, ctu_Uu = get_cttr(Uu, p)
    end

    # Theta-weighted average for wake shear (root)
    thsum = Ul[1] + Uu[1]
    ctw   = (ctl*Ul[1] + ctu*Uu[1]) / thsum

    # Derivatives of ctw wrt Ul and Uu (1×4 rows)
    # ctw = (ctl*th_l + ctu*th_u) / (th_l + th_u)
    ctw_Ul = (ctl_Ul .* Ul[1] .+ (ctl - ctw) .* [1.0, 0.0, 0.0, 0.0]) ./ thsum
    ctw_Uu = (ctu_Uu .* Uu[1] .+ (ctu - ctw) .* [1.0, 0.0, 0.0, 0.0]) ./ thsum

    # Residual (note: delta* in wake includes TE gap hTE)
    # R = [ th_w - (th_l + th_u);
    #       ds_w - (ds_l + ds_u + hTE);
    #       sa_w - ctw ]
    R = [
        Uw[1] - (Ul[1] + Uu[1]);
        Uw[2] - (Ul[2] + Uu[2] + hTE);
        Uw[3] - ctw
    ]

    # Jacobians wrt Ul, Uu, Uw -> build 3×4 blocks and then hcat
    # For Ul block:
    # rows 1–2 are -I on th, ds; row 3 is -ctw_Ul
    Ul_block_top = hcat(-Matrix{Float64}(I, 2, 2), zeros(2, 2))    # 2×4
    R_Ul = vcat(Ul_block_top, -reshape(ctw_Ul, 1, 4))               # 3×4

    # For Uu block:
    Uu_block_top = hcat(-Matrix{Float64}(I, 2, 2), zeros(2, 2))  # [-I 0]
    R_Uu = vcat(Uu_block_top, -reshape(ctw_Uu, 1, 4))

    # For Uw block: eye(3,4) (identity on first three comps, last column zeros)
    R_Uw = hcat(Matrix{Float64}(I, 3, 3), zeros(3, 1))              # 3×4

    R_U = hcat(R_Ul, R_Uu, R_Uw)

    # Node indices used (lower TE, upper TE, first wake)
    J = [il, iu, iw]

    return R, R_U, J
end

function residual_transition(M, param, x::AbstractVector, U::AbstractMatrix, Aux)
    @assert size(U,1) == 4 && size(U,2) == 2 "U must be 4×2"
    @assert length(x) == 2 "x must be length-2"
    @assert hasproperty(param, :is) "param.is (side index 1/2/3) is required."

    # states and handy slices
    U1 = @view U[:,1]
    U2 = @view U[:,2]
    sa = U[3, :]
    I1 = 1:4
    I2 = 5:8
    Z  = zeros(4)

    # forced transition inputs
    is = getfield(param, :is)
    forcet = M.oper.forcet[is]
    xft    = M.oper.xft[is]  * M.geom.chord
    xift   = M.oper.xift[is]

    # interval quantities
    x1, x2 = x
    dx = x2 - x1

    # Find transition location xt (either Newton on amplification or forced)
    if !forcet
        xt = x1 + 0.5*dx
        ncrit = param.ncrit
        nNewton = 20
        vprint(param, 3, "  Transition interval = %.5e .. %.5e\n", x1, x2)

        Rxt = 0.0
        Rxt_xt = 0.0
        damp1 = 0.0
        dampt = 0.0
        dampt_Ut = zeros(4)
        Ut = similar(U1)
        Ut_xt = similar(U1)

        for iNewton in 1:nNewton
            w2 = (xt - x1)/dx
            w1 = 1 - w2
            Ut .= w1 .* U1 .+ w2 .* U2
            Ut_xt .= (U2 .- U1) ./ dx
            Ut[3] = ncrit
            Ut_xt[3] = 0.0

            damp1, _damp1_U1 = get_damp(U1, param)
            dampt, dampt_Ut  = get_damp(Ut,  param)
            dampt_Ut[3] = 0.0  # amplification state fixed to ncrit at xt

            Rxt     = ncrit - sa[1] - 0.5*(xt - x1)*(damp1 + dampt)
            Rxt_xt  = -0.5*(damp1 + dampt) - 0.5*(xt - x1) * (dampt_Ut' * Ut_xt)

            dxt = -Rxt / Rxt_xt
            vprint(param, 4, "   Transition: iNewton=%d, Rxt=%.5e, xt=%.5e\n", iNewton, Rxt, xt)

            dmax = 0.2*dx*(1.1 - iNewton/nNewton)
            if abs(dxt) > dmax
                dxt = sign(dxt) * dmax
            end
            (abs(Rxt) < 1e-10) && break
            (iNewton < nNewton) && (xt += dxt)
        end

        # sensitivities of xt
        # Rxt_U = -0.5*(xt-x1)*[damp1_U1 + dampt_Ut*w1,  dampt_Ut*w2];  Rxt_U(3) -= 1;
        _, damp1_U1 = get_damp(U1, param)
        Rxt_U = -0.5*(xt - x1) .* vcat(damp1_U1 .+ dampt_Ut .* ((dx - (xt - x1))/dx),
                                       dampt_Ut .* ((xt - x1)/dx))
        Rxt_U[3] -= 1.0

        Ut_x1 = (U2 .- U1) .* (( (xt - x1)/dx - 1.0) ./ dx)     # (w2-1)/dx
        Ut_x2 = (U2 .- U1) .* (   (-(xt - x1)/dx) ./ dx)        # (-w2)/dx
        Ut_x1[3] = 0.0;  Ut_x2[3] = 0.0

        Rxt_x1 =  0.5*(damp1 + dampt) - 0.5*(xt - x1) * (dampt_Ut' * Ut_x1)
        Rxt_x2 = -0.5*(xt - x1) * (dampt_Ut' * Ut_x2)

        xt_U  = (-1.0 / Rxt_xt) .* Rxt_U
        xt_U1 = xt_U[I1]
        xt_U2 = xt_U[I2]
        xt_x1 = -Rxt_x1 / Rxt_xt
        xt_x2 = -Rxt_x2 / Rxt_xt
        M.vsol.xt = xt

    else
        # forced transition
        xt = xift
        w2 = (xt - x1)/dx
        w1 = 1 - w2
        Ut = w1 .* U1 .+ w2 .* U2
        Ut_xt = (U2 .- U1) ./ dx

        Rxt = 0.0; Rxt_xt = 1.0
        Rxt_x1 = -w1; Rxt_x2 = -w2
        Rxt_U = zeros(8)

        Ut_x1 = (U2 .- U1) .* (( (xt - x1)/dx - 1.0) ./ dx)
        Ut_x2 = (U2 .- U1) .* (   (-(xt - x1)/dx) ./ dx)

        xt_U1 = zeros(4); xt_U2 = zeros(4)
        xt_x1 = -Rxt_x1 / Rxt_xt
        xt_x2 = -Rxt_x2 / Rxt_xt
        M.vsol.xt = xt
    end

    # include d(xt) into ∂Ut/∂x
    Ut_x1 .= Ut_x1 .+ Ut_xt .* xt_x1
    Ut_x2 .= Ut_x2 .+ Ut_xt .* xt_x2

    # sensitivities of Ut w.r.t. U1, U2
    I4 = Matrix{Float64}(I, 4, 4)
    Ut_U1 = w1 * I4 .+ (U2 .- U1) * (xt_U1' ./ dx)  # 4×4
    Ut_U2 = w2 * I4 .+ (U2 .- U1) * (xt_U2' ./ dx)  # 4×4

    # laminar/turbulent states at xt
    Utl     = copy(Ut); Utl_U1 = copy(Ut_U1); Utl_U2 = copy(Ut_U2); Utl_x1 = copy(Ut_x1); Utl_x2 = copy(Ut_x2)
    if !forcet
        ncrit = param.ncrit
        Utl[3] = ncrit
        Utl_U1[3, :] .= Z
        Utl_U2[3, :] .= Z
        Utl_x1[3] = 0.0
        Utl_x2[3] = 0.0
    end

    Utt     = copy(Ut); Utt_U1 = copy(Ut_U1); Utt_U2 = copy(Ut_U2); Utt_x1 = copy(Ut_x1); Utt_x2 = copy(Ut_x2)

    # parameter structure (fresh), then set turbulent shear at transition
    p = build_param(M, 0)
    p.turb = true
    cttr, cttr_Ut = get_cttr(Ut, p)
    Utt[3] = cttr
    Utt_U1[3, :] = vec(reshape(cttr_Ut, 1, :) * Ut_U1)  # 1×4 * 4×4 -> 1×4
    Utt_U2[3, :] = vec(reshape(cttr_Ut, 1, :) * Ut_U2)
    Utt_x1[3]    = dot(cttr_Ut, Utt_x1)
    Utt_x2[3]    = dot(cttr_Ut, Utt_x2)

    # laminar residual on [x1, xt] with [U1, Utl]
    p.turb = false
    Rl, Rl_U, Rl_x = residual_station(p, [x1, xt], hcat(U1, Utl), Aux)
    Rl_U1  = Rl_U[:, I1]
    Rl_Utl = Rl_U[:, I2]
    if forcet
        Rl[3]           = 0.0
        Rl_U1[3, :]     .= Z
        Rl_Utl[3, :]    .= Z
    end

    # turbulent residual on [xt, x2] with [Utt, U2]
    p.turb = true
    Rt, Rt_U, Rt_x = residual_station(p, [xt, x2], hcat(Utt, U2), Aux)
    Rt_Utt = Rt_U[:, I1]
    Rt_U2  = Rt_U[:, I2]

    # combine
    R = Rl + Rt
    # Jacobians w.r.t U1, U2
    R_U1 = Rl_U1 + Rl_Utl * Utl_U1 + (Rl_x[:, 2] * xt_U1') + Rt_Utt * Utt_U1 + (Rt_x[:, 1] * xt_U1')
    R_U2 = Rl_Utl * Utl_U2 + (Rl_x[:, 2] * xt_U2') + Rt_Utt * Utt_U2 + Rt_U2 + (Rt_x[:, 1] * xt_U2')
    R_U  = hcat(R_U1, R_U2)

    # Jacobians w.r.t x1, x2
    Rx1 = Rl_x[:, 1] + Rl_x[:, 2]*xt_x1 + Rt_x[:, 1]*xt_x1 + Rl_Utl*Utl_x1 + Rt_Utt*Utt_x1
    Rx2 = Rt_x[:, 2] + Rl_x[:, 2]*xt_x2 + Rt_x[:, 1]*xt_x2 + Rl_Utl*Utl_x2 + Rt_Utt*Utt_x2
    R_x = hcat(Rx1, Rx2)

    return R, R_U, R_x
end

function store_transition!(M, is::Int, i::Int)
    forcet = M.oper.forcet[is]                 # forced transition flag (Bool)
    xft    = M.oper.xft[is] * M.geom.chord     # forced transition x location

    # pre/post transition nodes on the airfoil
    i0 = M.vsol.Is[is][i-1]
    i1 = M.vsol.Is[is][i]
    @assert (i0 ≤ M.foil.N) && (i1 ≤ M.foil.N) "Can only store transition on airfoil"

    # xi (s) and x at those nodes
    xi0, xi1 = M.isol.xi[i0], M.isol.xi[i1]
    x0,  x1  = M.foil.x[1, i0], M.foil.x[1, i1]

    # xi-location corresponding to a forced *x*-location between nodes
    xift = xi0 + (xi1 - xi0) * (xft - x0) / (x1 - x0)

    # choose free vs forced transition
    if (!forcet) || ((M.vsol.xt > 0) && (M.vsol.xt < xift))
        xt  = M.vsol.xt
        spre = "free"
    else
        xt  = xift
        # tuples are immutable; replace by making a new tuple
        M.oper.xift = Base.setindex(M.oper.xift, xt, is)
        spre = "forced"
    end

    # warn if out of interval
    if (xt < xi0) || (xt > xi1)
        vprint(M.param, 1, "Warning: transition (%.3f) off interval (%.3f, %.3f)!\n", xt, xi0, xi1)
    end

    # save xi and x locations; rows = side (1=lower,2=upper), cols: (1=xi,2=x)
    M.vsol.Xt[is, 1] = xt
    M.vsol.Xt[is, 2] = x0 + (xt - xi0) / (xi1 - xi0) * (x1 - x0)

    slu = ("lower", "upper")
    vprint(M.param, 1, "  %s transition on %s side at x=%.5f\n", spre, slu[is], M.vsol.Xt[is,2])
    return nothing
end

function update_transition!(M)
    for is in 1:2  # lower/upper surfaces
        Is = M.vsol.Is[is]
        N  = length(Is)
        xft = M.oper.xft[is] * M.geom.chord

        param = build_param(M, is)

        # current last laminar station index (within Is)
        lam_flags = .! M.vsol.turb[Is]              # true where laminar
        I = findall(identity, lam_flags)
        @assert !isempty(I) "No laminar stations found on side $is"
        ilam0 = I[end]

        # keep current sa (amp/ctau) so we can restore if nothing changes
        sa_saved = copy(M.glob.U[3, Is])

        # NEW last laminar station (you must implement this function)
        ilam = march_amplification(M, is)

        # if forcing, set xi of forced transition based on current bracket
        if M.oper.forcet[is] && ilam < N
            i0 = Is[ilam]
            i1 = Is[ilam+1]
            xi0, xi1 = M.isol.xi[i0], M.isol.xi[i1]
            x0,  x1  = M.foil.x[1, i0], M.foil.x[1, i1]
            xift = xi0 + (xi1 - xi0) * (xft - x0) / (x1 - x0)
            M.oper.xift = Base.setindex(M.oper.xift, xift, is)
        end

        # no change? restore and continue
        if ilam == ilam0
            M.glob.U[3, Is] = sa_saved
            continue
        end

        vprint(param, 2, "  Update transition: last lam [%d]->[%d]\n", ilam0, ilam)

        if ilam < ilam0
            # transition moved earlier: fill turbulent between [ilam+1, ilam0]
            p = build_param(M, is)
            p.turb = true
            p.wake = false

            # shear coefficient at first turb station after transition
            sa0, _ = get_cttr(M.glob.U[:, Is[ilam+1]], p)
            # target at the end of the segment, if it exists; otherwise flat
            sa1 = (ilam0 < N) ? M.glob.U[3, Is[ilam0+1]] : sa0

            xi = M.isol.xi[Is]
            dx = xi[min(ilam0+1, N)] - xi[ilam+1]

            for k in (ilam+1):ilam0
                f = (dx == 0 || k == ilam+1) ? 0.0 : (xi[k] - xi[ilam+1]) / dx
                if (ilam + 1) == ilam0
                    f = 1.0
                end
                M.glob.U[3, Is[k]] = sa0 + f * (sa1 - sa0)
                @assert M.glob.U[3, Is[k]] > 0.0 "negative ctau in update_transition!"
                M.vsol.turb[Is[k]] = true
            end

        elseif ilam > ilam0
            # transition moved later: mark region as laminar
            for k in ilam0:ilam
                M.vsol.turb[Is[k]] = false
            end
        end
    end
    return nothing
end

function march_amplification(M, is::Int)
    # stations on this side
    Is = M.vsol.Is[is]
    N  = length(Is)

    # params and local copies
    param = build_param(M, is)
    U     = copy(M.glob.U[:, Is])         # 4×N
    turb  = M.vsol.turb[Is]               # Vector{Bool}
    xft   = M.oper.xft[is] * M.geom.chord

    # laminar marching setup
    U[3, 1] = 0.0                         # no amplification at first station
    param.turb = false
    param.wake = false

    i = 2
    while i <= N
        U1 = view(U, :, i-1)
        U2 = view(U, :, i)

        # if current station is turbulent, seed amp slightly above previous
        if turb[i]
            U2[3] = U1[3] * 1.01
        end

        dx = M.isol.xi[Is[i]] - M.isol.xi[Is[i-1]]

        # Newton iterations (mostly for the extra amplification term smoothness)
        nNewton = 20
        converged = false
        for iNewton in 1:nNewton
            damp1, damp1_U1 = get_damp(U1, param)
            damp2, damp2_U2 = get_damp(U2, param)
            # symmetric average (no upwind); provide zero d(upw)/dU vector
            damp, damp_U = upwind(0.5, zeros(1,8), damp1, damp1_U1, damp2, damp2_U2)

            Ramp = U2[3] - U1[3] - damp*dx

            if iNewton > 12
                vprint(param, 3,
                    "i=%d, iNewton=%d, sa = [%.5e, %.5e], damp = %.5e, Ramp = %.5e\n",
                    i, iNewton, U1[3], U2[3], damp, Ramp)
            end

            if abs(Ramp) < 1e-12
                converged = true
                break
            end

            # derivative wrt the update variable U2[3] (slot 7 in [U1;U2])
            Ramp_U = [0,0,-1,0, 0,0,1,0] .- damp_U .* dx
            dU = -Ramp / Ramp_U[7]

            # small damping/limiter
            dmax = 0.5 * (1.01 - iNewton/nNewton)
            ω = abs(dU) > dmax ? dmax/abs(dU) : 1.0
            U2[3] += ω * dU
        end
        if !converged
            vprint(param, 1, "march amp Newton unconverged!\n")
        end

        # check for forced transition crossing in x
        # (sign change of x - xft across the interval)
        M.oper.forcet[is] = false
        x_left  = M.foil.x[1, Is[i-1]] - xft
        x_right = M.foil.x[1, Is[i]]   - xft
        if x_left * x_right < 0.0
            M.oper.forcet = Base.setindex(M.oper.forcet, true, is)
            vprint(param, 2, "  forced transition (is,i=%d,%d)\n", is, i)
            break
        elseif U2[3] > param.ncrit
            vprint(param, 2,
                "  march_amplification (is,i=%d,%d): %.5e is above critical.\n",
                is, i, U2[3])
            break
        else
            # store amplification back to the global state and local copy
            M.glob.U[3, Is[i]] = U2[3]
            U[3, i] = U2[3]
            @assert isreal(U[3, i]) "imaginary amp during march"
        end

        i += 1
    end

    ilam = i - 1
    return ilam
end

function init_boundary_layer!(M)
    # thresholds for separation checks
    Hmaxl = 3.8   # laminar separation trigger
    Hmaxt = 2.5   # turbulent separation trigger

    ueinv = get_ueinv(M)                   # inviscid ue at foil (+ wake)
    M.glob.Nsys = M.foil.N + M.wake.N      # total nodes

    # Reuse existing BL if allowed, but refresh ue
    if (!M.oper.initbl) && (size(M.glob.U, 2) == M.glob.Nsys)
        vprint(M.param, 1, "\n <<< Starting with current boundary layer >>> \n")
        M.glob.U[4, :] .= ueinv
        return
    end

    vprint(M.param, 1, "\n <<< Initializing the boundary layer >>> \n")

    # fresh state & flags
    M.glob.U = zeros(4, M.glob.Nsys)
    M.vsol.turb = falses(M.glob.Nsys)

    # loop over sides: 1=lower, 2=upper, 3=wake
    for is in 1:3
        vprint(M.param, 3, "\nSide is = %d:\n", is)

        Is  = M.vsol.Is[is]               # node indices on this side
        xi  = M.isol.xi[Is]               # distance from stagnation
        ue  = ueinv[Is]                   # edge speeds on this side
        N   = length(Is)

        # x-locations only for foil sides
        xg = is < 3 ? M.foil.x[1, Is] : similar(xi)  # dummy for wake
        if is < 3
            copyto!(xg, M.foil.x[1, Is])
        end

        # clamp tiny edge speeds
        uemax = maximum(abs.(ue))
        ue .= max.(ue, 1e-8 * uemax)

        # parameters & Aux (only wake has nonzero wgap)
        param = build_param(M, is)
        Aux   = zeros(1, N)
        if is == 3
            Aux[1, :] .= M.vsol.wgap
        end

        # local state array for this side
        U = zeros(4, N)

        # forced transition x-location
        xft = M.oper.xft[is] * M.geom.chord

        # initialize first point
        i0 = 1
        if is < 3
            # Thwaites init near stagnation
            hitstag = xi[1] < 1e-8 * xi[end]
            K = hitstag ? ue[2] / xi[2] : ue[1] / xi[1]

            th, ds = thwaites_init(K, M.param.mu0 / M.param.rho0)
            xst = 1e-6
            Ust = [th, ds, 0.0, K * xst]

            # Newton to satisfy stagnation (similarity) residual
            nNewton = 20
            for iNewton in 1:nNewton
                param.turb = false
                param.simi = true
                R, R_U, _ = residual_station(param, [xst, xst], hcat(Ust, Ust), zeros(1, 2))
                param.simi = false
                if norm(R) < 1e-10
                    break
                end
                ID = 1:3
                A  = R_U[:, ID .+ 4] + R_U[:, ID]  # (∂R/∂U2 + ∂R/∂U1) for th,ds,sa
                dU = vcat(A \ (-R), 0.0)

                dm = maximum(abs.([dU[1]/Ust[1], dU[2]/Ust[2]]))
                ω  = dm > 0.2 ? 0.2/dm : 1.0
                Ust .+= ω .* dU
            end

            if hitstag
                U[:, 1] .= Ust
                U[4, 1] = ue[1]
                i0 = 2
            end
            U[:, i0] .= Ust
            U[4, i0] = ue[i0]
        else
            # wake start from TE using wake system
            U[:, 1] .= wake_init(M, ue[1])
            param.turb = true
            M.vsol.turb[Is[1]] = true
        end

        # march along the side
        tran = false
        i = i0 + 1
        while i <= N
            Ip = (i-1):i

            # forced transition detection (foil only)
            if (!tran) && (!param.turb) && (is < 3)
                left  = xg[i-1] - xft
                right = xg[i]   - xft
                if left * right < 0.0
                    tran = true
                    # set forced xi transition location
                    xift = xi[i-1] + (xi[i] - xi[i-1]) * (xft - xg[i-1]) / (xg[i] - xg[i-1])
                    M.oper.xift = Base.setindex(M.oper.xift, xift, is)
                    M.oper.forcet = Base.setindex(M.oper.forcet, true, is)
                    vprint(param, 1, "forced transition during marching: xft=%.5f, xift=%.5f\n",
                           xft, xift)
                end
            end

            # initial guess: copy previous, set new ue
            U[:, i] .= U[:, i-1]
            U[4, i]   = ue[i]
            if tran
                ct, _ = get_cttr(U[:, i], param)
                U[3, i] = ct
            end
            M.vsol.turb[Is[i]] = (tran || param.turb)

            direct    = true
            nNewton   = 30
            iNswitch  = 12
            Hktgt     = NaN

            converged = false
            iNewton_last = 0

            for iNewton in 1:nNewton
                iNewton_last = iNewton
                if tran
                    vprint(param, 4, "i=%d, residual_transition (iNewton = %d)\n", i, iNewton)
                    R = nothing; R_U = nothing
                    try
                        param.is = is
                        R, R_U, _ = residual_transition(M, param, xi[Ip], U[:, Ip], Aux[:, Ip])
                    catch
                        @warn "Transition calculation failed in BL init. Continuing."
                        M.vsol.xt = 0.5 * (xi[i-1] + xi[i])
                        U[:, i]   .= U[:, i-1]
                        U[4, i]    = ue[i]
                        ct, _ = get_cttr(U[:, i], param)  # ignore gradient
                        U[3, i]    = ct
                        R = zeros(3) # so we "converge"
                    end
                else
                    vprint(param, 4, "i=%d, residual_station (iNewton = %d)\n", i, iNewton)
                    R, R_U, _ = residual_station(param, xi[Ip], U[:, Ip], Aux[:, Ip])
                end

                if norm(R) < 1e-10
                    converged = true
                    break
                end

                dU = zeros(4)
                if direct
                    # solve for th, ds, sa (ue prescribed)
                    A = R_U[:, 5:7]
                    dU[1:3] .= A \ (-R)
                    dU[4] = 0.0 
                else
                    # inverse: prescribe Hk target
                    Hk, Hk_U = get_Hk(U[:, i], param)      # Hk_U is a length-4 Vector
                    A = vcat(R_U[:, 5:8], (Hk_U[:])')      # 3×4 ; 1×4 -> 4×4
                    b = vcat(-R, Hktgt - Hk)               # 3 ; 1 -> 4
                    dU .= A \ b
                end

                # under-relax
                dm = maximum(abs.([dU[1]/U[1, i-1], dU[2]/U[2, i-1]]))
                if !direct
                    dm = max(dm, abs(dU[4] / max(U[4, i-1], 1e-16)))
                end
                if param.turb
                    dm = max(dm, abs(dU[3] / max(U[3, i-1], 1e-16)))
                else
                    dm = max(dm, abs(dU[3] / 10))
                end
                ω = dm > 0.3 ? 0.3 / dm : 1.0
                dU .*= ω

                Ui = U[:, i] .+ dU

                # clip extremes
                if param.turb
                    Ui[3] = clamp(Ui[3], 1e-7, 0.3)
                end

                # check for separation / switch to inverse
                Hmax = param.turb ? Hmaxt : Hmaxl
                Hk, _ = get_Hk(Ui, param)
                if direct && ((Hk > Hmax) || (iNewton > iNswitch))
                    direct = false
                    vprint(param, 2, "** switching to inverse: i=%d, iNewton=%d\n", i, iNewton)
                    Hk_prev, _ = get_Hk(U[:, i-1], param)
                    Hkr = (xi[i] - xi[i-1]) / U[1, i-1]

                    if param.wake
                        # implicit relation for wake
                        H2 = Hk_prev
                        for _ in 1:6
                            num = H2 + 0.03 * Hkr * (H2 - 1)^3 - Hk_prev
                            den = 1 + 0.09 * Hkr * (H2 - 1)^2
                            H2 -= num / den
                        end
                        Hktgt = max(H2, 1.01)
                    elseif param.turb
                        Hktgt = Hk_prev - 0.15 * Hkr
                    else
                        Hktgt = Hk_prev + 0.03 * Hkr
                    end
                    if !param.wake
                        Hktgt = max(Hktgt, Hmax)
                    end
                    if iNewton > iNswitch
                        U[:, i] .= U[:, i-1]
                        U[4, i]  = ue[i]
                    end
                else
                    U[:, i] .= Ui
                end
            end

            if !converged
                vprint(param, 1, "** BL init not converged: is=%d, i=%d **\n", is, i)
                # extrapolate fallback
                U[:, i] .= U[:, i-1]
                U[4, i]  = ue[i]
                if is < 3
                    rat = xi[i] / xi[i-1]
                    U[1, i] = U[1, i-1] * sqrt(rat)
                    U[2, i] = U[2, i-1] * sqrt(rat)
                else
                    rlen   = (xi[i] - xi[i-1]) / (10 * U[2, i-1])
                    U[2, i] = (U[2, i-1] + U[1, i-1] * rlen) / (1 + rlen)
                end
            end

            # transition detection (free transition)
            if (!param.turb) && (!tran) && (U[3, i] > param.ncrit)
                vprint(param, 2,
                       "Identified transition at (is=%d, i=%d): n=%.5f, ncrit=%.5f\n",
                       is, i, U[3, i], param.ncrit)
                tran = true
                continue  # redo station with transition model
            end

            if tran
                store_transition!(M, is, i)
                param.turb = true
                tran = false
                vprint(param, 2, "storing transition\n")
            end

            i += 1
        end

        # save back to global
        M.glob.U[:, Is] .= U
    end

    return nothing
end

function stagpoint_move!(M)
    N  = M.foil.N                           # number of airfoil nodes
    I  = collect(M.isol.Istag)              # [i_lower, i_upper] (adjacent panel indices)
    ue = collect(M.glob.U[4, :])            # copy 4th row as a Vector
    sstag0 = M.isol.sstag

    newpanel = true
    I1, I2 = I[1], I[2]

    if ue[I2] < 0
        # move stagnation point up (to larger s), find first positive ue above I2
        jrel = findfirst(>(0.0), @view ue[I2:end])  # relative index in slice
        jrel === nothing && error("stagpoint_move!: no positive ue above I2=$I2")
        I2new = I2 + (jrel - 1)
        for j in I2:(I2new-1)
            ue[j] = -ue[j]
        end
        I = [I2new-1, I2new]
    elseif ue[I1] < 0
        # move stagnation point down (to smaller s), find first positive ue below I1
        jrel = findfirst(>(0.0), @view ue[I1:-1:1])
        jrel === nothing && error("stagpoint_move!: no positive ue below I1=$I1")
        I1new = I1 - (jrel - 1)
        for j in (I1new+1):I1
            ue[j] = -ue[j]
        end
        I = [I1new, I1new+1]
    else
        newpanel = false
    end

    # move along the (possibly new) panel
    ues = ue[I]
    S   = M.foil.s[I]
    @assert (ues[1] > 0.0 && ues[2] > 0.0) "stagpoint_move!: velocity error at indices $I"

    den = ues[1] + ues[2]
    w1  = ues[2] / den
    w2  = ues[1] / den

    M.isol.sstag    = w1*S[1] + w2*S[2]
    M.isol.xstag = (
    w1*M.foil.x[1, I[1]] + w2*M.foil.x[1, I[2]],
    w1*M.foil.x[2, I[1]] + w2*M.foil.x[2, I[2]],
)
    k = (S[2] - S[1]) / (den*den)
    M.isol.sstag_ue = (k*ues[2], -k*ues[1])

    # vprint(M.param, 2, @sprintf("  Moving stagnation point: s=%.15e -> s=%.15e\n", sstag0, M.isol.sstag))

    # refresh xi for all nodes (airfoil then wake)
    sst = M.isol.sstag
    M.isol.xi = vcat(abs.(M.foil.s .- sst), M.wake.s .- sst)

    if newpanel
        # vprint(M.param, 2, @sprintf("  New stagnation panel = %d %d\n", I[1], I[2]))
        M.isol.Istag = (I[1], I[2])  # update stagnation indices
        sgnue = ones(eltype(ue), N)
        sgnue[1:I[1]] .= -1
        M.isol.sgnue = sgnue

        # Re-identify surfaces, commit ue sign changes near the stag, and rebuild sensitivities
        identify_surfaces!(M)     # your Julia port’s mutating version
        M.glob.U[4, :] .= ue
        rebuild_ue_m!(M)
    end

    return nothing
end

function stagnation_state(U::AbstractMatrix{<:Real}, x::AbstractVector{<:Real})
    @assert size(U,1) == 4 && size(U,2) == 2 "U must be 4×2"
    @assert length(x) == 2 "x must have length 2"

    U1 = @view U[:, 1]
    U2 = @view U[:, 2]
    x1 = float(x[1])
    x2 = float(x[2])

    dx = x2 - x1
    # light regularization to avoid singular derivatives
    if dx == 0.0
        dx = eps(Float64)
    end
    if x1 == 0.0
        x1 = eps(Float64)
    end

    dx_x = [-1.0, 1.0]                      # ∂dx/∂[x1,x2]

    rx   = x2 / x1
    rx_x = [-rx, 1.0] / x1                  # ∂(x2/x1)/∂[x1,x2]

    # linear extrapolation weights for th, ds, sa
    w1   =  x2 / dx
    w1_x = (-w1/dx) .* dx_x .+ [0.0, 1.0] / dx

    w2   = -x1 / dx
    w2_x = (-w2/dx) .* dx_x .+ [-1.0, 0.0] / dx

    Ust = U1 .* w1 .+ U2 .* w2              # 4×1

    # quadratic fit for ue slope near stagnation: ue ≈ K x
    wk1   =  rx / dx
    wk1_x =  rx_x ./ dx .- (wk1/dx) .* dx_x

    wk2   = -1.0 / (rx * dx)
    wk2_x = -wk2 .* (rx_x ./ rx .+ dx_x ./ dx)

    K   = wk1 * U1[4] + wk2 * U2[4]
    K_U = [0.0, 0.0, 0.0, wk1,  0.0, 0.0, 0.0, wk2]  # 1×8
    K_x = U1[4] .* wk1_x .+ U2[4] .* wk2_x           # 1×2

    # less-accurate linear option (kept as comments to match MATLAB):
    # K   = U1[4] / x1
    # K_U = [0.0, 0.0, 0.0, 1/x1,  0.0, 0.0, 0.0, 0.0]
    # K_x = [-K/x1, 0.0]

    # small but nonzero stagnation coordinate
    xst = 1e-6
    Ust[4] = K * xst

    # Build Ust_U (4×8): top 3 rows are [w1*I₃|0  w2*I₃|0], last row is K_U*xst
    I34 = zeros(3,4); I34[1,1]=1.0; I34[2,2]=1.0; I34[3,3]=1.0
    upper = hcat(w1 .* I34, w2 .* I34)                  # 3×8
    Ust_U = [upper; (K_U .* xst)']                      # 4×8

    # Build Ust_x (4×2): top 3 rows outer-product w.r.t. w1_x, w2_x; last row K_x*xst
    Ust_x_first3 = U1[1:3] * (w1_x') .+ U2[1:3] * (w2_x')  # 3×2
    Ust_x = vcat(Ust_x_first3, (K_x .* xst)')              # 4×2

    return Ust, Ust_U, Ust_x, xst
end

function build_glob_sys!(M)
    Nsys = M.glob.Nsys
    M.glob.R   = zeros(Float64, 3*Nsys)
    M.glob.R_U = spzeros(Float64, 3*Nsys, 4*Nsys)
    M.glob.R_x = spzeros(Float64, 3*Nsys, Nsys)

    for is in 1:3
        Is = M.vsol.Is[is]                 # indices on this surface
        xi = M.isol.xi[Is]                 # distance from LE stagnation
        N  = length(Is)
        U  = M.glob.U[:, Is]         # states [th,ds,sa,ue] on this surface
        Aux = zeros(Float64, 1, N)         # [wgap] row

        if is < 3
            xg  = M.foil.x[1, Is]          # global x on foil for side is
        end
        xft = M.oper.xft[is] * M.geom.chord

        # parameters for this side
        param = build_param(M, is)

        # set auxiliary data for wake
        if is == 3
            Aux[1, :] .= M.vsol.wgap
        end

        # special case of tiny first xi -> use second point as the "first" station
        i0 = (is < 3 && xi[1] < 1e-8 * xi[end]) ? 2 : 1

        # ----- first-point system
        if is < 3
            # stagnation-based first station (similarity)
            Ip = (i0):(i0+1)
            Ust, Ust_U, Ust_x, xst = stagnation_state(U[:, Ip], xi[Ip])
            param.turb = false
            param.simi = true
            xpair  = [xst, xst]                 # 2-element Vector{Float64}
            Upair  = hcat(Ust, Ust)             # 4×2 matrix, not Vector{Vector}
            AUpair = Aux[:, [i0, i0]]     # 1×2 matrix
            R1, R1_Ut, _ = residual_station(param, xpair, Upair, AUpair)
            param.simi = false

            # collapse the two-station tangent into Ust sensitivities, then to (U1,U2)
            R1_Ust = @views R1_Ut[:, 1:4] .+ R1_Ut[:, 5:8]   # 3×4
            R1_U   = R1_Ust * Ust_U                          # 3×8
            R1_x   = R1_Ust * Ust_x                          # 3×2
            J      = (Is[i0], Is[i0+1])

            if i0 == 2
                # i0=1 point landed on stagnation: enforce U1 = Ust on first node
                vprint(M.param, 2, "hit stagnation!\n")
                Ig  = (3*Is[1]-2):(3*Is[1])                  # rows for node Is[1]
                Jg1 = (4*Is[1]-3):(4*Is[1])
                M.glob.R[Ig] .= U[1:3, 1] .- Ust[1:3]
                M.glob.R_U[Ig, Jg1] .+= I(3)
                Jg12 = vcat((4*J[1]-3):(4*J[1]), (4*J[2]-3):(4*J[2]))
                M.glob.R_U[Ig, Jg12] .-= R1_U[1:3, :]
                M.glob.R_x[Ig, J[1]] .-= R1_x[1:3, 1]
                M.glob.R_x[Ig, J[2]] .-= R1_x[1:3, 2]
            end
        else
            # wake initialization at the first wake point
            R1, R1_U, J = wake_sys(M, param)
            R1_x = nothing                    # no xi dependence here
            param.turb = true                 # force turbulent in wake if needed
            param.wake = true
        end

        # store first-point system into the global residual/Jacobians
        Ig = (3*Is[i0]-2):(3*Is[i0])
        if is < 3
            M.glob.R[Ig] .= R1
            # two 4-column blocks, one per station in J
            M.glob.R_U[Ig, (4*Is[i0]-3):(4*Is[i0])]       .+= R1_U[:, 1:4]
            M.glob.R_U[Ig, (4*Is[i0+1]-3):(4*Is[i0+1])]   .+= R1_U[:, 5:8]
            # xi coupling, if available
            M.glob.R_x[Ig, Is[i0]]     .+= R1_x[:, 1]
            M.glob.R_x[Ig, Is[i0+1]]   .+= R1_x[:, 2]
        else
            M.glob.R[Ig] .= R1
            # R1_U has three 4-wide blocks: lower TE, upper TE, first wake
            M.glob.R_U[Ig, (4*J[1]-3):(4*J[1])] .+= R1_U[:, 1:4]
            M.glob.R_U[Ig, (4*J[2]-3):(4*J[2])] .+= R1_U[:, 5:8]
            M.glob.R_U[Ig, (4*J[3]-3):(4*J[3])] .+= R1_U[:, 9:12]
        end

        # ----- march over remaining stations
        tran = false
        for i in (i0+1):N
            Ip = (i-1):i

            # forced transition detection window
            M.oper.forcet[is] = false
            if !tran && !param.turb && (is < 3)
                if (xg[i-1]-xft) * (xg[i]-xft) < 0.0
                    tran = true
                    M.oper.xift[is] = xi[i-1] + (xi[i]-xi[i-1]) * (xft - xg[i-1]) / (xg[i] - xg[i-1])
                    M.oper.forcet[is] = true
                end
            end

            # actual transition flag based on current lam/turb node flags
            tran = xor(M.vsol.turb[Is[i-1]] != 0, M.vsol.turb[Is[i]] != 0)

            # residual/Jacobian at this station pair
            if tran
                param.is = is
                Ri, Ri_U, Ri_x = residual_transition(M, param, xi[Ip], U[:, Ip], Aux[:, Ip])
                store_transition!(M, is, i)
            else
                Ri, Ri_U, Ri_x = residual_station(param, xi[Ip], U[:, Ip], Aux[:, Ip])
            end

            # accumulate into global structures
            Ig  = (3*Is[i]-2):(3*Is[i])                     # rows for node i
            JgL = (4*Is[i-1]-3):(4*Is[i-1])
            JgR = (4*Is[i]-3):(4*Is[i])
            M.glob.R[Ig]                .+= Ri
            M.glob.R_U[Ig, JgL]         .+= Ri_U[:, 1:4]
            M.glob.R_U[Ig, JgR]         .+= Ri_U[:, 5:8]
            M.glob.R_x[Ig, Is[i-1]]     .+= Ri_x[:, 1]
            M.glob.R_x[Ig, Is[i]]       .+= Ri_x[:, 2]

            # once transitioned, all following stations are turbulent
            if tran
                param.turb = true
            end
        end
    end

    # (Optional) special stagnation residual treatment could go here.
    return nothing
end


function get_cDixt(U, x::Real, param)
    # cDixt = cDi * x / th
    cDi, cDi_U = get_cDi(U, param)
    th = U[1]

    cDixt = cDi * x / th
    cDixt_U = (x / th) .* cDi_U
    cDixt_U[1] -= cDixt / th
    cDixt_x = cDi / th

    return cDixt, cDixt_U, cDixt_x
end

function solve_coupled!(M)
    # Newton loop
    nNewton = M.param.niglob
    M.glob.conv = false
    vprint(M.param, 1, "\n <<< Beginning coupled solver iterations >>> \n")

    for iNewton in 1:nNewton
        # assemble global residual/Jacobian for current state
        build_glob_sys!(M)

        # update forces/post (cl, cm, etc.) for diagnostics and trim
        calc_force!(M)

        # convergence check on residual
        Rnorm = norm(M.glob.R, 2)
        vprint(M.param, 1, @sprintf("\nNewton iteration %d, Rnorm = %.5e\n", iNewton, Rnorm))
        if Rnorm < M.param.rtol
            M.glob.conv = true
            break
        end

        # solve for state increment (dU, possibly dalpha)
        solve_glob!(M)

        # apply update with under-relaxation and guardrails
        update_state!(M)

        # move stagnation point (R_x effects already accounted in Jacobian)
        stagpoint_move!(M)

        # refresh lam/turb flags and transition bookkeeping
        update_transition!(M)
    end

    if !M.glob.conv
        vprint(M.param, 1, "\n** Global Newton NOT CONVERGED **\n")
    end

    return nothing
end

function jacobian_add_Rx!(M)
    # include effects of R_x into R_U: R_ue += R_x * x_st * st_ue
    Nsys = M.glob.Nsys
    Iue  = collect(4:4:4*Nsys)            # ue column indices in the global Jacobian

    # Sensitivity of node x to stagnation location: x_st (length Nsys)
    sgnue = M.isol.sgnue                   # length N (airfoil)
    T = eltype(sgnue)
    Nw = M.wake.N
    x_st = vcat(-sgnue, -ones(T, Nw))      # wake same sign as upper surface

    # R_st = R_x * x_st  (size: 3Nsys)
    R_st = M.glob.R_x * x_st

    # Columns to update correspond to ue at the two stag-adjacent nodes
    Ist    = M.isol.Istag                  # e.g., [i_lower, i_upper]
    st_ue  = collect(M.isol.sstag_ue)      # ensure a 2-element Vector

    # Rank-1 updates per column to keep it sparse-friendly
    
    M.glob.R_U[:, Iue[Ist[1]]] .+= R_st .* st_ue[1]
    M.glob.R_U[:, Iue[Ist[2]]] .+= R_st .* st_ue[2]


    return nothing
end

function clalpha_residual(M)
    Nsys  = M.glob.Nsys             # total nodes (airfoil + wake)
    N     = M.foil.N                # airfoil nodes only
    α     = M.oper.alpha            # degrees

    if M.oper.givencl
        # cl constraint residual
        Rcla = M.post.cl - M.oper.cltgt

        # Jacobian of cl w.r.t. [th,ds,sa,ue]... and extra var alpha at the end
        Rcla_U = zeros(Float64, 4*Nsys + 1)
        Rcla_U[end] = M.post.cl_alpha
        # only ue at AIRFOIL nodes contribute to cl directly
        for i in 1:N
            Rcla_U[4*i] = M.post.cl_ue[i]
        end

        # d/dα of uinv = ueinvref * [cos(α); sin(α)]  (α in deg)
        # => uinv_α = ueinvref * [-sin(α); cos(α)] * (π/180)
        θ = α * (pi/180)
        scale = (pi/180)
        Ru_alpha = -(get_ueinvref(M) * [-sin(θ); cos(θ)]) * scale  # length Nsys

    else
        # alpha prescribed: no cl constraint residual, alpha eqn is simply α-α0
        Rcla = 0.0
        Ru_alpha = zeros(Float64, Nsys)   # alpha not changing in this mode
        Rcla_U = zeros(Float64, 4*Nsys + 1)
        Rcla_U[end] = 1.0                 # residual for fixed-alpha equation
    end

    return Rcla, Ru_alpha, Rcla_U
end

function get_ueinvref(M)
    @assert !isempty(M.isol.gam) "No inviscid solution"

    N = M.foil.N

    # airfoil: elementwise sign on each row of gamref (N×2)
    uearef = M.isol.gamref .* reshape(Float64.(M.isol.sgnue), N, 1)

    # wake: use precomputed 0°/90° refs, and enforce continuity if viscous+wake
    if M.oper.viscous && M.wake.N > 0
        uewref = copy(M.isol.uewiref)      # (Nw×2)
        uewref[1, :] .= uearef[end, :]     # continuity at first wake node
    else
        uewref = Array{Float64}(undef, 0, 2)
    end

    return vcat(uearef, uewref)            # (N+Nw)×2
end

function solve_glob!(M)
    Nsys = M.glob.Nsys
    docl = M.oper.givencl ? 1 : 0

    # pull ue, ds as plain Vectors and guard ue against 0/negatives
    ue = collect(M.glob.U[4, :])
    ds = collect(M.glob.U[2, :])
    uemax = maximum(abs.(ue))
    if uemax == 0.0
        uemax = 1.0
    end
    ue = max.(ue, 1e-10 * uemax)

    # inviscid edge velocity
    ueinv = get_ueinv(M)  # length Nsys

    # global Jacobian (augmented by +1 row/col if cl-constraint is on)
    R_V = spzeros(Float64, 4*Nsys + docl, 4*Nsys + docl)

    # indices for ds and ue columns in the big state vector
    Ids = collect(2:4:4*Nsys)
    Iue = collect(4:4:4*Nsys)

    # include stagnation-location coupling into the ue columns of R_U
    jacobian_add_Rx!(M)

    # residual: [ primary 3Nsys residuals ; ue-equation residuals ]
    R = vcat(
        M.glob.R,
        ue .- (ueinv .+ M.vsol.ue_m * (ds .* ue))
    )

    # assemble Jacobian blocks
    # top-left 3Nsys x 4Nsys block
    R_V[1:3Nsys, 1:4Nsys] .= M.glob.R_U

    # ue-equation rows
    rowidx = (3Nsys + 1):(4Nsys)
    Dds = Diagonal(ds)
    Due = Diagonal(ue)
    Isp = spdiagm(0 => ones(Float64, Nsys))

    # d(ue-resid)/d(ue) and d(ue-resid)/d(ds)
    R_V[rowidx, Iue] .= Isp .- (M.vsol.ue_m * Dds)
    R_V[rowidx, Ids] .= -(M.vsol.ue_m * Due)

    if docl == 1
        # cl-constraint residual & Jacobian wrt alpha
        Rcla, Ru_alpha, Rcla_U = clalpha_residual(M)
        R = vcat(R, Rcla)
        R_V[rowidx, 4*Nsys + 1] .= Ru_alpha
        R_V[4*Nsys + 1, :]      .= Rcla_U
    end

    # solve the linear system for update
    dV = -(R_V \ R)

    # stash updates
    M.glob.dU = reshape(dV[1:4*Nsys], 4, Nsys)
    if docl == 1
        M.glob.dalpha = dV[end]
    end

    return nothing
end

function rebuild_isol!(M)
    @assert !isempty(M.isol.gam) "No inviscid solution to rebuild"
    vprint(M.param, 2, "\n  Rebuilding the inviscid solution.\n")

    α = M.oper.alpha
    # Combine the 0°/90° reference solutions for the current alpha
    M.isol.gam = M.isol.gamref * [cosd(α); sind(α)]

    if !M.oper.viscous
        # In inviscid mode, update stagnation point here
        stagpoint_find!(M)
    elseif M.oper.redowake
        # In viscous mode, optionally rebuild wake + sensitivities if AoA changed
        build_wake!(M)
        identify_surfaces!(M)
        calc_ue_m!(M)   # rebuild matrices due to changed wake geometry
    end

    return nothing
end

function update_state!(M)
    # guard against accidental complex numbers in amp/ctau row
    if any(!isreal, M.glob.U[3, :])
        error("imaginary amp in U")
    end
    if any(!isreal, M.glob.dU[3, :])
        error("imaginary amp in dU")
    end

    # max ctau over turbulent nodes (might be empty)
    It = findall(M.vsol.turb .!= 0)
    ctmax = isempty(It) ? 0.0 : maximum(M.glob.U[3, It])

    # start with full step
    omega = 1.0

    # 1) limit decreases in θ and δ*
    for k in 1:2
        Uk  = M.glob.U[k, :]
        dUk = M.glob.dU[k, :]
        r   = dUk ./ Uk
        fmin = minimum(r)                      # most negative ratio
        om   = (fmin < -0.5) ? abs(0.5 / fmin) : 1.0
        if om < omega
            omega = om
            vprint(M.param, 3, @sprintf("  th/ds decrease: omega = %.5f\n", omega))
        end
    end

    # 2) prevent negative amp/ctau after update
    Uk  = M.glob.U[3, :]
    dUk = M.glob.dU[3, :]
    for i in eachindex(Uk)
        if (!M.vsol.turb[i]) && (Uk[i] < 0.2);           continue; end
        if ( M.vsol.turb[i]) && (Uk[i] < 0.1*ctmax);      continue; end
        if (Uk[i] == 0.0) || (dUk[i] == 0.0);            continue; end
        if Uk[i] + dUk[i] < 0.0
            om = 0.8 * abs(Uk[i] / dUk[i])
            if om < omega
                omega = om
                vprint(M.param, 3, @sprintf("  neg sa: omega = %.5f\n", omega))
            end
        end
    end

    # 3) prevent big changes in laminar amplification
    I_lam = findall(.! M.vsol.turb)
    if any(!isreal, Uk[I_lam])
        error("imaginary amplification")
    end
    dumax = isempty(I_lam) ? 0.0 : maximum(abs.(dUk[I_lam]))
    om = (dumax > 0) ? abs(2.0 / dumax) : 1.0
    if om < omega
        omega = om
        vprint(M.param, 3, @sprintf("  amp: omega = %.5f\n", omega))
    end

    # 4) prevent big changes in turbulent cτ
    I_turb = It
    dumax = isempty(I_turb) ? 0.0 : maximum(abs.(dUk[I_turb]))
    om = (dumax > 0) ? abs(0.05 / dumax) : 1.0
    if om < omega
        omega = om
        vprint(M.param, 3, @sprintf("  ctau: omega = %.5f\n", omega))
    end

    # 5) prevent large ue changes
    dUe = M.glob.dU[4, :]
    Vinf = M.oper.Vinf
    denom = (Vinf == 0.0) ? eps(Float64) : Vinf
    fmax = maximum(abs.(dUe)) / denom
    om = (fmax > 0) ? (0.2 / fmax) : 1.0
    if om < omega
        omega = om
        vprint(M.param, 3, @sprintf("  ue: omega = %.5f\n", omega))
    end

    # 6) prevent large alpha changes
    dα = getfield(M.glob, :dalpha)   # assumes field exists; else set to 0 upstream
    if abs(dα) > 2
        omega = min(omega, abs(2 / dα))
    end

    # take the update
    vprint(M.param, 2, @sprintf("  state update: under-relaxation = %.5f\n", omega))
    @. M.glob.U = M.glob.U + omega * M.glob.dU
    M.oper.alpha += omega * dα

    # fix bad Hk after the update
    for is in 1:3
        Hkmin = (is == 3) ? 1.00005 : 1.02
        Is = M.vsol.Is[is]
        param = build_param(M, is)
        for j in Is
            Uj = @view M.glob.U[:, j]
            param = station_param(M, param, j)
            Hk, _ = get_Hk(Uj, param)
            if Hk < Hkmin
                M.glob.U[2, j] += 2 * (Hkmin - Hk) * M.glob.U[1, j]
            end
        end
    end

    # fix negative ctau after the update (only on turbulent nodes)
    for i in It
        if M.glob.U[3, i] < 0.0
            M.glob.U[3, i] = 0.1 * ctmax
        end
    end

    # rebuild inviscid solution (γ, wake) if α moved
    if abs(omega * dα) > 1e-10
        rebuild_isol!(M)
    end

    return nothing
end

function update_transition!(M)
    # updates: M.vsol.turb flags and M.glob.U[3, :] (amp or √ctau) where needed
    for is in 1:2  # lower / upper only
        Is = M.vsol.Is[is]                 # surface node indices
        N  = length(Is)
        xft = M.oper.xft[is] * M.geom.chord

        # build per-side params
        param = build_param(M, is)

        # current last laminar station (local index along Is)
        lam_mask = M.vsol.turb[Is] .== 0
        I_lam = findall(lam_mask)
        ilam0 = isempty(I_lam) ? 1 : I_lam[end]

        # save current amp/ctau row so we can restore if nothing changes
        sa_saved = copy(@view M.glob.U[3, Is])

        # march amplification to get new last-laminar local index
        ilam = march_amplification(M, is)

        # if forced transition, set xi value at the forced point
        if M.oper.forcet[is]
            i0 = Is[ilam]
            i1 = Is[min(ilam + 1, N)]
            xi0, xi1 = M.isol.xi[i0], M.isol.xi[i1]
            x0,  x1  = M.foil.x[1, i0], M.foil.x[1, i1]
            denom = (x1 - x0)
            M.oper.xift[is] = xi0 + (denom == 0 ? 0.0 : (xi1 - xi0) * (xft - x0) / denom)
        end

        # no change? restore and continue
        if ilam == ilam0
            M.glob.U[3, Is] .= sa_saved
            continue
        end

        vprint(M.param, 2, @sprintf("  Update transition: last lam [%d]->[%d]\n", ilam0, ilam))

        if ilam < ilam0
            # transition moved earlier: fill newly turbulent band [ilam+1 : ilam0]
            param.turb = true
            # shear stress at transition start
            sa0 = first(get_cttr( M.glob.U[:, Is[ilam+1]], param))
            # target at end of band (use existing next-turb value if available)
            sa1 = (ilam0 < N) ? M.glob.U[3, Is[ilam0+1]] : sa0

            xi = M.isol.xi[Is]
            dx = xi[min(ilam0+1, N)] - xi[ilam+1]

            for i in (ilam+1):ilam0
                f = if dx == 0.0 || i == ilam+1
                    0.0
                else
                    (xi[i] - xi[ilam+1]) / dx
                end
                if (ilam + 1) == ilam0
                    f = 1.0
                end
                val = sa0 + f * (sa1 - sa0)
                @assert val > 0.0 "negative ctau in update_transition"
                M.glob.U[3, Is[i]] = val
                M.vsol.turb[Is[i]] = 1
            end

        elseif ilam > ilam0
            # transition moved later: mark [ilam0 : ilam] laminar; leave values alone
            for i in ilam0:ilam
                M.vsol.turb[Is[i]] = 0
            end
        end
    end

    return nothing
end

function get_distributions!(M::MfoilTypes.Mfoil)
    @assert size(M.glob.U, 2) > 0 "no global solution"

    # pull basics straight from the global state
    Nsys = M.glob.Nsys
    M.post.th  = vec(M.glob.U[1, :])             # θ
    M.post.ds  = vec(M.glob.U[2, :])             # δ*
    M.post.sa  = vec(M.glob.U[3, :])             # amp or √ctau
    M.post.uei = get_ueinv(M)                    # inviscid edge speed (airfoil + wake)

    # compressibility-corrected edge speed uk(ue)
    u_edge = vec(M.glob.U[4, :])
    ue_corr = similar(u_edge)
    @inbounds for i in eachindex(u_edge)
        ue_corr[i], _ = get_uk(u_edge[i], M.param)
    end
    M.post.ue = ue_corr

    # derived viscous quantities (free-stream based cf, Ret, Hk)
    cf  = zeros(Float64, Nsys)
    Ret = zeros(Float64, Nsys)
    Hk  = zeros(Float64, Nsys)

    for is in 1:3                       # 1=lower, 2=upper, 3=wake
        Is = M.vsol.Is[is]
        param = build_param(M, is)
        for j in Is
            param = station_param(M, param, j)
            Uj = @view M.glob.U[:, j]

            uk, _     = get_uk(Uj[4], param)     # corrected edge speed
            cfloc, _  = get_cf(Uj, param)        # local skin-friction coeff (edge-based)
            Retj, _   = get_Ret(Uj, param)       # Re_θ
            Hkj, _    = get_Hk(Uj, param)        # kinematic shape factor

            # convert to freestream-based cf
            cf[j]  = cfloc * (uk / param.Vinf)^2
            Ret[j] = Retj
            Hk[j]  = Hkj
        end
    end

    M.post.cf  = cf
    M.post.Ret = Ret
    M.post.Hk  = Hk

    return nothing
end


function solve_viscous!(M) 

    enforce_CW_and_TE!(M)
    solve_inviscid!(M)
    M.oper.viscous = true
    init_thermo!(M)
    build_wake!(M)
end

using MAT
function update_MATLAB!(M)
    vars = matread("scripts/InitGeom.mat")
    M.foil.N = vars["Np"]
    M.foil.x = vars["xp"]
    M.foil.s = vec(vars["sp"])
    M.foil.t = vars["tp"]
    M.wake.N = vars["Nw"]
    M.wake.x = vars["xw"]
    M.wake.s = vec(vars["sw"])
    M.wake.t = vars["tw"]
    M.isol.uewi = vec(vars["uewi"])
    M.isol.uewiref = vars["uewiref"]
    return nothing
    
end

M = init_M(5)

update_MATLAB!(M)

enforce_CW_and_TE!(M)
solve_inviscid!(M)
M.oper.viscous = true
init_thermo!(M)
#build_wake!(M)
stagpoint_find!(M)
identify_surfaces!(M)
set_wake_gap!(M)
calc_ue_m!(M)
init_boundary_layer!(M)
stagpoint_move!(M)
solve_coupled!(M)
calc_force!(M)
get_distributions!(M)




using Plots

plt = plot(aspect_ratio=:equal, legend=false, xlabel="x", ylabel="y")

# gather geometry & states
xy = hcat(M.foil.x, M.wake.x)          # 2×(N+Nw)
t  = hcat(M.foil.t, M.wake.t)          # 2×(N+Nw)
N  = M.foil.N
Nw = M.wake.N
ds = M.post.ds                         # length N+Nw

# outward unit normals from tangents
n = hcat(-t[2, :],t[1, :])
n .= n ./ sqrt.(sum(n.^2, dims=1))     # normalize by column

# split factors at TE for wake envelopes
rl, ru = 0.0, 1.0
if Nw > 0
    rl = 0.5 * (1 + (ds[1] - ds[N]) / ds[N+1])
    ru = 1 - rl
end

plot!(xy[1,:], xz[2,:],c=:black)



# full δ* envelope along lower/upper
xyd_full = xy .+ n' .* reshape(ds, 1, :)
colors = Dict(1=>:red, 2=>:blue, 3=>:black)

for is in 1:2
    Is = M.vsol.Is[is]
    plot!(xyd_full[1, Is], xyd_full[2, Is]; color=colors[is], linewidth=2)
end

# wake: draw both edges if present
if Nw > 0
    Isw = M.vsol.Is[3]
    # upper wake edge (+ru)
    xyd = xy .+ n' .* reshape(ds .* ru, 1, :)
    plot!(xyd[1, Isw], xyd[2, Isw]; color=colors[3], linewidth=2)
    # lower wake edge (−rl)
    xyd = xy .- n' .* reshape(ds .* rl, 1, :)
    plot!(plt, xyd[1, Isw], xyd[2, Isw]; color=colors[3], linewidth=2)
end

display(plt)