using AeroGeometry
using Plots
plotly()
using Interpolations
using LinearAlgebra


wing = cessna152().wings[1]


chordwise_num = 10
spanwise_num = 10

points, lines, section_num = AeroGeometry.mesh(wing; camberline=true)




# Biot-Savart law for a finite vortex segment of unit strength
function biot_segment(P::AbstractVector{T},A::AbstractVector{T},B::AbstractVector{T}) where T<:Real
    r1 = P - A
    r2 = P - B
    r0 = B - A
    cr = cross(r1, r2)
    denom = norm(cr)^2
    factor = dot(r0, (r1/norm(r1) .- r2/norm(r2))) / (4π * denom)
    return cr * factor
end

# Induced velocity from a horseshoe vortex of unit strength at point P
function horseshoe_influence(P::AbstractVector{T}, A::AbstractVector{T}, B::AbstractVector{T}; L::Float64=1e3) where T<:Real
    # bound segment A->B
    v_bound = biot_segment(P, A, B)
    # trailing legs parallel to +x
    xhat = [1.0, 0.0, 0.0]
    A_inf = A + L*xhat
    B_inf = B + L*xhat
    # trailing leg from B->B_inf
    v_trailB = biot_segment(P, B, B_inf)
    # trailing leg from A_inf->A
    v_trailA = biot_segment(P, A_inf, A)
    return v_bound + v_trailA + v_trailB
end

"""
    lifting_line(wing; spanwise=10, V∞=1.0, α=0.0)

Compute Prandtl lifting line solution for a given wing.

# Keywords
- `spanwise`: number of spanwise panels (default 10)
- `V∞`: freestream velocity magnitude (default 1.0)
- `α`: angle of attack in radians (default 0.0)

# Returns
- `y_cp`: spanwise locations of collocation points (length `spanwise`)
- `γ`: circulation strength at each panel (length `spanwise`)
"""
function lifting_line(wing; spanwise::Int=10, V∞::Float64=1.0, α::Float64=0.0)
    # 1) Mesh the wing camberline
    pts, _, sec = AeroGeometry.mesh(wing; camberline=true)

    # 2) Compute quarter-chord points and chord vectors per section
    sections = unique(sec)
    Nsec = length(sections)
    Q = zeros(3,Nsec)
    Q3 = zeros(3,Nsec)
    chords = zeros(3,Nsec)
    for (i, s) in enumerate(sections)
        idx = findall(sec .== s)
        sp = pts[idx, :]
        xs = sp[:,1]; ys = sp[:,2]; zs = sp[:,3]
        # Leading and trailing edge
        LE = sp[argmin(xs), :]
        TE = sp[argmax(xs), :]
        chords[:,i] = TE - LE
        # Quarter-chord reference point
        qc = LE .+ 0.25*(TE - LE)
        q3c = LE .+ 0.75*(TE - LE)
        # Nearest camberline sample
        d = [norm(sp[j,:] - qc) for j in eachindex(xs)]
        Q[:,i] = sp[argmin(d), :]
        d = [norm(sp[j,:] - q3c) for j in eachindex(xs)]
        Q3[:,i] = sp[argmin(d), :]
    end

    sortidx = sortperm([q[2] for q in eachcol(Q)])
    y_sec = Q[2, sortidx]
    x_sec = Q[1, sortidx]
    z_sec = Q[3, sortidx]

    y_nodes = range(y_sec[1], stop=y_sec[end], length=spanwise+1)
    itp_x = linear_interpolation(y_sec, x_sec)
    itp_z = linear_interpolation(y_sec, z_sec)

    bound = hcat(itp_x(y_nodes), y_nodes, itp_z(y_nodes))


    sortidx = sortperm([q[2] for q in eachcol(Q3)])
    y_sec = Q3[2, sortidx]
    x_sec = Q3[1, sortidx]
    z_sec = Q3[3, sortidx]

    y_nodes = range(y_sec[1], stop=y_sec[end], length=spanwise+1)
    itp_x = linear_interpolation(y_sec, x_sec)
    itp_z = linear_interpolation(y_sec, z_sec)

    coll = hcat(itp_x(y_nodes), y_nodes, itp_z(y_nodes))

    # 5) Assemble influence matrix
    N = spanwise
    A = zeros(N, N)
    for i in 1:N
        P = coll[i]
        for j in 1:N
            A[i,j] = horseshoe_influence(P, bound[j], bound[j+1])[3]
        end
    end

    # 6) Solve for circulation
    b = -V∞ * sin(α) * ones(N)
    γ = A \ b

    # 7) Return collocation spanwise stations and gamma
    y_cp = [(y_nodes[i] + y_nodes[i+1]) / 2 for i in 1:N]
    return y_cp, γ
end





