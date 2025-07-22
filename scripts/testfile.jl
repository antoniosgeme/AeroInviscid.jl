using AeroGeometry
using AeroInviscid
using Plots

airfoil = Airfoil("NACA6409")

prob = InviscidProblem(airfoil,14)

sol = solve(prob)
flowplot(sol,clim=(-8,2),min_density=2,max_density=4)


plot(sol)

alphas = 0:1:14
probs = InviscidProblem.(Ref(airfoil),alphas)
sols = solve.(probs);

plot(alphas,[sol.cl for sol in sols])
plot!(alphas,2π*deg2rad.(alphas),ls=:dash)
flowplot(sol,clim=(-5,5))


x_vec = -1:0.01:1.5
y_vec = -1:0.01:1

U,V = induced_velocity(sol,x_vec,y_vec)



# 3. Call the even_stream_data function
# We'll set compute_dist=false for this example to keep it simple.
xy = streamlines(x_vec, y_vec, U', V',min_density=2,max_density=5)


cp = @. 1 - U^2 - V^2
contourf!(x_vec,y_vec,cp',lw=0,color=:RdBu,levels=30,clim=(-5, 2))
plot(xy[:, 1], xy[:, 2],
    title="Evenly Spaced Streamlines of a Vortex Field",
    xlabel="X",
    ylabel="Y",
    legend=false,
    aspect_ratio=:equal,
    linewidth=1.5
)

plot!(airfoil.x,airfoil.y,seriestype=:shape,fill=:black)





xy = streamlines_from_grid(x_vec, y_vec, U', V',density=0.8)






airfoil1 = Airfoil("NACA0012")
airfoil2 = Airfoil("naca6409")
airfoil3 = Airfoil("s3024")

multielement = MultielementAirfoil([airfoil1,airfoil2,airfoil3],[25,20,0],[1,1,1],[[0,0],[1.2,0],[0,-0.75]])
prob = InviscidProblem(multielement,0)
sol = solve(prob)
plot(sol)
flowplot(sol,xlims=(-1,3),clims=(-10,10),ylims=(-1,1))


x_vec = -0.5:0.01:3
y_vec = -1:0.01:1
U,V = induced_velocity(sol,x_vec,y_vec)
xy = streamlines(x_vec, y_vec, U', V',min_density=1,max_density=3)

p = plot(xy[:, 1], xy[:, 2],
    title="Evenly Spaced Streamlines of a Vortex Field",
    xlabel="X",
    ylabel="Y",
    legend=false,
    aspect_ratio=:equal,
    linewidth=1.5,
    size=(600, 600)
)

for (i,airfoil) in enumerate(multielement.airfoils)
    xy_vort = shift_scale_rotate.(airfoil.x,airfoil.y,
                                        multielement.le_loc[i][1],multielement.le_loc[i][2],
                                        multielement.pitch[i],multielement.chord[i]) 
    x_sheet  = [xy[1] for xy in xy_vort]
    y_sheet  = [xy[2] for xy in xy_vort]
    plot!(x_sheet,y_sheet,seriestype=:shape,fill=:black)
end 
display(p)
flowplot(sol)



airplane = cessna152()


gr()
plot(airplane)

wing = airplane.wings[1]
# Wing Discretization + VLM Skeleton
using Interpolations, LinearAlgebra

"""
    discretize_wing(wing::Wing; n_chord::Int=10, n_span::Int=20)

Return arrays X, Y, Z of size (n_chord+1)×(n_span+1) of mesh‐points sampling
camberline via `camber`.
"""
function discretize_wing(wing::Wing; n_chord::Int=10, n_span::Int=20)
    X = zeros(n_chord+1, n_span+1)
    Y = similar(X)
    Z = similar(X)
    y_le = [sec.le_loc[2] for sec in wing.xsecs]

    for j in 1:n_span+1
        η = (j-1)/n_span
        yj = (1-η)*y_le[1] + η*y_le[end]
        idx = searchsortedfirst(y_le, yj)
        i_lo = clamp(idx-1, 1, endof(y_le))
        i_hi = clamp(idx,   1, endof(y_le))
        t = (i_lo==i_hi) ? 0.0 : (yj-y_le[i_lo])/(y_le[i_hi]-y_le[i_lo])
        sec_lo, sec_hi = wing.xsecs[i_lo], wing.xsecs[i_hi]
        chord = (1-t)*sec_lo.chord + t*sec_hi.chord
        twist = deg2rad((1-t)*sec_lo.twist + t*sec_hi.twist)
        le_loc = (1-t)*sec_lo.le_loc .+ t*sec_hi.le_loc

        for i in 1:n_chord+1
            λ = (i-1)/n_chord
            x_loc = λ*chord
            z_cam = camber(sec_lo.airfoil; x_over_c=[λ])[1]*chord
            x_rot =  cos(twist)*x_loc - sin(twist)*z_cam
            z_rot =  sin(twist)*x_loc + cos(twist)*z_cam
            X[i,j] = le_loc[1] + x_rot
            Y[i,j] = le_loc[2]
            Z[i,j] = le_loc[3] + z_rot
        end
    end
    return X, Y, Z
end

"""
    build_panels(X,Y,Z)

From corner grids X,Y,Z returns:
- P1,P2,P3,P4: arrays of panel corner coordinates
- coll_pts: collocation points at 3/4 chord
- normals: unit normal vector per panel
- vortex_ends: horseshoe vortex endpoints (A,B,C,D) per panel
"""
function build_panels(X, Y, Z)
    nc, ns = size(X)
    np_ch = nc-1; np_sp = ns-1; Np = np_ch*np_sp
    P1 = zeros(3, Np); P2 = similar(P1); P3 = similar(P1); P4 = similar(P1)
    coll = similar(P1); normals = similar(P1)
    A = similar(P1); B = similar(P1); C = similar(P1); D = similar(P1)

    idx = 1
    for j in 1:np_sp, i in 1:np_ch
        p1 = [X[i,j],   Y[i,j],   Z[i,j]]
        p2 = [X[i+1,j], Y[i+1,j], Z[i+1,j]]
        p3 = [X[i+1,j+1], Y[i+1,j+1], Z[i+1,j+1]]
        p4 = [X[i,j+1],   Y[i,j+1],   Z[i,j+1]]
        P1[:,idx],P2[:,idx],P3[:,idx],P4[:,idx] = p1,p2,p3,p4
        coll[:,idx] = 0.25*(p1 + 2p4 + p3)
        normals[:,idx] = normalize(cross(p2-p1, p4-p1))
        A[:,idx] = 0.75*(p1 + p4)
        B[:,idx] = 0.75*(p2 + p3)
        wake_dir = [1.0, 0.0, 0.0]
        C[:,idx] = B[:,idx] + 10*wake_dir
        D[:,idx] = A[:,idx] + 10*wake_dir
        idx += 1
    end
    return (P1,P2,P3,P4,coll,normals,A,B,C,D)
end

"""
    compute_influence(coll, normals, A,B,C,D)
"""
function compute_influence(coll, normals, A, B, C, D)
    Np = size(coll,2)
    M = zeros(Np, Np)
    for i in 1:Np, j in 1:Np
        V = horseshoe_velocity(coll[:,i], A[:,j], B[:,j], C[:,j], D[:,j])
        M[i,j] = dot(V, normals[:,i])
    end
    return M
end

"""
    solve_gamma(infmat, rhs)
"""
function solve_gamma(A, rhs)
    return A \ rhs
end

# === Biot–Savart for Straight Vortex Segments ===
ε = 1e-8

"""
    vortex_segment(P, R1, R2)

Induced velocity at P from a unit-strength vortex filament from R1→R2.
"""
function vortex_segment(P::AbstractVector, R1::AbstractVector, R2::AbstractVector)
    r1 = P .- R1
    r2 = P .- R2
    r0 = R2 .- R1
    cr = cross(r1, r2)
    cr2 = dot(cr, cr)
    norm_r1 = norm(r1)
    norm_r2 = norm(r2)
    if cr2 < ε || norm_r1 < ε || norm_r2 < ε
        return zeros(3)
    end
    coeff = dot(r0, r1/norm_r1 .- r2/norm_r2) / cr2
    return (1/(4π)) * cr * coeff
end

"""
    horseshoe_velocity(P, A, B, C, D)

Sum induced velocity from bound segment A→B and wake legs B→C and D→A.
"""
function horseshoe_velocity(P::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, D::AbstractVector)
    v = vortex_segment(P, A, B)
    v .+= vortex_segment(P, B, C)
    v .+= vortex_segment(P, D, A)
    return v
end

"""
Workflow:
1. X,Y,Z = discretize_wing(wing)
2. P1,P2,P3,P4,coll,normals,A,B,C,D = build_panels(X,Y,Z)
3. inf = compute_influence(coll,normals,A,B,C,D)
4. rhs = -dot(V∞, normals)
5. γ = solve_gamma(inf, rhs)
6. compute forces from γ.
"""
X,Y,Z = discretize_wing(wing)
P1,P2,P3,P4,coll,normals,A,B,C,D = build_panels(X,Y,Z)
X, Y, Z = discretize_wing(wing)
plot(wing)
plotly()
Plots.scatter!(X, Y, Z,c=:red,ms=2)
