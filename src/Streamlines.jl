module Streamlines

using Interpolations: interpolate, Gridded, Linear
using Random: shuffle!
using NearestNeighbors: KDTree, nn

export streamlines, streamlines_from_grid

# Helper: MATLAB-style meshgrid
function meshgrid(x::AbstractVector, y::AbstractVector)
    X = [i for j in y, i in x]
    Y = [j for j in y, i in x]
    return X, Y
end

# Safe index computation: returns clamped Int or nothing if NaN/Inf
function safecell(val::Real, factor::Real, N::Integer)
    v = val * factor
    isfinite(v) || return nothing
    return clamp(floor(Int, v) + 1, 1, N)
end

# Trace a single streamline using RK2
function _trace(
    x0::Real, y0::Real,
    itp_u, itp_v,
    stepsize::Real, maxvert::Integer,
    xmin::Real, xmax::Real,
    ymin::Real, ymax::Real,
    sign::Integer
)
    verts = Vector{Float64}[]
    push!(verts, [x0, y0])
    x, y = x0, y0
    for _ in 1:maxvert
        if !(xmin <= x <= xmax && ymin <= y <= ymax)
            break
        end
        u1 = itp_u(y, x); v1 = itp_v(y, x)
        x_mid = x + sign * u1 * stepsize/2
        y_mid = y + sign * v1 * stepsize/2
        if !(xmin <= x_mid <= xmax && ymin <= y_mid <= ymax)
            break
        end
        u2 = itp_u(y_mid, x_mid); v2 = itp_v(y_mid, x_mid)
        x += sign * u2 * stepsize
        y += sign * v2 * stepsize
        push!(verts, [x, y])
    end
    return hcat(verts...)'
end

# Corrected version of the main streamline generation function
function _get_stream_xy(x, y, u, v, x_coords, y_coords, min_density, max_density)
    num     = 20
    nrstart = ceil(Int, num * min_density)
    ncstart = ceil(Int, num * min_density)
    nrend   = ceil(Int, num * max_density)
    ncend   = ceil(Int, num * max_density)

    xmin, xmax = minimum(x), maximum(x)
    ymin, ymax = minimum(y), maximum(y)
    xrange     = xmax - xmin
    yrange     = ymax - ymin

    incstartx = xrange / ncstart
    incstarty = yrange / nrstart
    ixrangecs = ncstart / xrange * (1 - eps())
    iyrangers = nrstart / yrange * (1 - eps())
    ixrangece = ncend   / xrange * (1 - eps())
    iyrangere = nrend   / yrange * (1 - eps())

    stepsize = min(xrange / (ncend * 2), yrange / (nrend * 2), 0.1) # A more robust stepsize
    maxvert  = min(10000, round(Int, sum(size(v)) * 4 / stepsize))

    startgrid = zeros(Bool, nrstart, ncstart)
    endgrid   = zeros(Bool, nrend,   ncend)
    rc_list   = collect(vec(CartesianIndices((1:nrstart,1:ncstart))))
    shuffle!(rc_list)

    # Continuous interpolators
    itp_u = interpolate((y_coords, x_coords), u, Gridded(Linear()))
    itp_v = interpolate((y_coords, x_coords), v, Gridded(Linear()))

    vertsout = Vector{Array{Float64,2}}()
    lengths  = Int[]

    for idx in rc_list
        r, c = idx.I
        if !startgrid[r, c]
            x0 = xmin + (c - 0.5) * incstartx
            y0 = ymin + (r - 0.5) * incstarty

            # Correctly trace forward and backward
            vf = _trace(x0, y0, itp_u, itp_v, stepsize, maxvert, xmin, xmax, ymin, ymax,  1)
            vb = _trace(x0, y0, itp_u, itp_v, stepsize, maxvert, xmin, xmax, ymin, ymax, -1)

            # Stitch into a single candidate streamline
            candidate = vcat(reverse(vb, dims=1), vf[2:end, :])
            if size(candidate, 1) < 2
                continue
            end

            # Check this candidate for collisions and truncate
            break_k = size(candidate, 1) + 1
            for k in 1:size(candidate, 1)
                xi = candidate[k, 1] - xmin
                yi = candidate[k, 2] - ymin

                re = safecell(yi, iyrangere, nrend)
                ce = safecell(xi, ixrangece, ncend)

                if ce === nothing || re === nothing || endgrid[re, ce]
                    break_k = k
                    break
                end
            end
            final_streamline = candidate[1:break_k-1, :]

            # Now, update grids with the final, accepted streamline
            if size(final_streamline, 1) > 1 # Only add if it has some length
                for k in 1:size(final_streamline, 1)
                    xi = final_streamline[k, 1] - xmin
                    yi = final_streamline[k, 2] - ymin

                    sr = safecell(yi, iyrangers, nrstart)
                    sc = safecell(xi, ixrangecs, ncstart)
                    sr !== nothing && sc !== nothing && (startgrid[sr, sc] = true)

                    re = safecell(yi, iyrangere, nrend)
                    ce = safecell(xi, ixrangece, ncend)
                    re !== nothing && ce !== nothing && (endgrid[re, ce] = true)
                end
                push!(vertsout, final_streamline)
                push!(lengths, size(final_streamline, 1))
                push!(vertsout, [NaN NaN]) # Separator
            end
        end
    end

    isempty(vertsout) && return zeros(0,2), Int[]
    # Remove last NaN separator
    return vcat(vertsout[1:end-1]...), lengths
end

# Compute nearest-neighbor distances per point
function _get_stream_dist(xy, lengths)
    isempty(lengths) && return Float64[]
    pts = xy[.!isnan.(xy[:, 1]), :]
    d   = fill(NaN, size(xy, 1))
    pi = 1; xi = 1
    for L in lengths
        if L == 0
            xi += 1
            continue
        end
        inds = pi:pi+L-1
        mask = trues(size(pts, 1)); mask[inds] .= false
        others = pts[mask, :]
        if isempty(others)
            d[xi:xi+L-1] .= Inf
        else
            tree = KDTree(others')
            _, dd = nn(tree, pts[inds, :]' )
            d[xi:xi+L-1] = dd
        end
        pi += L; xi += L+1
    end
    return d
end

"""
    streamlines(xx, yy, uu, vv;
                min_density::Real=1.0,
                max_density::Real=0.5)

Compute evenly-spaced streamlines on the grid `(xx,yy)` with vector field `(uu,vv)`.
If `compute_dist` is true, also return nearest-neighbor distances.
"""
function streamlines(xx, yy, uu, vv;
                     min_density::Real=1.0,
                     max_density::Real=5)

    
    if xx isa AbstractVector && yy isa AbstractVector
        xc, yc = xx, yy
        xg, yg = meshgrid(xc, yc)
    else
        xc = xx[1, :]; yc = yy[:, 1]
        xg, yg = xx, yy
    end
    @assert size(xg) == size(yg) == size(uu) == size(vv)
    @assert min_density > 0 && max_density > 0

    xy, lens = _get_stream_xy(xg, yg, uu, vv, xc, yc, min_density, max_density)
    return xy
end


"""
    streamlines_from_grid(xx, yy, uu, vv; density::Real=1.0)

Generates streamlines from a simple, uniform grid of seed points without any
collision detection or removal.

This function is much faster than the even-spacing algorithm. It seeds streamlines
from a uniform grid and traces them until they leave the domain. The `density`
parameter controls how many seed points are used.

# Arguments
- `xx`, `yy`: Coordinate grids (can be `Vector`s or `Matrix`/`Matrix`).
- `uu`, `vv`: Vector field components on the grid.
- `density::Real`: A factor to control the number of seed points. `density=1.0`
  corresponds to a 20x20 seed grid. `density=2.0` would be 40x40.

# Returns
- An `(N, 2)` `Matrix` of vertex coordinates, with individual streamlines separated
  by a row of `[NaN NaN]`.
"""
function streamlines_from_grid(xx, yy, uu, vv; density::Real = 1.0)
    # 1. Setup coordinates and domain boundaries
    if xx isa AbstractVector && yy isa AbstractVector
        x_coords, y_coords = xx, yy
    else
        x_coords = xx[1, :]
        y_coords = yy[:, 1]
    end
    @assert size(uu) == (length(y_coords), length(x_coords)) "Dimension mismatch between vector field and coordinates."
    @assert density > 0 "Density must be positive."
    
    
    xmin, xmax = minimum(x_coords), maximum(x_coords)
    ymin, ymax = minimum(y_coords), maximum(y_coords)
    xrange = xmax - xmin
    yrange = ymax - ymin

    # 2. Create the vector field interpolators
    itp_u = interpolate((y_coords, x_coords), uu, Gridded(Linear()))
    itp_v = interpolate((y_coords, x_coords), vv, Gridded(Linear()))

    # 3. Define the grid of seed points based on density
    nx = round(Int, 20 * density)
    ny = round(Int, 20 * density)
    x_seeds = range(xmin, stop=xmax, length=nx)
    y_seeds = range(ymin, stop=ymax, length=ny)

    # 4. Define tracing parameters
    stepsize = min(xrange / (nx * 2), yrange / (ny * 2))
    maxvert  = round(Int, (xrange + yrange) / stepsize * 2)

    vertsout = Vector{Matrix{Float64}}()

    # 5. Loop through every seed point and trace a streamline
    for y0 in y_seeds, x0 in x_seeds
        # Trace forward and backward from the seed point
        vf = _trace(x0, y0, itp_u, itp_v, stepsize, maxvert, xmin, xmax, ymin, ymax,  1)
        vb = _trace(x0, y0, itp_u, itp_v, stepsize, maxvert, xmin, xmax, ymin, ymax, -1)

        # Stitch the backward and forward traces together
        line = vcat(reverse(vb, dims=1), vf[2:end, :])

        if size(line, 1) > 1
            push!(vertsout, line)
            push!(vertsout, [NaN NaN]) # Add separator for plotting
        end
    end

    # 6. Combine all streamlines into a single matrix for output
    if isempty(vertsout)
        return zeros(0, 2)
    end
    
    # Remove the last NaN separator before concatenating
    return vcat(vertsout[1:end-1]...)
end

end # module Streamlines
