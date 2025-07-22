module Streamlines

using Interpolations: interpolate, Gridded, Linear
using Random: shuffle!

export streamlines, streamlines_from_grid

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
            #==
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
            ==#
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
                push!(vertsout, [NaN NaN]) # Separator
            end
        end
    end

    isempty(vertsout) && return zeros(0,2), Int[]
    # Remove last NaN separator
    return vcat(vertsout[1:end-1]...)
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

    xy = _get_stream_xy(xg, yg, uu, vv, xc, yc, min_density, max_density)
    return xy
end


end 