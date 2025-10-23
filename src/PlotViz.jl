using LaTeXStrings
using RecipesBase
using PlotUtils
using PrettyStreamlines

@recipe function f(sol::InviscidSolution{A,S};
    ) where {A<:Airfoil,S<:LinearVortex}

    af   = sol.geometry
    x    = af.x; 
    cp   = sol.cp

    yflip      := true
    xlabel     := L"\frac{x}{c}"
    ylabel     := L"C_P"
    legend     := nothing
    lw         := 3
    grid --> true  # Enable grid
    gridlinewidth --> 3
    gridstyle --> :dash
    gridcolor --> :white  # Set white grid lines
    
    # Adjust minor grid visibility
    minorgrid --> true
    minorgridcolor --> :white
    minorgridlinewidth --> 0.4
    minorgridstyle --> :dot
    bg             --> :black
    return x, cp
end


@recipe function f(sol::InviscidSolution{M,S};
    ) where {M<:MultielementAirfoil{T},S<:LinearVortex} where T

    multielement   = sol.geometry
    le_loc = multielement.le_loc
    pitch = multielement.pitch
    chord = multielement.chord
    cp   = sol.cp

    yflip      := true
    xlabel     := L"x"
    ylabel     := L"C_P"
    legend     := nothing
    lw         := 3
    grid --> true  # Enable grid
    gridlinewidth --> 3
    gridstyle --> :dash
    gridcolor --> :white  # Set white grid lines
    
    # Adjust minor grid visibility
    minorgrid --> true
    minorgridcolor --> :white
    minorgridlinewidth --> 0.4
    minorgridstyle --> :dot
    bg             --> :black
    offset = 1
    for (i,airfoil) in enumerate(multielement.airfoils)
        Nv = length(airfoil.x)
        cp_seg = cp[offset:offset+Nv-1]
        xy_vort = shift_scale_rotate.(airfoil.x,airfoil.y,le_loc[i][1],le_loc[i][2],pitch[i],chord[i])     
        x = [xy[1] for xy in xy_vort]
        @series begin
            (x,cp_seg)
        end 
        offset += Nv
    end 
end
@userplot FlowPlot

@recipe function f(fp::FlowPlot;
                   xlims = (-1.0, 2.0),
                   ylims = (-1.0, 1.0),
                   Nx    = 300,
                   Ny    = 200,
                   min_density = 2,
                   max_density = 10,
                   linewidth = 1.0,
                   clims = (-5.0, 1.0),
                   draw_geometry = true,
                   geometry_color = :white,
                   colormap = :RdBu
                   )

    sol = fp.args[end]

    # Build uniform grid (Xs, Ys, X, Y)
    xs = range(xlims[1], xlims[2], length=Nx)
    ys = range(ylims[1], ylims[2], length=Ny)
    X  = repeat(permutedims(collect(xs)), length(ys), 1)
    Y  = repeat(collect(ys), 1, length(xs))

    # Velocity field on grid
    U, V = induced_velocity(sol, X, Y)

    mask_any = falses(size(X))
    for coords in element_coordinates(sol.geometry)
        xpoly = view(coords, :, 1)
        ypoly = view(coords, :, 2)
        mask_any .|= mask_inside_polygon(xpoly, ypoly, X, Y)
    end

    U[mask_any] .= NaN
    V[mask_any] .= NaN

    # Streamlines colored by Cp
    xy = compute_streamlines(xs, ys, U, V, min_density=min_density, max_density=max_density)

    # Nearest-neighbor lookup helper
    nearest_index(arr, v) = argmin(abs.(arr .- v))

    # Compute Cp at streamline points using nearest-neighbor lookup
    Cpline = similar(xy[:, 1])
    @inbounds for k in eachindex(Cpline)
        x = xy[k, 1]
        y = xy[k, 2]
        if isnan(x) || isnan(y)
            Cpline[k] = NaN
        else
            ix = nearest_index(xs, x)
            iy = nearest_index(ys, y)
            Uline = U[iy, ix]
            Vline = V[iy, ix]
            Cpline[k] = 1 - (Uline^2 + Vline^2)
        end
    end

    # Plot aesthetics
    aspect_ratio := :equal
    xlabel       := "x"
    ylabel       := "y"
    frame        := :box
    grid         := false
    bg           := :black
    xlims        := xlims
    ylims        := ylims
    colorbar     := true
    cbar_title := L"C_p"
    seriescolor  := colormap
    clims        := clims
    label        := nothing

    # Draw geometry shapes first
    if draw_geometry
        for coords in element_coordinates(sol.geometry)
            xt = view(coords, :, 1)
            yt = view(coords, :, 2)
            @series begin
                seriestype   := :shape
                fillcolor    := geometry_color
                linecolor    := geometry_color
                label        := nothing
                (xt, yt)
            end
        end
    end

    # Streamlines as a single polyline with NaN breaks and line_z coloring
    @series begin
        seriestype := :path
        lw         := linewidth
        seriescolor := colormap
        line_z     := Cpline
        label      := nothing
        (xy[:, 1], xy[:, 2])
    end

end