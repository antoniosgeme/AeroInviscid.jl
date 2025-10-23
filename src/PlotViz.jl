using LaTeXStrings
using RecipesBase
using PlotUtils
using Interpolations
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

    # Interpolate U,V on streamline points for coloring
    itpU = Interpolations.interpolate((collect(ys), collect(xs)), U, Gridded(Linear()))
    itpV = Interpolations.interpolate((collect(ys), collect(xs)), V, Gridded(Linear()))
    Uline = similar(xy[:, 1])
    Vline = similar(xy[:, 2])

    @inbounds for k in eachindex(Uline)
        y = xy[k, 2]; x = xy[k, 1]
        if isnan(x) || isnan(y)
            Uline[k] = NaN; Vline[k] = NaN
        else
            Uline[k] = itpU(y, x)
            Vline[k] = itpV(y, x)
        end
    end
    Cpline = @. 1 - (Uline^2 + Vline^2)

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

@userplot PrettyFlowPlot

@recipe function f(pfp::PrettyFlowPlot;
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

    sol = pfp.args[end]

    # Build uniform grid (Xs, Ys, X, Y)
    xs = range(xlims[1], xlims[2], length=Nx)
    ys = range(ylims[1], ylims[2], length=Ny)
    X  = repeat(permutedims(collect(xs)), length(ys), 1)
    Y  = repeat(collect(ys), 1, length(xs))

    # Velocity field on grid
    U, V = induced_velocity(sol, X, Y)

    # Mask inside airfoil shapes (no flow inside body)
    function mask_inside_polygon(xp::AbstractVector, yp::AbstractVector, X, Y)
        n = length(xp)
        @assert n == length(yp) && n ≥ 3
        inside = falses(size(X))

        x1 = xp; y1 = yp
        x2 = circshift(xp, -1); y2 = circshift(yp, -1)

        @inbounds for i in 1:n
            ycross = ((y1[i] .<= Y) .& (Y .< y2[i])) .| ((y2[i] .<= Y) .& (Y .< y1[i]))
            xints  = (x2[i] - x1[i]) .* (Y .- y1[i]) ./ (y2[i] - y1[i]) .+ x1[i]
            inside .= xor.(inside, ycross .& (X .< xints))
        end
        return inside
    end

    mask_any = falses(size(X))

    if typeof(sol.geometry) <: MultielementAirfoil
        for (i, airfoil) in enumerate(sol.geometry.airfoils)
            isempty(airfoil.x) && continue
            xy = shift_scale_rotate.(airfoil.x, airfoil.y,
                                     sol.geometry.le_loc[i][1], sol.geometry.le_loc[i][2],
                                     sol.geometry.pitch[i],    sol.geometry.chord[i])
            xpoly = getindex.(xy, 1)
            ypoly = getindex.(xy, 2)
            mask_any .|= mask_inside_polygon(xpoly, ypoly, X, Y)
        end
    elseif typeof(sol.geometry) <: Airfoil
        xpoly = sol.geometry.x
        ypoly = sol.geometry.y
        mask_any .|= mask_inside_polygon(xpoly, ypoly, X, Y)
    end

    U[mask_any] .= NaN
    V[mask_any] .= NaN

    # Streamlines colored by Cp
    xy = streamlines(xs, ys, U, V, min_density=min_density, max_density=max_density)

    # Interpolate U,V on streamline points for coloring
    itpU = Interpolations.interpolate((collect(ys), collect(xs)), U, Gridded(Linear()))
    itpV = Interpolations.interpolate((collect(ys), collect(xs)), V, Gridded(Linear()))
    Uline = similar(xy[:, 1])
    Vline = similar(xy[:, 2])

    @inbounds for k in eachindex(Uline)
        y = xy[k, 2]; x = xy[k, 1]
        if isnan(x) || isnan(y)
            Uline[k] = NaN; Vline[k] = NaN
        else
            Uline[k] = itpU(y, x)
            Vline[k] = itpV(y, x)
        end
    end
    Cpline = @. 1 - (Uline^2 + Vline^2)

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
    seriescolor  := colormap
    clims        := clims
    label        := nothing

    # Draw geometry shapes first
    if draw_geometry
        if typeof(sol.geometry) <: MultielementAirfoil
            for (i, airfoil) in enumerate(sol.geometry.airfoils)
                isempty(airfoil.x) && continue
                xy_t = shift_scale_rotate.(airfoil.x, airfoil.y,
                                           sol.geometry.le_loc[i][1], sol.geometry.le_loc[i][2],
                                           sol.geometry.pitch[i],    sol.geometry.chord[i])
                xt = getindex.(xy_t, 1)
                yt = getindex.(xy_t, 2)
                @series begin
                    seriestype   := :shape
                    fillcolor    := geometry_color
                    linecolor    := geometry_color
                    label        := nothing
                    (xt, yt)
                end
            end
        else
            xt = sol.geometry.x
            yt = sol.geometry.y
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
