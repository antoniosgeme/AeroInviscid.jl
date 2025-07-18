using LaTeXStrings
using RecipesBase
using PlotUtils
using Interpolations

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
    ) where {M<:MultielementAirfoil,S<:LinearVortex}

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
                   Nx    = 150,
                   Ny    = 150,
                   xlim       = (-1.0, 1.5),
                   ylim       = (-1.0, 1.5),
                   density    = 1.0) 

    sol = fp.args[end]
    # build the mesh
    xs = range(xlim[1], xlim[2], length=Nx)
    ys = range(ylim[1], ylim[2], length=Ny)
   

    U,V = induced_velocity(sol,xs,ys)
    Cp = @. 1 - U^2 - V^2

    # compute Cp
    Cp = 1 .- (U.^2 .+ V.^2)
    @series begin
        seriestype   := :contourf
        levels       := 50
        colorbar     := true
        aspect_ratio := :equal
        title        := "Pressure Coefficient Cₚ"
        size         := (600,600)
        colormap     := :RdBu    
        lw           := 0 
        clim         := (-5, 5)
        (
            xs,
            ys,
            Cp'
        )
    end 

    xy = streamlines(xs, ys, U', V',min_density=2,max_density=5)

    @series begin
        linecolor    := :blue
        label        := nothing
        lw           := 1.5
        (xy[:, 1],xy[:, 2])
    end        

    x_sheet  = sol.geometry.x
    y_sheet  = sol.geometry.y

    @series begin
        seriestype := :shape
        fillcolor    := :black
        linecolor    := :black
        aspect_ratio := :equal
        label        :=nothing
        (x_sheet,y_sheet)
    end

        
end 
