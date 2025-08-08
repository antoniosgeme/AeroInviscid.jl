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
                   xlims = (-1,1.5),
                   ylims= (-1,1.5),
                   streams=true,
                   min_density=2,
                   max_density=5
                   ) 

    sol = fp.args[end]
    # build the meshs
    xs = range(xlims[1], xlims[2], length=Nx)
    ys = range(ylims[1], ylims[2], length=Ny)
   
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
        (
            xs,
            ys,
            Cp
        )
    end 

    if streams
        xy = streamlines(xs, ys, U, V,min_density=min_density,max_density=max_density)

        @series begin
            linecolor    := :black
            label        := nothing
            lw           := 1.5
            (xy[:, 1],xy[:, 2])
        end       
    end  

    if typeof(sol.geometry) <: MultielementAirfoil
        for (i,airfoil) in enumerate(sol.geometry.airfoils)
            xy_vort = shift_scale_rotate.(airfoil.x,airfoil.y,
                                        sol.geometry.le_loc[i][1],sol.geometry.le_loc[i][2],
                                        sol.geometry.pitch[i],sol.geometry.chord[i]) 
            x_sheet  = [xy[1] for xy in xy_vort]
            y_sheet  = [xy[2] for xy in xy_vort]
    
            @series begin
                seriestype := :shape
                fillcolor    := :black
                linecolor    := :black
                aspect_ratio := :equal
                label        :=nothing
                (x_sheet,y_sheet)
            end
        end 

    else typeof(sol.geometry) <: Airfoil
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

        
end 
