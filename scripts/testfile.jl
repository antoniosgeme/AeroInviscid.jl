using AeroGeometry
using AeroInviscid
using Plots

airfoil = Airfoil("NACA0012")

prob = InviscidProblem(airfoil,14)

sol = solve(prob)

plot(sol)


flowplot(sol)


x_vec = -1:0.01:1.5
y_vec = -1:0.01:1

U,V = induced_velocity(sol,x_vec,y_vec)



# 3. Call the even_stream_data function
# We'll set compute_dist=false for this example to keep it simple.
xy = streamlines(x_vec, y_vec, U', V',min_density=2,max_density=5)

plot(xy[:, 1], xy[:, 2],
    title="Evenly Spaced Streamlines of a Vortex Field",
    xlabel="X",
    ylabel="Y",
    legend=false,
    aspect_ratio=:equal,
    linewidth=1.5,
    size=(600, 600)
)

plot!(airfoil.x,airfoil.y,seriestype=:shape,fill=:black)


xy = streamlines_from_grid(x_vec, y_vec, U', V',density=0.8)






airfoil1 = Airfoil("NACA0012")
airfoil2 = Airfoil("naca6409")
airfoil3 = Airfoil("s3024")

multielement = MultielementAirfoil([airfoil1,airfoil2,airfoil3],[25,20,0],[1,1,1],[[0,0],[1.2,0],[0,-0.5]])
prob = InviscidProblem(multielement,0)
sol = solve(prob)
plot(sol)


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
