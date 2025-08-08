using AeroGeometry
using AeroInviscid
using Plots
using LinearAlgebra
using PrettyStreamlines

airfoil = Airfoil("NACA6409")

prob = InviscidProblem(airfoil,2)
sol = solve(prob)

x_vec = -1:0.02:2
y_vec = -1:0.02:1
X = [i for j in y_vec, i in x_vec]
Y = [j for j in y_vec, i in x_vec]

U,V = induced_velocity(sol,X,Y)
U,V = induced_velocity(sol,1,1)
streamlines(x_vec,y_vec,U,V,min_density=2,max_density=10)
plot!(airfoil.x,airfoil.y,seriestype=:shape,fill=:black)

function streamlines(sol::InviscidSolution,x::AbstractVector,y::AbstractVector)
    U,V = induced_velocity(sol,x,y)
    Cp(x,y,u,v) = 1 - (u^2 + v^2)
    streamlines(x,y,U,V,color_by=Cp,cmap=:turbo,legend=false)
    plot!(airfoil.x,airfoil.y,seriestype=:shape,fill=:black)
end 

streamlines(sol,x_vec,y_vec)



flowplot(sol,clim=(-5,4),min_density=2,max_density=15)
plot(sol)

alphas = 0:1:14
probs = InviscidProblem.(Ref(airfoil),alphas)
sols = solve.(probs);

plot(alphas,[sol.cl for sol in sols])
plot!(alphas,2π*deg2rad.(alphas),ls=:dash)
flowplot(sols,clim=(-5,5))
airfoil1 = Airfoil("NACA0012")
airfoil2 = Airfoil("naca6409")
airfoil3 = Airfoil("s3024")

multielement = MultielementAirfoil([airfoil1,airfoil2,airfoil3],[25,20,0],[1,1,1],[[0,0],[1.2,0],[0,-0.75]])
prob = InviscidProblem(multielement,0)
sol = solve(prob)
plot(sol)
flowplot(sol,xlims=(-1,3),clims=(-10,10),ylims=(-1,1),min_density=2,max_density=8)


x_vec = -0.5:0.01:3
y_vec = -1:0.01:1
X = [i for j in y_vec, i in x_vec]
Y = [j for j in y_vec, i in x_vec]

U,V = induced_velocity(sol,X,Y)

p = streamlines(x_vec, y_vec, U, V)

for (i,airfoil) in enumerate(multielement.airfoils)
    xy_vort = shift_scale_rotate.(airfoil.x,airfoil.y,
                                        multielement.le_loc[i][1],multielement.le_loc[i][2],
                                        multielement.pitch[i],multielement.chord[i]) 
    x_sheet  = [xy[1] for xy in xy_vort]
    y_sheet  = [xy[2] for xy in xy_vort]
    plot!(x_sheet,y_sheet,seriestype=:shape,fill=:black)
end 
