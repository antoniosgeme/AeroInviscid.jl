using AeroInviscid
using Plots
using LaTeXStrings


airfoils = [Airfoil("NACA$i") for i in ["6412","8412","4412"]]
chord = [1.0, 0.5, 0.3]
pitch = [0.0, 30.0, 50.0]
le_loc = [[0.0,0.0],[0.9,-0.15],[1.2,-0.45]]

me = MultielementAirfoil(airfoils, pitch, chord, le_loc)

ip = InviscidProblem(me,0)

sol = solve(ip)

p1 = plot(sol, title="Pressure Coefficient Distribution", xlabel=L"x", ylabel=L"C_p",frame=:box)
xlims!( -0.1, 1.5)
ylims!(-15, 3)

p2 = flowplot(sol, title="Flow Field around Multielement Airfoil", xlabel=L"x", ylabel=L"y")

p3 = plot(p1,p2, layout=(1,2), size=(1200,400),dpi=300,margin=5Plots.mm)

savefig(p3, "assets/multielement_airfoil_solution.png")


airfoils = [Airfoil("NACA$i") for i in ["6412","8412","6412"]]
push!(airfoils, Airfoil("RAE102"))
chord = [0.15,1.0, 0.5, 0.3]
pitch = [-45,0.0, 30.0, 50.0]
le_loc = [[-0.07,0],[0.0,0.0],[0.9,-0.15],[1.2,-0.45]]

me = MultielementAirfoil(airfoils, pitch, chord, le_loc)

ip = InviscidProblem(me,0)

sol = solve(ip)

flowplot(sol)