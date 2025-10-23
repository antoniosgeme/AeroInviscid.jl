using AeroGeometry
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

p1 = plot(sol, title="Pressure Coefficient Distribution", xlabel=L"x", ylabel=L"C_p")
xlims!( -1, 2.6)
ylims!(-15, 3)

p2 = flowplot(sol, title="Flow Field around Multielement Airfoil", xlabel=L"x", ylabel=L"y")

p3 = plot(p1,p2, layout=(2,1), size=(800,1000))

