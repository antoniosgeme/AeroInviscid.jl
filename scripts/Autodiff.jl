using ForwardDiff
using AeroInviscid
using Plots

airfoil = Airfoil("NACA0012")
function compute_cl(α)
    ip = InviscidProblem(airfoil, α)
    sol = solve(ip)
    return sol.cl
end

dcl_dα = ForwardDiff.derivative(compute_cl, 5.0)


# Do the same for multielement airfoil
airfoils = [Airfoil("NACA$i") for i in ["6412","8412","4412"]]
chord = [1.0, 0.5, 0.3]
pitch = [0.0, 30.0, 50.0]
le_loc = [[0.0,0.0],[0.9,-0.15],[1.2,-0.45]]    
me = MultielementAirfoil(airfoils, pitch, chord, le_loc)

function compute_cl_multielement(α)
    ip = InviscidProblem(me, α)
    sol = solve(ip)
    return sol.cl
end

dcl_dα_multielement = ForwardDiff.derivative(compute_cl_multielement, 1.0)


function compute_cl_flap(flap_deflection)
    airfoil = Airfoil("NACA0012")
    airfoil_deflected = deflect_control_surface(airfoil, deflection=flap_deflection)
    ip = InviscidProblem(airfoil_deflected, 0.0)
    sol = solve(ip)
    return sol.cl
end

plot(compute_cl_flap,0:0.1:10)

airfoil = Airfoil("NACA0012")
airfoil_deflected = deflect_control_surface(airfoil, deflection=45)
plot(airfoil_deflected)

plot(compute_cl_flap,0:0.1:10)
dcl_dflap = ForwardDiff.derivative(compute_cl_flap, 20)