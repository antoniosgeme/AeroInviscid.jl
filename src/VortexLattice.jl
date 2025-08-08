using AeroGeometry
using Plots
using Interpolations
wing = cessna152().wings[1]


chordwise_num = 10
spanwise_num = 10

points,faces,section_num = AeroGeometry.mesh(wing,camberline=true)

for (i,sec) in enumerate(wing.sections)
    idx = findall(section_num .== i)
    secpoints = points[idx,:]
end 

