###############################################################################
#  mfoil_types.jl — Julia analogue of the MATLAB `mfoil` object               #
#                                                                              #
#  This file contains **data‑only** definitions: a lightweight container that  #
#  mirrors the nested struct layout used in Krzysztof Fidkowski’s MATLAB       #
#  implementation.  Solver logic lives elsewhere; here we simply provide a     #
#  place for every variable the algorithms need.                               #
#                                                                              #
#  Usage                                                                       #
#  -----                                                                       #
#  include("mfoil_types.jl")          # bring structs into scope               #
#  M = Mfoil()                         # default‑constructed empty object       #
#  M.geom.name = "NACA 0012"           # mutate like any mutable struct         #
###############################################################################

module MfoilTypes

using SparseArrays, LinearAlgebra
using Base: @kwdef

# ---------------------------------------------------------------------------
#  Geometry & discretisation containers
# ---------------------------------------------------------------------------

@kwdef mutable struct Geom
    chord   :: Float64            = 1.0               # chord length [m]
    wakelen :: Float64            = 1.0               # wake extent (×c)
    npoint  :: Int                = 0                 # raw geometry points
    name    :: String             = "noname"          # e.g. "NACA0012"
    xpoint  :: Matrix{Float64}    = zeros(2, 0)       # 2×N raw coords
    xref    :: NTuple{2,Float64}  = (0.25, 0.0)       # moment ref point
end

@kwdef mutable struct PanelMesh
    N  :: Int                 = 0               # number of nodes
    x  :: Matrix{Float64}     = zeros(2,0)      # 2×N coords
    s  :: Vector{Float64}     = Float64[]       # arclength at nodes
    t  :: Matrix{Float64}     = zeros(2,0)      # 2×N tangents dx/ds
end

# ---------------------------------------------------------------------------
#  Operating‑condition record
# ---------------------------------------------------------------------------

@kwdef mutable struct Oper
    Vinf    :: Float64                    = 1.0
    alpha   :: Float64                    = 0.0    # deg
    rho     :: Float64                    = 1.0
    cltgt   :: Float64                    = 0.0
    givencl :: Bool                       = false  # if true, trim α
    initbl  :: Bool                       = true   # (re)init BL each solve
    viscous :: Bool                       = false
    redowake:: Bool                       = false  # rebuild wake after α change
    Re      :: Float64                    = 1e5
    Ma      :: Float64                    = 0.0
    forcet  :: NTuple{3,Bool}             = (false,false,false)
    xft     :: NTuple{3,Float64}          = (1.0,1.0,1.0)   # forced‑trip x/c
    xift    :: NTuple{3,Float64}          = (0.0,0.0,0.0)   # forced‑trip ξ
end

# ---------------------------------------------------------------------------
#  Inviscid & viscous solver auxiliaries
# ---------------------------------------------------------------------------

@kwdef mutable struct Isol
    AIC       :: Matrix{Float64}     = zeros(0,0)  # influence coeffs
    gamref    :: Matrix{Float64}     = zeros(0,2)  # γ for α=0,90°
    gam       :: Vector{Float64}     = Float64[]   # γ at current α
    sstag     :: Float64             = 0.0         # stagnation s
    sstag_g   :: NTuple{2,Float64}   = (0.0,0.0)
    sstag_ue  :: NTuple{2,Float64}   = (0.0,0.0)
    Istag     :: NTuple{2,Int}       = (0,0)       # neighbouring node idxs
    sgnue     :: Vector{Int}         = Int[]       # +/− sign per foil node
    xi        :: Vector{Float64}     = Float64[]   # distance from stag
    xstag     :: NTuple{2,Float64}   = (0.0,0.0)   # stag point coords
    uewi      :: Vector{Float64}     = Float64[]   # wake edge speed (α)
    uewiref   :: Matrix{Float64}     = zeros(0,2)  # wake speed (0,90°)
end

@kwdef mutable struct Vsol
    θ         :: Vector{Float64}                  = Float64[]                             # momentum thickness
    δstar     :: Vector{Float64}                  = Float64[]                             # displacement thickness
    Is        :: Vector{Vector{Int}}              = Vector{Vector{Int}}()                  # lower/upper/wake index lists
    wgap      :: Vector{Float64}                  = Float64[]                             # wake gap δ* extension
    ue_m      :: Matrix{Float64}                  = zeros(0,0)                            # d(ue)/d(mass)
    sigma_m   :: Matrix{Float64}                  = zeros(0,0)
    ue_sigma  :: Matrix{Float64}                  = zeros(0,0)
    turb      :: Vector{Bool}                    = Bool[]                                # turbulence flags
    xt        :: Float64                          = 0.0                                  # transition x/c
    Xt        :: Matrix{Float64}                  = zeros(2,2)                            # 2×2 (lower,upper)
end

@kwdef mutable struct Glob
    Nsys   :: Int                       = 0
    U      :: Matrix{Float64}           = zeros(0,0)   # 4×Nsys
    dU     :: Matrix{Float64}           = zeros(0,0)   # Newton increment
    dalpha :: Float64                   = 0.0
    conv   :: Bool                      = true
    R      :: Vector{Float64}           = Float64[]
    R_U    :: SparseMatrixCSC{Float64,Int} = spzeros(0,0)
    R_x    :: SparseMatrixCSC{Float64,Int} = spzeros(0,0)
end

@kwdef mutable struct Post
    cp      :: Vector{Float64} = Float64[]
    cpi     :: Vector{Float64} = Float64[]
    cl      :: Float64         = 0.0
    cl_ue   :: Vector{Float64} = Float64[]
    cl_alpha:: Float64         = 0.0
    cm      :: Float64         = 0.0
    cdpi    :: Float64         = 0.0
    cd      :: Float64         = 0.0
    cdf     :: Float64         = 0.0
    cdp     :: Float64         = 0.0
    # distributions
    th      :: Vector{Float64} = Float64[]
    ds      :: Vector{Float64} = Float64[]
    sa      :: Vector{Float64} = Float64[]
    ue      :: Vector{Float64} = Float64[]
    uei     :: Vector{Float64} = Float64[]
    cf      :: Vector{Float64} = Float64[]
    Ret     :: Vector{Float64} = Float64[]
    Hk      :: Vector{Float64} = Float64[]
end

# ---------------------------------------------------------------------------
#  Global parameter block (subset)
# ---------------------------------------------------------------------------

@kwdef mutable struct Param
    # basic solver settings
    verb    :: Int       = 1
    rtol    :: Float64   = 1e-10
    niglob  :: Int       = 50
    doplot  :: Bool      = true

    # freestream & thermodynamics
    Vinf    :: Float64   = 1.0     # freestream speed [m/s]
    muinf   :: Float64   = 0.0     # freestream dynamic viscosity
    mu0     :: Float64   = 0.0     # stagnation dynamic viscosity
    rho0    :: Float64   = 1.0     # stagnation density
    Minf    :: Float64   = 0.0     # freestream Mach number
    gam     :: Float64   = 1.4     # ratio of specific heats
    KTb     :: Float64   = 1.0     # Karman–Tsien β
    KTl     :: Float64   = 0.0     # Karman–Tsien λ
    H0      :: Float64   = 0.0     # stagnation enthalpy
    Tsrat   :: Float64   = 0.35    # Sutherland’s T_s/T_ref ratio
    cps     :: Float64   = 0.0     # compressible cp correction

    # viscous model constants (same names as MATLAB code)
    ncrit   :: Float64   = 9.0
    Cuq     :: Float64   = 1.0
    Dlr     :: Float64   = 0.9
    SlagK   :: Float64   = 5.6
    CtauC   :: Float64   = 1.8
    CtauE   :: Float64   = 3.3
    GA      :: Float64   = 6.7
    GB      :: Float64   = 0.75
    GC      :: Float64   = 18.0

    wake  :: Bool  = false
    turb  :: Bool  = false
    simi  :: Bool  = false

    is     :: Int      = 0
end

# ---------------------------------------------------------------------------
#  Master container — the Julia twin of MATLAB’s `mfoil` handle object
# ---------------------------------------------------------------------------

@kwdef mutable struct Mfoil
    version::String   = "2025‑08‑07‑jl"
    geom   :: Geom     = Geom()
    foil   :: PanelMesh= PanelMesh()
    wake   :: PanelMesh= PanelMesh()
    oper   :: Oper     = Oper()
    isol   :: Isol     = Isol()
    vsol   :: Vsol     = Vsol()
    glob   :: Glob     = Glob()
    post   :: Post     = Post()
    param  :: Param    = Param()
end

export Geom, PanelMesh, Oper, Isol, Vsol, Glob, Post, Param, Mfoil

end # module
