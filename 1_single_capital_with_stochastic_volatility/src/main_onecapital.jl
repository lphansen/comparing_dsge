ENV["JULIA_PKG_OFFLINE"] = "true"
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Optim
using Roots
using NPZ
using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--Delta"
            help = "time step"
            arg_type = Float64
            default = 1.  
        "--gamma"
            help = "risk aversion"
            arg_type = Float64
            default = 8.0
        "--rho"
            help = "inverse IES"
            arg_type = Float64
            default = 1.0
        "--delta"
            help = "discount rate"
            arg_type = Float64
            default = 0.1 
        "--alpha"
            help = "productivity"
            arg_type = Float64
            default = 1.0
        "--action_name"
            help = "action name"
            arg_type = String
            default = "publish"
    end
    return parse_args(s)
end

@show parsed_args = parse_commandline()
gamma                = parsed_args["gamma"]
rho                  = parsed_args["rho"]
Delta                = parsed_args["Delta"]
delta                = parsed_args["delta"]
alpha                = parsed_args["alpha"]
action_name          = parsed_args["action_name"]
include("utils_onecapital.jl")

outputdir = "./output/"*action_name*"/Delta_"*string(Delta)*"/delta_"*string(delta)*"/gamma_"*string(gamma)*"_rho_"*string(rho)*"_alpha_"*string(alpha)*"/"
isdir(outputdir) || mkpath(outputdir)

## Calibration
a11 = 0.056;  ## long run risk persistence (beta_1 in the paper)
a22 = 0.194;  ## stochastic volatility persistence (beta_2 in the paper)
eta = 0.04;   ## depreciation rate (eta_k in the paper)
beta = 0.04;  ## loading factor on the capital stock in the long run risk process (beta_k in the paper)
phi = 8.0;    ## adjustment cost
sigma_k = vec([0.92,0.4, 0.0]) * sqrt(12)     ## shock exposure in the capital evolution process
sigma_z = vec([0.0, 5.7, 0.0]) * sqrt(12)     ## shock exposure in the long run risk evolution process
sigma_y = vec([0,   0,   0.00031]) * sqrt(12) ## shock exposure in the stochastic volatility process

## Construct state space grid
II, JJ = trunc(Int, 201), trunc(Int, 201); ## number of grid points for the long run risk state variable and the stochastic volatility state variable (z1, z2 in the paper)
zmax = 5.0;                                ## upper bound of the long run risk state variable
zmin = -zmax;                              ## lower bound of the long run risk state variable
ymean = 6.3e-06;                           ## stochastic volatility mean
ymax = 2e-05;                              ## upper bound of the stochastic volatility state variable
ymin = 5e-07;                              ## lower bound of the stochastic volatility state variable

maxit = 50000;      # maximum number of iterations in the HJB loop
crit  = 10e-6;      # criterion HJB loop

## Initialize model objects
baseline = Baseline(a11, a22, beta, eta, sigma_k, sigma_z, sigma_y, delta);
technology = Technology(alpha, phi);
model = OneCapitalEconomy(baseline, technology);

## Initialize grid and FDM
grid = Grid_zy(zmin, zmax, II, ymin, ymax, JJ);
params = FinDiffMethod(maxit, crit, Delta);

## Initialize value function and make a guess for consumption-capital ratio
if rho == 1.0
    preloadV0 = -6*ones(grid.I, grid.J)
    preloadcons = 0.03*ones(grid.I, grid.J)
else
    preload_rho = 1.0
    preload_Delta = 1.0
    if delta == 0.01
        preload_alpha = 0.0922
    elseif delta == 0.015
        preload_alpha = 0.1002
    end
    preloaddir = "./output/"*action_name*"/Delta_"*string(preload_Delta)*"/delta_"*string(delta)*"/gamma_"*string(gamma)*"_rho_"*string(preload_rho)*"_alpha_"*string(preload_alpha)*"/"
    preload = npzread(preloaddir*"res.npz")
    println("preload location : "*preloaddir)
    preloadV0 = preload["V"]
    preloadcons = preload["cons"]
end

## Solve the model
times = @elapsed begin
A, V, val, d, hk_F, hz_F, hy_F, hz_B, hk_B, hy_B, mu_k, mu_z, mu_y, V0, Vz, Vz_F, Vz_B, Vy, Vy_B, Vy_F, c, zz, yy, dz, dy =
        value_function_onecapital(gamma, rho, model, grid, params, preloadV0, preloadcons, ymean);
end
println("Convegence time (minutes): ", times/60)
g = stationary_distribution(A, grid)

## Control variables
hk = (hk_F + hk_B)/2.; ## Robust control
hz = (hz_F + hz_B)/2.; ## Robust control
hy = (hy_F + hy_B)/2.; ## Robust control
c = alpha*ones(II,JJ) - d ## consumption capital ratio

## Save results
results = Dict(
# Parameters
"delta" => delta, "eta" => eta, "a11"=> a11, "a22"=> a22, "beta" => beta, "sigma_k" => sigma_k, "sigma_z" =>  sigma_z, "sigma_y" => sigma_y, "alpha" => alpha,  "phi" => phi,  "rho" => rho, "gamma" => gamma, "ymean" => ymean,
# Grid
"I" => II, "J" => JJ, "zmax" => zmax, "zmin" => zmin, "ymax" => ymax, "ymin" => ymin, "zz" => zz, "yy" => yy,
# Iteration
"maxit" => maxit, "crit" => crit, "Delta" => Delta,  "times" => times,
# Results
"cons" => c, "d" => d, "hk" => hk, "hz" => hz, "hy" => hy,
"V" => V, "Vz" => Vz, "Vy" => Vy, "val" => val, "dz" => dz, "dy" => dy,
"mu_k" => mu_k, "mu_z" => mu_z, "mu_y" => mu_y, 
"g" => g, # stationary density
)
npzwrite(outputdir*"res.npz", results)
