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
        "--alpha"
            help = "productivity"
            arg_type = Float64
            default = 1.0
        "--action_name"
            help = "action name"
            arg_type = String
            default = "publish"
        "--q"
            help = "relative entropy"
            arg_type = Float64
            default = 0.05
        "--twoparameter"
            help = "two parameter ambiguity"
            arg_type = Int
            default = 1
    end
    return parse_args(s)
end

@show parsed_args = parse_commandline()
gamma                = parsed_args["gamma"]
rho                  = parsed_args["rho"]
alpha                = parsed_args["alpha"]
Delta                = parsed_args["Delta"]
action_name          = parsed_args["action_name"]
q                    = parsed_args["q"]
twoparameter         = parsed_args["twoparameter"]
include("utils_onecapital.jl")

outputdir = "./output/"*action_name*"/Delta_"*string(Delta)*"_twoparameter_"*string(twoparameter)*"/q_"*string(q)*"_gamma_"*string(gamma)*"_rho_"*string(rho)*"_alpha_"*string(alpha)*"/"
isdir(outputdir) || mkpath(outputdir)

## Construct shock exposure matrix
ymean = 6.3e-06; ## stochastic volatility mean, we abstract this process as a constant scaling factor 
sigma_k = vec([0.92,0.4]) * sqrt(12) * sqrt(ymean) ## shock exposure in the capital evolution process
sigma_z = vec([0.0, 5.7]) * sqrt(12) * sqrt(ymean) ## shock exposure in the long run risk evolution process

## Calibration
delta = 0.01 ## discount rate
a11 = 0.056  ## long run risk persistence (beta_1 in the paper)
eta = 0.04   ## depreciation rate (eta_k in the paper)
beta = 0.04  ## loading factor on the capital stock in the long run risk process (beta_k in the paper)
phi = 8.0    ## adjustment cost
rho1 = 0.0   ## linear coefficient of test function, see (Hansen and Sargent, 2020)
if twoparameter == 0
    rho2 = q^2 / dot(sigma_z, sigma_z) / 2  ## quardratic coefficient of test function, see (Hansen and Sargent, 2020)
elseif twoparameter == 1
    rho2 = q^2 / dot(sigma_z, sigma_z)      ## quardratic coefficient of test function, see (Hansen and Sargent, 2020)
end 

## Construct state space grid
II = trunc(Int,401);    ## number of grid points for the long run risk state variable (z1 in the paper)
zmax = 5.0;             ## upper bound of the long run risk state variable
zmin = -zmax;           ## lower bound of the long run risk state variable

maxit = 50000;      # maximum number of iterations in the HJB loop
crit  = 10e-6;      # criterion HJB loop

## Initialize model objects
baseline = Baseline(a11, sigma_z, beta, eta, sigma_k, delta);
technology = Technology(alpha, phi);
robust = Robustness(q, rho1, rho2);
model = OneCapitalEconomy(baseline, technology, robust);

## Initialize grid and FDM
grid = Grid_z(zmin, zmax, II);
params = FinDiffMethod(maxit, crit, Delta);

preloadV0 = -2*ones(grid.I)
preloadcons = 0.03*ones(grid.I)

times = @elapsed begin
A, V, val, d_F, d_B, hk_F, hz_F, hk_B, hz_B, s1_F, s2_F, lambda_F, s1_B, s2_B, lambda_B, mu_z_distorted_F, mu_z_distorted_B,
    mu_1_F, mu_1_B, mu_z_F, mu_z_B, V0, Vz_B, Vz_F, cF, cB, Vz, zz, dz, uu =
    value_function_onecapital(gamma, rho, model, grid, params, preloadV0, preloadcons, twoparameter);
end
println("Convegence time (minutes): ", times/60)
g = stationary_distribution(A, grid)

## Control variables
hk = (hk_F + hk_B)/2.; ## Robust control for misspecification 
hz = (hz_F + hz_B)/2.; ## Robust control for misspecification 
s1 = (s1_F + s1_B)/2.; ## Robust control for ambiguity 
s2 = (s2_F + s2_B)/2.; ## Robust control for ambiguity 
d = (d_F + d_B)/2; ## Investment capital ratio
mu_1 = (mu_1_F + mu_1_B)/2.; ## drift term for capital evolution process
mu_z = (mu_z_F + mu_z_B)/2.; ## drift term for long run risk evolution process
mu_z_distorted = (mu_z_distorted_F + mu_z_distorted_B)/2.; ## drift term distortion induced by ambiguity
c = alpha*ones(II) - d; ## consumption capital ratio

results = Dict(
# Parameters
"delta" => delta, "eta" => eta, "a11"=> a11,  "beta" => beta, "sigma_k" => sigma_k, "sigma_z" =>  sigma_z, "alpha" => alpha,  "phi" => phi,
"rho1" => rho1, "rho2" => rho2, "q" => q,
"gamma" => gamma, "rho" => rho,
# Grid
"I" => II, "zmax" => zmax, "zmin" => zmin, "zz" => zz, 
# Iteration
"maxit" => maxit, "crit" => crit, "Delta" => Delta,
# Results
"cons" => c,"hk" => hk,"hz" => hz, "s1" => s1, "s2" => s2,"d" => d,
"V" => V, "Vz" => Vz, "dz" => dz,
"mu_z_distorted" => mu_z_distorted,"mu_1" => mu_1, "mu_z" => mu_z,
"g" => g,  # stationary density
)
npzwrite(outputdir*"res.npz", results)
