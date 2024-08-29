using Pkg
using Optim
using Roots
using NPZ
using ArgParse

function run_simulation(Delta=1.0, gamma=8.0, rho=1.0, alpha=0.1844, kappa=0.0, zeta=0.5, beta1=0.01, beta2=0.01, action_name="publish")
    ENV["JULIA_PKG_OFFLINE"] = "true"
    Pkg.activate(".")
    Pkg.instantiate()

    include("utils_twocapitals_two_dimensions.jl")

    outputdir = "./output/"*action_name*"/Delta_"*string(Delta)*"/beta1_"*string(beta1)*"_beta2_"*string(beta2)*"/kappa_"*string(kappa)*"_zeta_"*string(zeta)*"/gamma_"*string(gamma)*"_rho_"*string(rho)*"_alpha_"*string(alpha)*"/"
    isdir(outputdir) || mkpath(outputdir)

    ## Calibration
    delta = 0.01    ## discount rate
    betaz = 0.056   ## long run risk persistence (beta_1 in the paper)
    eta1 = 0.04     ## depreciation rate (eta_k in the paper)
    eta2 = 0.04     ## depreciation rate (eta_k in the paper)
    phi1 = 8.0      ## adjustment cost
    phi2 = 8.0      ## adjustment cost
    smean = 6.30e-06;   ## stochastic volatility mean, we abstract this process as a constant scaling factor
    sigma_k1 = [sqrt(2)*0.92, .0, .4] * sqrt(smean) * sqrt(12)  ## shock exposure in the capital 1 evolution process
    sigma_k2 = [0, sqrt(2)*0.92, .4] * sqrt(smean) * sqrt(12)   ## shock exposure in the capital 2 evolution process
    sigma_z =  [0, 0, 5.7] * sqrt(smean) * sqrt(12)             ## shock exposure in the long run risk evolution process

    ## Construct state space grid
    II, JJ = trunc(Int,1001), trunc(Int,201);   ## number of grid points for the endougenous state variable (log K2 - logK1) and the long run risk state variable (\hat Y, z1 in the paper)
    rmax = kappa == 0.0 ? 6.0 : 1.0             ## upper bound of the endougenous state variable (log K2 - logK1)
    rmin = -rmax;                               ## lower bound of the endougenous state variable (log K2 - logK1)
    zmax = 1.0;                                 ## upper bound of the long run risk state variable
    zmin = -zmax;                               ## lower bound of the long run risk state variable

    maxit = 200000;     # maximum number of iterations in the HJB loop
    crit  = 10e-6;      # criterion HJB loop

    # Initialize model objects 
    baseline1 = Baseline(zeta, kappa, betaz, sigma_z, beta1, sigma_k1, delta);
    baseline2 = Baseline(zeta, kappa, betaz, sigma_z, beta2, sigma_k2, delta);
    technology1 = Technology(alpha, phi1, eta1);
    technology2 = Technology(alpha, phi2, eta2);
    model = TwoCapitalEconomy(baseline1, baseline2, technology1, technology2);

    ## Initialize grid and FDM
    grid = Grid_rz(rmin, rmax, II, zmin, zmax, JJ);
    params = FinDiffMethod(maxit, crit, Delta);

    ## Initialize value function and make a guess for consumption-capital ratio
    if rho == 1.0
        preloadV0 = -3*ones(grid.I, grid.J)
        preloadcons = 0.03*ones(grid.I, grid.J)
    else
        preload_rho = 1.0
        preload_Delta = kappa == 0.0 ? (beta1 == 0.0 ? 0.1 : 1.0) : 1.0
        preload_alpha = 0.1844
        preloaddir = "./output/"*action_name*"/Delta_"*string(preload_Delta)*"/beta1_"*string(beta1)*"_beta2_"*string(beta2)*"/kappa_"*string(kappa)*"_zeta_"*string(zeta)*"/gamma_"*string(gamma)*"_rho_"*string(preload_rho)*"_alpha_"*string(preload_alpha)*"/"
        preload = npzread(preloaddir*"res.npz")
        println("preload location : "*preloaddir)
        preloadV0 = preload["V"]
        preloadcons = preload["cons"]
    end

    ## Solve the model
    times = @elapsed begin
        A, V, val, d1_F, d2_F, d1_B, d2_B, h1_F, h2_F, hz_F, h1_B, h2_B, hz_B,
        mu_1_F, mu_1_B, mu_r_F, mu_r_B, mu_z_F, mu_z_B, V0, Vr, Vr_F, Vr_B, Vz_B, Vz_F, cF, cB, Vz, rr, zz, dr, dz =
            value_function_twocapitals(gamma, rho, model, grid, params, preloadV0, preloadcons, beta1);
    end
    println("Convergence time (minutes): ", times/60)
    g = stationary_distribution(A, grid)

    ## Control variables
    h1 = (h1_F + h1_B)/2.;  ## Robust control
    h2 = (h2_F + h2_B)/2.;  ## Robust control
    hz = (hz_F + hz_B)/2.;  ## Robust control
    mu_1 = (mu_1_F + mu_1_B)/2.;    ## drift term for composite capital evolution process (K^a process in the paper)
    mu_r = (mu_r_F + mu_r_B)/2.;    ## drift term for the endougenous state variable (log K2 - logK1) evolution process
    mu_z = (mu_z_F + mu_z_B)/2.;    ## drift term for the long run risk evolution process
    d1 = (d1_F + d1_B)/2;           ## Investment capital 1 ratio
    d2 = (d2_F + d2_B)/2;           ## Investment capital 2 ratio

    r = range(rmin, stop=rmax, length=II);    
    rr = r * ones(1, JJ);                     
    IJ = II*JJ;
    k1a = zeros(II,JJ)
    k2a = zeros(II,JJ)
    for i=1:IJ
        p = rr[i];
        k1a[i] = (1-zeta + zeta*exp.(p*(1-kappa))).^(1/(kappa-1));
        k2a[i] = ((1-zeta)*exp.(p*((kappa-1))) + zeta).^(1/(kappa-1));
    end
    d1k = d1.*k1a
    d2k = d2.*k2a
    c = alpha*ones(II,JJ) - d1k - d2k        ## consumption capital ratio

    ## Save results
    results = Dict(
    # Parameters    
    "delta" => delta, "betaz"=> betaz,  "beta1" => beta1, "beta2" => beta2, "eta1" => eta1, "eta2" => eta2, 
    "sigma_k1" => sigma_k1, "sigma_k2" => sigma_k2, "sigma_z" =>  sigma_z, 
    "alpha" => alpha, "kappa" => kappa,"zeta" => zeta, "phi1" => phi1, "phi2" => phi2,  
    "gamma" => gamma, "rho" => rho,
    # Grid
    "I" => II, "J" => JJ, "rmax" => rmax, "rmin" => rmin, "zmax" => zmax, "zmin" => zmin, "rr" => rr, "zz" => zz,
    # Iteration
    "maxit" => maxit, "crit" => crit, "Delta" => Delta,
    # Results
    "cons" => c, "h1" => h1, "h2" => h2, "hz" => hz,"d1" => d1, "d2" => d2,"k1a" => k1a, "k2a"=> k2a,
    "V0" => V0, "V" => V, "Vr" => Vr, "Vz" => Vz, "val" => val,"dr" => dr, "dz" => dz,
    "mu_1" => mu_1, "mu_r" => mu_r, "mu_z" => mu_z,
    "g" => g, # stationary density
    )
    npzwrite(outputdir*"res.npz", results)
end
