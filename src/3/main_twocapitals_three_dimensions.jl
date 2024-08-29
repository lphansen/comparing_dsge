ENV["JULIA_PKG_OFFLINE"] = "true"
Pkg.activate(".")
Pkg.instantiate()

using Pkg
using Optim
using Roots
using NPZ
using ArgParse

function main(Delta=1.0, gamma=8.0, rho=1.00001, alpha=0.1844, kappa=0.0, zeta=0.5, beta1=0.01, beta2=0.01, action_name="output")

    include("utils_twocapitals_three_dimensions.jl")

    outputdir = "./output/"*action_name*"/Delta_"*string(Delta)*"/beta1_"*string(beta1)*"_beta2_"*string(beta2)*"/kappa_"*string(kappa)*"_zeta_"*string(zeta)*"/gamma_"*string(gamma)*"_rho_"*string(rho)*"_alpha_"*string(alpha)*"/"
    isdir(outputdir) || mkpath(outputdir)

    ## Calibration
    delta = 0.01    ## discount rate
    a11 = 0.056     ## long run risk persistence (beta_1 in the paper)
    a22 = 0.194     ## stochastic volatility persistence (beta_2 in the paper)
    eta1 = 0.04     ## depreciation rate (eta_k in the paper)
    eta2 = 0.04     ## depreciation rate (eta_k in the paper)
    phi1 = 8.0      ## adjustment cost
    phi2 = 8.0      ## adjustment cost
    smean = 6.30e-06    ## stochastic volatility mean
    sigma_k1 = [sqrt(2)*0.92,.0, .4,  0.0]  * sqrt(12)      ## shock exposure in the capital 1 evolution process
    sigma_k2 = [0, sqrt(2)*0.92, .4,  0.0]  * sqrt(12)      ## shock exposure in the capital 2 evolution process
    sigma_z =  [0.0  , 0.0,     5.7,  0.0]  * sqrt(12)      ## shock exposure in the long run risk evolution process
    sigma_s =  [0.0  , 0.0,     0.0, 0.00031]* sqrt(12)     ## shock exposure in the stochastic volatility process

    ## Construct state space grid
    II, JJ, SS = trunc(Int,301), trunc(Int,41), trunc(Int,21); ## number of grid points for the endogenous state variable (log K2 - logK1), the long run risk state variable, and the stochastic volatility state variable
    rmax = kappa == 0.0 ? 6.0 : 1.0             ## upper bound of the endogenous state variable (log K2 - logK1)
    rmin = -rmax;                               ## lower bound of the endogenous state variable (log K2 - logK1)
    zmax = 1.0;                                 ## upper bound of the long run risk state variable
    zmin = -zmax;                               ## lower bound of the long run risk state variable
    smax = 2e-05;                               ## upper bound of the stochastic volatility state variable
    smin = 5e-07;                               ## lower bound of the stochastic volatility state variable

    maxit = 200000;        # maximum number of iterations in the HJB loop
    crit  = 10e-6;         # criterion HJB loop

    # Initialize model objects 
    baseline1 = Baseline(a11, a22, zeta, kappa, sigma_z, sigma_s, beta1, eta1, sigma_k1, delta);
    baseline2 = Baseline(a11, a22, zeta, kappa, sigma_z, sigma_s, beta2, eta2, sigma_k2, delta);
    technology1 = Technology(alpha, phi1);
    technology2 = Technology(alpha, phi2);
    model = TwoCapitalEconomy(baseline1, baseline2, technology1, technology2);

    ## Initialize grid and FDM
    grid = Grid_rz(rmin, rmax, II, zmin, zmax, JJ, smin, smax, SS);
    params = FinDiffMethod(maxit, crit, Delta);

    ## Initialize value function and make a guess for consumption-capital ratio
    preload_action = "twocap_model_without_stochastic_volatility"
    preload_Delta = beta1 == 0.04 ? 1.0 : 0.1

    preloaddir = "./output/"*preload_action*"/Delta_"*string(preload_Delta)*"/beta1_"*string(beta1)*"_beta2_"*string(beta2)*"/kappa_"*string(kappa)*"_zeta_"*string(zeta)*"/gamma_"*string(gamma)*"_rho_"*string(rho)*"_alpha_"*string(alpha)*"/"
    preload = npzread(preloaddir*"res.npz")
    println("preload location : "*preloaddir)
    preloadV0 = preload["V"]
    preloadcons = preload["cons"]
    preload_II = 1001
    preload_JJ = 201
    A_x1 = range(rmin,rmax,trunc(Int,preload_II))
    A_x2 = range(zmin,zmax,trunc(Int,preload_JJ))
    A_rr = range(rmin, stop=rmax, length=II);
    A_zz = range(zmin, stop=zmax, length=JJ);

    preloadV0 = ones(II, JJ, SS)
    itp = interpolate(preload["V"], BSpline(Cubic(Line(OnGrid()))))
    sitp = scale(itp, A_x1, A_x2)
    preloadV0tdm = ones(II, JJ)
    for i = 1:II
        for j = 1:JJ
            preloadV0tdm[i,j] = sitp(A_rr[i], A_zz[j])
        end
    end
    for i = 1:SS
        preloadV0[:,:,i] = preloadV0tdm 
    end

    preloadcons = ones(II, JJ, SS)
    itp = interpolate(preload["cons"], BSpline(Cubic(Line(OnGrid()))))
    sitp = scale(itp, A_x1, A_x2)
    preloadconstdm = ones(II, JJ)
    for i = 1:II
        for j = 1:JJ
            preloadconstdm[i,j] = sitp(A_rr[i], A_zz[j])
        end
    end
    for i = 1:SS
        preloadcons[:,:,i] = preloadconstdm 
    end

    ## Solve the model
    times = @elapsed begin
        A, V, val, d1_F, d2_F, d1_B, d2_B, h1_F, h2_F, hz_F, hs_F, h1_B, h2_B, hz_B, hs_B,
               mu_1_F, mu_1_B, mu_r_F, mu_r_B, mu_z, mu_s, V0, Vr, Vz, Vs, Vr_F, Vr_B, Vz_B, Vz_F, Vs_B, Vs_F, cF, cB, rrr, zzz, sss, pii, dr, dz, ds =
            value_function_twocapitals(gamma, rho, model, grid, params, preloadV0, preloadcons, beta1, outputdir,smean);
    end
    println("Convergence time (minutes): ", times/60)
    g = stationary_distribution(A, grid)

    # Construct drift terms under the baseline
    mu_1 = (mu_1_F + mu_1_B)/2.;    ## drift term for composite capital evolution process
    mu_r = (mu_r_F + mu_r_B)/2.;    ## drift term for the endogenous state variable (log K2 - logK1) evolution process
    h1 = (h1_F + h1_B)/2.;          ## Robust control
    h2 = (h2_F + h2_B)/2.;          ## Robust control
    hz = (hz_F + hz_B)/2.;          ## Robust control
    hs = (hs_F + hs_B)/2.;          ## Robust control
    d1 = (d1_F + d1_B)/2;           ## Investment capital 1 ratio
    d2 = (d2_F + d2_B)/2;           ## Investment capital 2 ratio

    r = range(rmin, stop=rmax, length=II);    
    rr = r * ones(1, JJ);
    rrr = ones(II, JJ, SS)
    for i = 1:SS
        rrr[:,:,i] = rr
    end
    IJS = II*JJ*SS;
    k1a = zeros(II,JJ,SS)
    k2a = zeros(II,JJ,SS)
    for i=1:IJS
        p = rrr[i];
        k1a[i] = (1-zeta + zeta*exp.(p*(1-kappa))).^(1/(kappa-1));
        k2a[i] = ((1-zeta)*exp.(p*((kappa-1))) + zeta).^(1/(kappa-1));
    end
    d1k = d1.*k1a
    d2k = d2.*k2a
    c = alpha*ones(II,JJ,SS) - d1k - d2k    ## consumption capital ratio

    ## Save results
    results = Dict(
    # Parameters        
    "delta" => delta, "eta1" => eta1, "eta2" => eta2, "a11"=> a11, "a22"=> a22,  "beta1" => beta1, "beta2" => beta2,
    "sigma_k1" => sigma_k1, "sigma_k2" => sigma_k2, "sigma_z" =>  sigma_z, "sigma_s" =>  sigma_s, "smean" => smean,
    "alpha" => alpha, "kappa" => kappa, "zeta" => zeta, "phi1" => phi1, "phi2" => phi2, 
    "gamma" => gamma, "rho" => rho,
    # Grid
    "I" => II, "J" => JJ, "S" => SS,"rmax" => rmax, "rmin" => rmin, "zmax" => zmax, "zmin" => zmin,"rrr" => rrr, "zzz" => zzz, "sss" => sss,
    # Iteration
    "maxit" => maxit, "crit" => crit, "Delta" => Delta,
    # Results
    "cons" => c,"h1" => h1, "h2" => h2, "hz" => hz, "hs" => hs, "k1a" => k1a, "k2a"=> k2a,
    "V0" => V0, "V" => V, "Vr" => Vr, "Vz" => Vz, "Vs" => Vs, "dr" => dr, "dz" => dz,
    "d1" => d1, "d2" => d2, "d1k" => d1k, "d2k"=> d2k,
    "mu_1" => mu_1, "mu_r" => mu_r, "mu_z" => mu_z, "mu_s" => mu_s,
    "g" => g, # stationary density
    )
    npzwrite(outputdir*"res.npz", results)
end
