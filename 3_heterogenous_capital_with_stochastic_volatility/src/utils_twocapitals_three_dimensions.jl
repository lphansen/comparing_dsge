using LinearAlgebra
using SparseArrays
using Interpolations
using SuiteSparse
using NPZ
using HDF5

mutable struct Baseline{T}
    a11::T
    a22::T
    zeta::T
    kappa::T
    sigma_z::Array{T, 1}
    sigma_s::Array{T, 1}

    beta::T
    eta::T
    sigma_k::Array{T, 1}

    delta::T
end

mutable struct Technology{T}
    alpha::T
    phi::T
end

mutable struct TwoCapitalEconomy{T}
    k1::Baseline{T}
    k2::Baseline{T}

    t1::Technology{T}
    t2::Technology{T}
end

mutable struct Grid_rz{T}
  rmin::T
  rmax::T
  I::Integer           # number of z points
  zmin::T
  zmax::T
  J::Integer           # number of z points
  smin::T
  smax::T
  S::Integer           # number of z points
end

mutable struct FinDiffMethod
  maxit::Integer      # maximum number of iterations in the HJB loop
  crit::Float64       # criterion HJB loop
  Delta::Float64      # delta in HJB algorithm
end

mutable struct PolicyFunctions
  d1_F::Array{Float64,3}
  d2_F::Array{Float64,3}
  d1_B::Array{Float64,3}
  d2_B::Array{Float64,3}

  h1_F::Array{Float64,3}
  h2_F::Array{Float64,3}
  hz_F::Array{Float64,3}
  hs_F::Array{Float64,3}

  h1_B::Array{Float64,3}
  h2_B::Array{Float64,3}
  hz_B::Array{Float64,3}
  hs_B::Array{Float64,3}

end

function dstar_twocapitals!(d1::Array{Float64,3},
                            d2::Array{Float64,3},
                            Vr::Array{Float64,3},
                            rrr::Array{Float64,3},
                            zzz::Array{Float64,3},
                            sss::Array{Float64,3},
                            IJS::Int64,
                            rho::Float64,
                            V::Array{Float64,3},
                            model::TwoCapitalEconomy{Float64},
                            C::Array{Float64,3})

    # Check for the positivity of the critical term in the quad root formula
    # This might not hold if the technology parameters are weird
    alpha = model.t1.alpha; 
    zeta = model.k1.zeta;
    kappa = model.k1.kappa;
    delta = model.k1.delta;
    phi1 = model.t1.phi;
    phi2 = model.t2.phi;

    for i=1:IJS
        p, vr, z, s = rrr[i], Vr[i], zzz[i], sss[i]
        if kappa ==1.0
            k1a = exp.(p*(-zeta))
            k2a = exp.(p*(1-zeta))
        else
            k1a = (1-zeta + zeta*exp.(p*(1-kappa))).^(1/(kappa-1));
            k2a = ((1-zeta)*exp.(p*(kappa-1)) + zeta).^(1/(kappa-1));
        end

        c_constant_1 = 1/phi1*k1a 
        c_constant_2 = 1/phi2*k2a
        c_constant = alpha + c_constant_1 + c_constant_2
        c_rho_coeff_1 = ((1-zeta)*k1a.^(1-kappa)-Vr[i])/phi1/(delta*exp.(V[i]*(rho-1)))
        c_rho_coeff_2 = (zeta*(k2a).^(1-kappa)+Vr[i])/phi2/(delta*exp.(V[i]*(rho-1)))
        c_rho_coeff = c_rho_coeff_1 + c_rho_coeff_2
        clowerlim = 0.0001
        function g(cx)
            if cx < clowerlim
                cx = clowerlim
            end
            return c_constant - c_rho_coeff * cx.^(rho) -cx
        end

        if rho == 1.0            
            croot = c_constant/(c_rho_coeff+1)

            if croot < clowerlim
                croot = clowerlim
            end
            d1_root = c_rho_coeff_1*croot/k1a - 1/phi1
            d2_root = c_rho_coeff_2*croot/k2a - 1/phi2

        else
            x0 = C[i];
            croot = find_zero(g, x0, Roots.Order1(), maxiters = 100000000000, xatol = 10e-15, xrtol = 10e-15, atol = 10e-18, rtol = 10e-18, strict = true);
            if croot < clowerlim
                croot = clowerlim
            end
            d1_root = c_rho_coeff_1*croot.^(rho)/k1a - 1/phi1
            d2_root = c_rho_coeff_2*croot.^(rho)/k2a - 1/phi2
            C[i] = croot;
        end
        d1[i] = d1_root;
        d2[i] = d2_root;
    end

    nothing
end


function upwind_transform!(var::Array{Float64, 3},
                           var_F::Array{Float64, 3},
                           var_B::Array{Float64, 3},
                           drift_F::Array{Float64, 3},
                           drift_B::Array{Float64, 3},
                           IJS::Int64,
                           dim::Integer=1)

    If = drift_F .>= 0.;
    Ib = .!If .& (drift_B .< 0.);

    if dim ==1
      If[1, :, :] .= 1.;    # force to use forward at the first row
      If[end, :, :] .= 0.;
      Ib[end, :, :] .= 1.;  # force to use backward at the last row
      Ib[1, :, :] .= 0.;
    elseif dim==2
      If[:, 1, :] .= 1.;    # force to use forward at the first col
      If[:, end, :] .= 0.;
      Ib[:, end, :] .= 1.;  # force to use backward at the last col
      Ib[:, 1, :] .= 0.;
    elseif dim==3
        If[:, :, 1] .= 1.;    # force to use forward at the first col
        If[:, :, end] .= 0.;
        Ib[:, :, end] .= 1.;  # force to use backward at the last col
        Ib[:, :, 1] .= 0.;
    end

    for i=1:IJS
        I0 = (If[i] + Ib[i]) == 0;  # when (drift_F < 0 and drift_B > 0)
        var[i] = If[i]*var_F[i] + Ib[i]*var_B[i] + I0*var_F[i];
    end

    nothing
end



function drifts!(mu_1::Array{Float64, 3},
                 mu_r::Array{Float64, 3},
                 d1::Array{Float64, 3},
                 d2::Array{Float64, 3},
                 zzz::Array{Float64, 3},
                 sss::Array{Float64, 3},
                 rrr::Array{Float64, 3},
                 IJS::Int64,
                 model::TwoCapitalEconomy)

    phi1, phi2 = model.t1.phi, model.t2.phi;

    zeta = model.k1.zeta;
    kappa = model.k1.kappa;
    beta1 = model.k1.beta;
    beta2 = model.k2.beta;
    eta1 = model.k1.eta;
    eta2 = model.k2.eta;
    s_k1 = model.k1.sigma_k;
    s_k2 = model.k2.sigma_k;

    for i=1:IJS
        p, z, s = rrr[i], zzz[i], sss[i];
        if kappa == 1.0
            k1a = exp.(p*(-zeta))
            k2a = exp.(p*(1-zeta))
        else
            k1a = (1-zeta + zeta*exp.(p*(1-kappa))).^(1/(kappa-1));
            k2a = ((1-zeta)*exp.(p*(kappa-1)) + zeta).^(1/(kappa-1));
        end

        dkadk1dk1 = (kappa-1)*(1-zeta).^2*(k1a).^(-2*kappa+2) - kappa*(1-zeta)*(k1a).^(-kappa+1);
        dkadk1dk2 = (kappa-1)*zeta*(1-zeta)*(k1a).^(-kappa+1)*(k2a).^(-kappa+1);
        dkadk2dk2 = (kappa-1)*zeta.^2*(k2a).^(-2*kappa+2) - kappa*(1-zeta)*(k2a).^(-kappa+1)

        mu_k1 = log(1+phi1*d1[i])/phi1 + beta1*z - eta1;
        mu_k2 = log(1+phi2*d2[i])/phi2 + beta2*z - eta2;

        mu_r[i] = mu_k2 - mu_k1 - (s/2)*(dot(s_k2,s_k2) - dot(s_k1,s_k1)) 

        mu_1[i] = mu_k1*(1-zeta)*(k1a).^(1-kappa)+
                    mu_k2*(zeta)*(k2a).^(1-kappa)+
                    s/2*(dot(s_k1,s_k1)*dkadk1dk1 + dot(s_k2,s_k2)*dkadk2dk2 + 2*dot(s_k1,s_k2)*dkadk1dk2)  
    end

    nothing

end


function drifts_distortion!(h::Array{Float64, 3},
                            s_k1::Float64,
                            s_k2::Float64,
                            s_z::Float64,
                            s_s::Float64,
                            IJS::Int64,
                            rrr::Array{Float64, 3},
                            Vr::Array{Float64, 3},
                            Vz::Array{Float64, 3},
                            Vs::Array{Float64, 3},
                            model::TwoCapitalEconomy)

    zeta = model.k1.zeta;
    kappa = model.k1.kappa;

    for i=1:IJS
        p = rrr[i];
        if kappa ==1.0
            k1a = exp.(p*(-zeta))
            k2a = exp.(p*(1-zeta))
        else
            k1a = (1-zeta + zeta*exp.(p*(1-kappa))).^(1/(kappa-1));
            k2a = ((1-zeta)*exp.(p*(kappa-1)) + zeta).^(1/(kappa-1));
        end
        h[i] = (s_k1*(1-zeta)*(k1a).^(1-kappa) + s_k2*(zeta)*(k2a).^(1-kappa) + (s_k2-s_k1)*Vr[i]) + s_z*Vz[i] + s_s*Vs[i];
    end

    nothing
end


function create_uu!(uu::Array{Float64, 1},
                    gamma::Float64,
                    rho::Float64,
                    d1::Array{Float64, 3},
                    d2::Array{Float64, 3},
                    h1::Array{Float64, 3},
                    h2::Array{Float64, 3},
                    hz::Array{Float64, 3},
                    hs::Array{Float64, 3},
                    mu_1::Array{Float64, 3},
                    rrr::Array{Float64, 3},
                    zzz::Array{Float64, 3},
                    sss::Array{Float64, 3},
                    IJS::Int64,
                    V::Array{Float64, 3},
                    model::TwoCapitalEconomy)

    alpha = model.t1.alpha;
    delta = model.k1.delta;
    zeta = model.k1.zeta;
    kappa = model.k1.kappa;

    for i=1:IJS
        p, z, s = rrr[i], zzz[i], sss[i]
        if kappa == 1.0
            k1a = exp.(p*(-zeta))
            k2a = exp.(p*(1-zeta))
        else
            k1a = (1-zeta + zeta*exp.(p*(1-kappa))).^(1/(kappa-1));
            k2a = ((1-zeta)*exp.(p*(kappa-1)) + zeta).^(1/(kappa-1));
        end
        c = alpha - d1[i]*k1a - d2[i]*k2a;

        penalty_term = (1-gamma)*s*(h1[i]^2 + h2[i]^2 + hz[i]^2 + hs[i]^2)/2;
        if rho == 1.0
            uu[i] = (delta*log(c) + penalty_term + mu_1[i]);
        else
            uu[i] = (delta/(1-rho)*(c.^(1-rho)*exp.((rho-1)*V[i])-1)+ penalty_term + mu_1[i]);
        end
    end

    nothing
end



function create_Aval!(Aval, d_, c_, e_, b_, f_, g_, h_, a_1, a_2, a_3, a_4, a_5, a_6, II, JJ, SS)

    iter = 1
    j = 1
    Aval[iter] = d_[j]; iter += 1;
    Aval[iter] = e_[1+j]; iter += 1;
    Aval[iter] = f_[II+j]; iter += 1;
    Aval[iter] = a_1[II+1+j]; iter += 1;
    Aval[iter] = h_[II*JJ+j]; iter += 1;
    Aval[iter] = a_3[II*JJ+1+j]; iter += 1;
    Aval[iter] = a_5[II*JJ+II+j]; iter += 1;

    for j=2:II
        Aval[iter] = c_[-1+j]; iter += 1;
        Aval[iter] = d_[j]; iter += 1;
        Aval[iter] = e_[1+j]; iter += 1;
        Aval[iter] = f_[II+j]; iter += 1;
        Aval[iter] = a_1[II+1+j]; iter += 1;
        Aval[iter] = h_[II*JJ+j]; iter += 1;
        Aval[iter] = a_3[II*JJ+1+j]; iter += 1;
        Aval[iter] = a_5[II*JJ+II+j]; iter += 1;
    end

    j = II+1
    Aval[iter] = b_[-II+j]; iter += 1;
    Aval[iter] = c_[-1+j]; iter += 1;
    Aval[iter] = d_[j]; iter += 1;
    Aval[iter] = e_[1+j]; iter += 1;
    Aval[iter] = f_[II+j]; iter += 1;
    Aval[iter] = a_1[II+1+j]; iter += 1;
    Aval[iter] = h_[II*JJ+j]; iter += 1;
    Aval[iter] = a_3[II*JJ+1+j]; iter += 1;
    Aval[iter] = a_5[II*JJ+II+j]; iter += 1;

    for j=(II+2):(II*JJ)
        Aval[iter] = a_2[-(II+1)+j]; iter += 1;
        Aval[iter] = b_[-II+j]; iter += 1;
        Aval[iter] = c_[-1+j]; iter += 1;
        Aval[iter] = d_[j]; iter += 1;
        Aval[iter] = e_[1+j]; iter += 1;
        Aval[iter] = f_[II+j]; iter += 1;
        Aval[iter] = a_1[II+1+j]; iter += 1;
        Aval[iter] = h_[II*JJ+j]; iter += 1;
        Aval[iter] = a_3[II*JJ+1+j]; iter += 1;
        Aval[iter] = a_5[II*JJ+II+j]; iter += 1;
    end

    j = II*JJ+1
    Aval[iter] = g_[-(II*JJ)+j]; iter += 1;
    Aval[iter] = a_2[-(II+1)+j]; iter += 1;
    Aval[iter] = b_[-II+j]; iter += 1;
    Aval[iter] = c_[-1+j]; iter += 1;
    Aval[iter] = d_[j]; iter += 1;
    Aval[iter] = e_[1+j]; iter += 1;
    Aval[iter] = f_[II+j]; iter += 1;
    Aval[iter] = a_1[II+1+j]; iter += 1;
    Aval[iter] = h_[II*JJ+j]; iter += 1;
    Aval[iter] = a_3[II*JJ+1+j]; iter += 1;
    Aval[iter] = a_5[II*JJ+II+j]; iter += 1;

    for j=(II*JJ+2):(II*JJ+II)
        Aval[iter] = a_4[-(II*JJ+1)+j]; iter += 1;
        Aval[iter] = g_[-(II*JJ)+j]; iter += 1;
        Aval[iter] = a_2[-(II+1)+j]; iter += 1;
        Aval[iter] = b_[-II+j]; iter += 1;
        Aval[iter] = c_[-1+j]; iter += 1;
        Aval[iter] = d_[j]; iter += 1;
        Aval[iter] = e_[1+j]; iter += 1;
        Aval[iter] = f_[II+j]; iter += 1;
        Aval[iter] = a_1[II+1+j]; iter += 1;
        Aval[iter] = h_[II*JJ+j]; iter += 1;
        Aval[iter] = a_3[II*JJ+1+j]; iter += 1;
        Aval[iter] = a_5[II*JJ+II+j]; iter += 1;
    end

    for j=(II*JJ+II+1):(II*JJ*SS-II*JJ-II)
        Aval[iter] = a_6[-(II*JJ+II)+j]; iter += 1;
        Aval[iter] = a_4[-(II*JJ+1)+j]; iter += 1;
        Aval[iter] = g_[-(II*JJ)+j]; iter += 1;
        Aval[iter] = a_2[-(II+1)+j]; iter += 1;
        Aval[iter] = b_[-II+j]; iter += 1;
        Aval[iter] = c_[-1+j]; iter += 1;
        Aval[iter] = d_[j]; iter += 1;
        Aval[iter] = e_[1+j]; iter += 1;
        Aval[iter] = f_[II+j]; iter += 1;
        Aval[iter] = a_1[II+1+j]; iter += 1;
        Aval[iter] = h_[II*JJ+j]; iter += 1;
        Aval[iter] = a_3[II*JJ+1+j]; iter += 1;
        Aval[iter] = a_5[II*JJ+II+j]; iter += 1;
    end

    for j=(II*JJ*SS-II*JJ-II+1):(II*JJ*SS-II*JJ-1)
        Aval[iter] = a_6[-(II*JJ+II)+j]; iter += 1;
        Aval[iter] = a_4[-(II*JJ+1)+j]; iter += 1;
        Aval[iter] = g_[-(II*JJ)+j]; iter += 1;
        Aval[iter] = a_2[-(II+1)+j]; iter += 1;
        Aval[iter] = b_[-II+j]; iter += 1;
        Aval[iter] = c_[-1+j]; iter += 1;
        Aval[iter] = d_[j]; iter += 1;
        Aval[iter] = e_[1+j]; iter += 1;
        Aval[iter] = f_[II+j]; iter += 1;
        Aval[iter] = a_1[II+1+j]; iter += 1;
        Aval[iter] = h_[II*JJ+j]; iter += 1;
        Aval[iter] = a_3[II*JJ+1+j]; iter += 1;
    end

    j = II*JJ*SS-II*JJ
    Aval[iter] = a_6[-(II*JJ+II)+j]; iter += 1;
    Aval[iter] = a_4[-(II*JJ+1)+j]; iter += 1;
    Aval[iter] = g_[-(II*JJ)+j]; iter += 1;
    Aval[iter] = a_2[-(II+1)+j]; iter += 1;
    Aval[iter] = b_[-II+j]; iter += 1;
    Aval[iter] = c_[-1+j]; iter += 1;
    Aval[iter] = d_[j]; iter += 1;
    Aval[iter] = e_[1+j]; iter += 1;
    Aval[iter] = f_[II+j]; iter += 1;
    Aval[iter] = a_1[II+1+j]; iter += 1;
    Aval[iter] = h_[II*JJ+j]; iter += 1;
    
    for j=(II*JJ*SS-II*JJ+1):(II*JJ*SS-II-1)
        Aval[iter] = a_6[-(II*JJ+II)+j]; iter += 1;
        Aval[iter] = a_4[-(II*JJ+1)+j]; iter += 1;
        Aval[iter] = g_[-(II*JJ)+j]; iter += 1;
        Aval[iter] = a_2[-(II+1)+j]; iter += 1;
        Aval[iter] = b_[-II+j]; iter += 1;
        Aval[iter] = c_[-1+j]; iter += 1;
        Aval[iter] = d_[j]; iter += 1;
        Aval[iter] = e_[1+j]; iter += 1;
        Aval[iter] = f_[II+j]; iter += 1;
        Aval[iter] = a_1[II+1+j]; iter += 1;
    end

    j = II*JJ*SS-II
    Aval[iter] = a_6[-(II*JJ+II)+j]; iter += 1;
    Aval[iter] = a_4[-(II*JJ+1)+j]; iter += 1;
    Aval[iter] = g_[-(II*JJ)+j]; iter += 1;
    Aval[iter] = a_2[-(II+1)+j]; iter += 1;
    Aval[iter] = b_[-II+j]; iter += 1;
    Aval[iter] = c_[-1+j]; iter += 1;
    Aval[iter] = d_[j]; iter += 1;
    Aval[iter] = e_[1+j]; iter += 1;
    Aval[iter] = f_[II+j]; iter += 1;

    for j=(II*JJ*SS-II+1):(II*JJ*SS-1)
        Aval[iter] = a_6[-(II*JJ+II)+j]; iter += 1;
        Aval[iter] = a_4[-(II*JJ+1)+j]; iter += 1;
        Aval[iter] = g_[-(II*JJ)+j]; iter += 1;
        Aval[iter] = a_2[-(II+1)+j]; iter += 1;
        Aval[iter] = b_[-II+j]; iter += 1;
        Aval[iter] = c_[-1+j]; iter += 1;
        Aval[iter] = d_[j]; iter += 1;
        Aval[iter] = e_[1+j]; iter += 1;
    end

    j = II*JJ*SS
    Aval[iter] = a_6[-(II*JJ+II)+j]; iter += 1;
    Aval[iter] = a_4[-(II*JJ+1)+j]; iter += 1;
    Aval[iter] = g_[-(II*JJ)+j]; iter += 1;
    Aval[iter] = a_2[-(II+1)+j]; iter += 1;
    Aval[iter] = b_[-II+j]; iter += 1;
    Aval[iter] = c_[-1+j]; iter += 1;
    Aval[iter] = d_[j]; iter += 1;

    nothing
end



function value_function_twocapitals(gamma::Float64,
                                    rho::Float64,
                                    model::TwoCapitalEconomy,
                                    grid::Grid_rz,
                                    params::FinDiffMethod,
                                    preloadV0::Array{Float64, 3},
                                    preloadcons::Array{Float64, 3},
                                    beta1::Float64,
                                    outputdir::String,
                                    smean::Float64)

      rmin, rmax, II = grid.rmin, grid.rmax, grid.I;
      zmin, zmax, JJ = grid.zmin, grid.zmax, grid.J;
      smin, smax, SS = grid.smin, grid.smax, grid.S;

      IJS = II*JJ*SS;
      II_half = convert(Integer, round(II/2));
      JJ_half = convert(Integer, round(JJ/2));
      SS_half = convert(Integer, round(SS/2));
    
      maxit  = params.maxit;       # max number of iterations in the HJB loop
      crit = params.crit;          # criterion HJB loop
      Delta = params.Delta;        # Delta in HJB algorithm
      
      r = range(rmin, stop=rmax, length=II);    # capital ratio vector
      dr = (rmax - rmin)/(II-1);
      z = range(zmin, stop=zmax, length=JJ)';   # productivity vector
      dz = (zmax - zmin)/(JJ-1);
      s = range(smin, stop=smax, length=SS)';   # stochastic volatility vector
      ds = (smax - smin)/(SS-1);
      dr2, dz2, ds2, drdz, drds, dzds = dr*dr, dz*dz, ds*ds, dr*dz, dr*ds, dz*ds;

      println("rmax = ", rmax, ", rmin = ",rmin, ", rlength = ", II)
      println("zmax = ", zmax, ", zmin = ",zmin, ", zlength = ", JJ)
      println("smax = ", smax, ", smin = ",smin, ", slength = ", SS)

      rr = r * ones(1, JJ);
      rrr = ones(II, JJ, SS)
      for i = 1:SS
            rrr[:,:,i] = rr
      end
      zz = ones(II, 1) * z;
      zzz = ones(II, JJ, SS)
      for i = 1:SS
            zzz[:,:,i] = zz
      end
      sss = ones(II, JJ, SS)
      for i = 1:SS
            sss[:,:,i] = s[i] * ones(II,JJ)
      end

      #========================================================================#
      # Storing matrices
      #========================================================================#
      # Value function and forward/backward finite difference matrices
      V, V0 = zeros(II, JJ, SS), zeros(II, JJ, SS);

      # These matrices need to compute choices
      Vr_F, Vr_B = zeros(II, JJ, SS), zeros(II, JJ, SS);
      Vz_F, Vz_B = zeros(II, JJ, SS), zeros(II, JJ, SS);
      Vs_F, Vs_B = zeros(II, JJ, SS), zeros(II, JJ, SS);
      Vr, Vz, Vs = zeros(II, JJ, SS), zeros(II, JJ, SS), zeros(II, JJ, SS);

      # Choice variables (capital ratio and worst-case drift)
      d1_F, d2_F = zeros(II, JJ, SS), zeros(II, JJ, SS);
      d1_B, d2_B = zeros(II, JJ, SS), zeros(II, JJ, SS);
      h1_F, h1_B  = zeros(II, JJ, SS), zeros(II, JJ, SS);
      h2_F, h2_B  = zeros(II, JJ, SS), zeros(II, JJ, SS);
      hz_F, hz_B  = zeros(II, JJ, SS), zeros(II, JJ, SS);
      hs_F, hs_B  = zeros(II, JJ, SS), zeros(II, JJ, SS);

      # Choice variables (capital ratio and worst-case drift)
      d1, d2 = zeros(II, JJ, SS), zeros(II, JJ, SS);
      h1, h2, hz, hs = zeros(II, JJ, SS), zeros(II, JJ, SS), zeros(II, JJ, SS), zeros(II, JJ, SS);

      # Drifts
      mu_1_F, mu_1_B, mu_1 = zeros(II, JJ, SS), zeros(II, JJ, SS), zeros(II, JJ, SS);
      mu_r_F, mu_r_B = zeros(II, JJ, SS), zeros(II, JJ, SS);

      uu = zeros(II*JJ*SS);
      A = spdiagm( 0 => ones(II*JJ*SS),
                   1 => ones(II*JJ*SS-1),
                  -1 => ones(II*JJ*SS-1),
                  II => ones(II*(JJ*SS-1)),
                 -II => ones(II*(JJ*SS-1)),
               -II-1 => ones(II*(JJ*SS-1)-1),
                II+1 => ones(II*(JJ*SS-1)-1),
              -II*JJ => ones(II*JJ*(SS-1)),
               II*JJ => ones(II*JJ*(SS-1)),
            -II*JJ-1 => ones(II*JJ*(SS-1)-1),
             II*JJ+1 => ones(II*JJ*(SS-1)-1),
           -II*JJ-II => ones(II*JJ*(SS-1)-II),
            II*JJ+II => ones(II*JJ*(SS-1)-II));
      Aval = zeros(nnz(A))

      #========================================================================#
      # MODEL PARAMETERS                                                       #
      #========================================================================#
      delta = model.k1.delta;
      a11 = model.k1.a11;
      a22 = model.k1.a22;
      zeta = model.k1.zeta;
      kappa = model.k1.kappa;

      eta1 = model.k1.eta;
      eta2 = model.k2.eta;
      beta1 = model.k1.beta;
      beta2 = model.k2.beta;

      s_k1 = model.k1.sigma_k;
      s_k2 = model.k2.sigma_k;
      s_z =  model.k1.sigma_z;
      s_s =  model.k1.sigma_s;

      alpha = model.t1.alpha;
      phi1, phi2 = model.t1.phi, model.t2.phi;

      t1 = dot(s_k2 - s_k1, s_k2 - s_k1)/(2*dr2);
      t2 = dot(s_k2 - s_k1, s_z)/(2*drdz);
      t3 = dot(s_z, s_z)/(2*dz2);

      t4 = dot(s_k2 - s_k1, s_s)/(2*drds);
      t5 = dot(s_z, s_s)/(2*dzds);
      t6 = dot(s_s, s_s)/(2*ds2);

      #========================================================================#
      # INITIALIZATION                                                         #
      #========================================================================#
    
      V0 = preloadV0;
      C0F = preloadcons;
      C0B = preloadcons;
      v = copy(V0);
      cF = copy(C0F);
      cB = copy(C0B);
      distance = zeros(maxit);

      #========================================================================#
      # HAMILTON-JACOBI-BELLMAN EQUATION
      #========================================================================#
      mu_z = ones(II, JJ, SS);
      for i = 1:SS
        mu_z[:,:,i] = -a11*zz
      end
      mu_s = ones(II, JJ, SS);
      for i = 1:SS
        mu_s[:,:,i] = -a22 * (s[i]-smean) * ones(II,JJ)
      end

      results_list = []
      converge = false
      while converge == false
        try 
            if rho == 1.0
                I_delta = sparse((1/Delta + delta)*I, IJS, IJS);
            else
                I_delta = sparse((1/Delta)*I, IJS, IJS);
            end

            for n=1:maxit

                V = copy(v);
                CF = copy(cF)
                CB = copy(cB)
                V_stacked = vec(V);

                # forward diff (last row never used - known value function there)
                # backward diff (first row never used - known value function there)
                # Diff in the z dimension: 1st/last col = 0 imposed
                Vr_B[2:II, :, :] = Vr_F[1:II-1, :, :] = (V[2:II, :, :] - V[1:II-1, :, :])./dr;
                Vz_B[:, 2:JJ, :] = Vz_F[:, 1:JJ-1, :] = (V[:, 2:JJ, :] - V[:, 1:JJ-1, :])./dz;
                Vs_B[:, :, 2:SS] = Vs_F[:, :, 1:SS-1] = (V[:, :, 2:SS] - V[:, :, 1:SS-1])./ds;

                # Investment-capital ratios
                dstar_twocapitals!(d1_F, d2_F, Vr_F, rrr, zzz, sss, IJS, rho, V, model, CF);
                dstar_twocapitals!(d1_B, d2_B, Vr_B, rrr, zzz, sss, IJS, rho, V, model, CB);
                cF = copy(CF)
                cB = copy(CB)

                # Drifts
                drifts!(mu_1_F, mu_r_F, d1_F, d2_F, zzz, sss, rrr, IJS, model);
                drifts!(mu_1_B, mu_r_B, d1_B, d2_B, zzz, sss, rrr, IJS, model);

                upwind_transform!(Vr, Vr_F, Vr_B, mu_r_F, mu_r_B, IJS);
                upwind_transform!(Vz, Vz_F, Vz_B, mu_z  , mu_z  , IJS, 2);
                upwind_transform!(Vs, Vs_F, Vs_B, mu_s  , mu_s  , IJS, 3);
                upwind_transform!(d1, d1_F, d1_B, mu_r_F, mu_r_B, IJS);
                upwind_transform!(d2, d2_F, d2_B, mu_r_F, mu_r_B, IJS);
                upwind_transform!(mu_1, mu_1_F, mu_1_B, mu_r_F, mu_r_B, IJS);

                # Drifts distortion
                drifts_distortion!(h1, s_k1[1], s_k2[1], s_z[1], s_s[1], IJS, rrr, Vr, Vz, Vs, model);
                drifts_distortion!(h2, s_k1[2], s_k2[2], s_z[2], s_s[2], IJS, rrr, Vr, Vz, Vs, model);
                drifts_distortion!(hz, s_k1[3], s_k2[3], s_z[3], s_s[3], IJS, rrr, Vr, Vz, Vs, model);
                drifts_distortion!(hs, s_k1[4], s_k2[4], s_z[4], s_s[4], IJS, rrr, Vr, Vz, Vs, model);

                drifts_distortion!(h1_F, s_k1[1],s_k2[1],s_z[1],s_s[1], IJS,rrr,Vr_F,Vz_F, Vs_F, model);
                drifts_distortion!(h2_F, s_k1[2],s_k2[2],s_z[2],s_s[2], IJS,rrr,Vr_F,Vz_F, Vs_F, model);
                drifts_distortion!(hz_F, s_k1[3],s_k2[3],s_z[3],s_s[3], IJS,rrr,Vr_F,Vz_F, Vs_F, model);
                drifts_distortion!(hs_F, s_k1[4],s_k2[4],s_z[4],s_s[4], IJS,rrr,Vr_F,Vz_F, Vs_F, model);
                drifts_distortion!(h1_B, s_k1[1],s_k2[1],s_z[1],s_s[1], IJS,rrr,Vr_B,Vz_B, Vs_B, model);
                drifts_distortion!(h2_B, s_k1[2],s_k2[2],s_z[2],s_s[2], IJS,rrr,Vr_B,Vz_B, Vs_B, model);
                drifts_distortion!(hz_B, s_k1[3],s_k2[3],s_z[3],s_s[3], IJS,rrr,Vr_B,Vz_B, Vs_B, model);
                drifts_distortion!(hs_B, s_k1[4],s_k2[4],s_z[4],s_s[4], IJS,rrr,Vr_B,Vz_B, Vs_B, model);

                # FLOW TERM
                if beta1 == 0.04
                    create_uu!(uu, gamma, rho, d1_F, d2_F, h1_F, h2_F, hz_F, hs_F, mu_1_F, rrr, zzz, sss, IJS, V, model);
                else
                    create_uu!(uu, gamma, rho, d1, d2, h1, h2, hz, hs, mu_1, rrr, zzz, sss, IJS, V, model);
                end
                
                #CONSTRUCT MATRIX A
                a_1 = sss*t2; 
                a_2 = sss*t2;
                a_3 = sss*t4;
                a_4 = sss*t4;
                a_5 = sss*t5;
                a_6 = sss*t5;

                g_ = max.(mu_s, 0.)/ds .+ (t6 .- t4 .- t5).*sss;
                b_ = max.(mu_z, 0.)/dz .+ (t3 .- t2 .- t5).*sss;
                c_ = max.(mu_r_F, 0.)/dr .+ (t1 .- t2 .- t4).*sss;
                d_ = (-max.(mu_r_F, 0.)/dr + min.(mu_r_B, 0.)/dr - max.(mu_z, 0.)/dz +
                        min.(mu_z, 0.)/dz - max.(mu_s, 0.)/ds +
                        min.(mu_s, 0.)/ds .- 2*(t1 + t3 + t6 - t2 - t4 - t5).*sss);
                e_ = -min.(mu_r_B, 0.)/dr .+ (t1 .- t2 .- t4).*sss;
                f_ = -min.(mu_z, 0.)/dz .+ (t3 .- t2 .- t5).*sss;
                h_ = -min.(mu_s, 0.)/ds .+ (t6 .- t4 .- t5).*sss;

                # Adding reflection boundary in I dimension
                f_[1 , :, :] += a_1[1, :, :];
                h_[1 , :, :] += a_3[1, :, :];
                a_1[1, :, :] .= 0.0;
                a_3[1, :, :] .= 0.0;
                b_[end, :, :] += a_2[end, :, :];
                g_[end, :, :] += a_4[end, :, :];
                a_2[end, :, :] .= 0.0;
                a_4[end, :, :] .= 0.0;

                d_[1, :, :] += e_[1, :, :];
                e_[1, :, :] .= 0.0;
                d_[end, :, :] += c_[end, :, :];
                c_[end, :, :] .= 0.0;

                # Adding reflection boundary in J dimension
                d_[:, 1, :] += f_[:, 1, :];
                e_[:, 1, :] += a_1[:, 1, :];
                h_[:, 1, :] += a_5[:, 1, :];
                f_[:, 1, :] .= 0.0;
                a_1[:, 1, :] .= 0.0;
                a_5[:, 1, :] .= 0.0;
                d_[:, end, :] += b_[:, end, :];
                c_[:, end, :] += a_2[:, end, :];
                g_[:, end, :] += a_6[:, end, :];
                b_[:, end, :] .= 0.0;
                a_2[:, end, :] .= 0.0;
                a_6[:, end, :] .= 0.0;

                # Adding reflection boundary in S dimension
                d_[:, :, 1] += h_[:, :, 1];
                e_[:, :, 1] += a_3[:, :, 1];
                f_[:, :, 1] += a_5[:, :, 1];
                h_[:, :, 1] .= 0.0;
                a_3[:, :, 1] .= 0.0;
                a_5[:, :, 1] .= 0.0;
                d_[:, :, end] += g_[:, :, end];
                c_[:, :, end] += a_4[:, :, end];
                b_[:, :, end] += a_6[:, :, end];
                g_[:, :, end] .= 0.0;
                a_4[:, :, end] .= 0.0;
                a_6[:, :, end] .= 0.0;

                create_Aval!(Aval, d_, c_, e_, b_, f_, g_, h_, a_1, a_2, a_3, a_4, a_5, a_6, II, JJ, SS)
                A.nzval .= Aval;

                #Cblas function for y:=a*x + y (in our case: uu^n + v^n -> uu^n)
                BLAS.axpy!(IJS, 1/Delta, V_stacked, 1, uu, 1);

                # TIME CONSUMING PART
                V_stacked = (I_delta - A) \ uu
                #ldiv!(V_stacked, factorize(I_delta - A), uu);
                #lsmr!(V_stacked, I_delta - A, uu, atol=1e-8, btol=1e-8)

                Vchange = reshape(V_stacked, II, JJ, SS) - v;
                distance[n] = maximum(abs.(Vchange));

                v = reshape(V_stacked, II, JJ, SS);

                println("----------------------------------");
                println("Iteration = ",n);
                println("Distance = ",distance[n]);
                println("v max = ",maximum(v));
                println("v min = ",minimum(v));
                println("----------------------------------")
                
                c = (cF + cB)/2;
                results = Dict("V" => v, "cons" => c,
                  "Vr" => Vr, "Vz" => Vz, "Vs" => Vs, "rrr" => rrr, "zzz" => zzz, "sss" => sss)
                  push!(results_list, results)
                if distance[n]<crit
                    converge = true
                    val = v[II_half, JJ_half, SS_half];
                    println("    Value Function Converged, Iteration = ", n,
                            " with gamma=", gamma,
                            " with rho=", rho,
                            " and v=", val);
                    create_Aval!(Aval, d_, c_, e_, b_, f_, g_, h_, a_1, a_2, a_3, a_4, a_5, a_6, II, JJ, SS)
                    A.nzval .= Aval;
                    break
                end
            end
        catch
            println("An error occurred, when Delta = ", Delta, " try to
                    decrease Delta and try again")
            Delta = Delta/2
            try
                v = results_list[end-1]["V"]
            catch 
                v = copy(V0)
                println("restarting with preload V0")
            end
            pop!(results_list)        
            println("Delta = ", Delta)
            println("v max = ",maximum(v));
            println("v min = ",minimum(v));
        end
    end

    val = v[II_half, JJ_half, SS_half];     # Value at (r_0, z_0)  (objective)

    return A, v, val, d1_F, d2_F, d1_B, d2_B, h1_F, h2_F, hz_F, hs_F, h1_B, h2_B, hz_B, hs_B,
           mu_1_F, mu_1_B, mu_r_F, mu_r_B, mu_z, mu_s, V0, Vr, Vz, Vs, Vr_F, Vr_B, Vz_B, Vz_F, Vs_B, Vs_F, cF, cB, rrr, zzz, sss, rrr, dr, dz, ds;
end



function stationary_distribution(A::SparseMatrixCSC{Float64,Int64},
                                 grid::Grid_rz)

      #=======================================================#
      # Construct grid
      #=======================================================#
      rmin, rmax, I = grid.rmin, grid.rmax, grid.I;
      zmin, zmax, J = grid.zmin, grid.zmax, grid.J;
      smin, smax, S = grid.smin, grid.smax, grid.S;
      r = range(rmin, stop=rmax, length=I);    # capital ratio vector
      dr = (rmax - rmin)/(I-1);
      z = range(zmin, stop=zmax, length=J)';   # productivity vector
      dz = (zmax - zmin)/(J-1);
      s = range(smin, stop=smax, length=S)';   # stochastic volatility vector
      ds = (smax - smin)/(S-1);

    #   rr = r * ones(1, J);
    #   rrr = exp.(rr) ./ (1 .+ exp.(rr));


      b = zeros(I*J*S);
      AT = copy(A');

      #need to fix one value, otherwise matrix is singular
      i_fix = 1;
      b[i_fix] = .1;
      for j = 1:I*J*S
        AT[i_fix, j] = 0;
      end
      AT[i_fix, i_fix] = 1;

      #Solve linear system
      gg = AT \ b[:, 1];
      g_sum = gg' * ones(I * J * S, 1) * dr * dz * ds; 
      gg = gg./g_sum;

      g = reshape(gg, I, J, S);

      return g
end
