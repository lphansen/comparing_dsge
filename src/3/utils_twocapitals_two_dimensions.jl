using LinearAlgebra
using SparseArrays
using Interpolations
using SuiteSparse

mutable struct Baseline{T}
    zeta::T
    kappa::T
    betaz::T
    sigma_z::Array{T, 1}

    beta::T
    sigma_k::Array{T, 1}

    delta::T
end

mutable struct Technology{T}
    alpha::T
    phi::T
    eta::T
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
end

mutable struct FinDiffMethod
  maxit::Integer      # maximum number of iterations in the HJB loop
  crit::Float64       # criterion HJB loop
  Delta::Float64      # delta in HJB algorithm
end

function dstar_twocapitals!(d1::Array{Float64,2},
                            d2::Array{Float64,2},
                            Vr::Array{Float64,2},
                            rr::Array{Float64,2},
                            zz::Array{Float64,2},
                            IJ::Int64,
                            rho::Float64,
                            V::Array{Float64, 2},
                            model::TwoCapitalEconomy{Float64},
                            C::Array{Float64, 2})

    alpha = model.t1.alpha; 
    zeta = model.k1.zeta;
    kappa = model.k1.kappa;
    delta = model.k1.delta;
    phi1 = model.t1.phi;
    phi2 = model.t2.phi;

    for i=1:IJ
        p, vr, z = rr[i], Vr[i], zz[i]
        if kappa == 1.0
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

            C[i] = c_rho_coeff_1*croot/k1a; 

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

function upwind_transform!(var::Array{Float64, 2},
                           var_F::Array{Float64, 2},
                           var_B::Array{Float64, 2},
                           drift_F::Array{Float64, 2},
                           drift_B::Array{Float64, 2},
                           IJ::Int64,
                           dim::Integer=1)

    If = drift_F .>= 0.;
    Ib = .!If .& (drift_B .< 0.);

    if dim ==1
      If[1, :] .= 1.;    # force to use forward at the first row
      If[end, :] .= 0.;
      Ib[end, :] .= 1.;  # force to use backward at the last row
      Ib[1, :] .= 0.;
    elseif dim==2
      If[:, 1] .= 1.;    # force to use forward at the first col
      If[:, end] .= 0.;
      Ib[:, end] .= 1.;  # force to use backward at the last col
      Ib[:, 1] .= 0.;
    end

    for i=1:IJ
        I0 = (If[i] + Ib[i]) == 0;  # when (drift_F < 0 and drift_B > 0)
        var[i] = If[i]*var_F[i] + Ib[i]*var_B[i] + I0*var_F[i];
    end

    nothing
end


function drifts!(mu_1::Array{Float64, 2},
                 mu_r::Array{Float64, 2},
                 mu_z::Array{Float64, 2},
                 d1::Array{Float64, 2},
                 d2::Array{Float64, 2},
                 zz::Array{Float64, 2},
                 rr::Array{Float64, 2},
                 II::Int64,
                 JJ::Int64,
                 model::TwoCapitalEconomy)

    phi1, phi2 = model.t1.phi, model.t2.phi;
    eta1, eta2 = model.t1.eta, model.t2.eta;
    zeta = model.k1.zeta;
    kappa = model.k1.kappa;
    beta1 = model.k1.beta;
    beta2 = model.k2.beta;
    betaz = model.k1.betaz;
    s_k1 = model.k1.sigma_k;
    s_k2 = model.k2.sigma_k;
    s_z = model.k1.sigma_z;
    zbar = 0

    i=0
    for jj = 1:JJ
        for ii = 1:II
            i = i+1;
            p, z = rr[i], zz[i];
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
            
            mu_k1 = log(1 + phi1*d1[i])/phi1 + beta1*z - eta1;
            mu_k2 = log(1 + phi2*d2[i])/phi2 + beta2*z - eta2;
            mu_r[i] = mu_k2 - mu_k1 - (1/2)*(dot(s_k2,s_k2) - dot(s_k1,s_k1))         
            mu_z[i] = -betaz*(z-zbar);
            mu_1[i] = mu_k1*(1-zeta)*(k1a).^(1-kappa)+
                            mu_k2*(zeta)*(k2a).^(1-kappa)+
                            1/2*(dot(s_k1,s_k1)*dkadk1dk1 + dot(s_k2,s_k2)*dkadk2dk2 + 2*dot(s_k1,s_k2)*dkadk1dk2);
            
        end
    end
    nothing

end
    


function drifts_distortion!(h::Array{Float64, 2},
                            s_k1::Float64,
                            s_k2::Float64,
                            s_z::Float64,
                            IJ::Int64,
                            rr::Array{Float64, 2},
                            Vr::Array{Float64, 2},
                            Vz::Array{Float64, 2},
                            model::TwoCapitalEconomy)

    zeta = model.k1.zeta;
    kappa = model.k1.kappa;

    for i=1:IJ
        p = rr[i];
        if kappa == 1.0
            k1a = exp.(p*(-zeta))
            k2a = exp.(p*(1-zeta))
        else
            k1a = (1-zeta + zeta*exp.(p*(1-kappa))).^(1/(kappa-1));
            k2a = ((1-zeta)*exp.(p*(kappa-1)) + zeta).^(1/(kappa-1));
        end
        h[i] = (s_k1*(1-zeta)*(k1a).^(1-kappa) + s_k2*(zeta)*(k2a).^(1-kappa) + (s_k2-s_k1)*Vr[i]) + s_z*Vz[i];
    end

    nothing
end


function create_uu!(uu::Array{Float64, 1},
                    gamma::Float64,
                    rho::Float64,
                    d1::Array{Float64, 2},
                    d2::Array{Float64, 2},
                    h1::Array{Float64, 2},
                    h2::Array{Float64, 2},
                    hz::Array{Float64, 2},
                    mu_1::Array{Float64, 2},
                    rr::Array{Float64, 2},
                    zz::Array{Float64, 2},
                    II::Int64,
                    JJ::Int64,
                    V::Array{Float64, 2},
                    Vr::Array{Float64, 2},
                    Vz::Array{Float64, 2},
                    model::TwoCapitalEconomy)

    alpha = model.t1.alpha;
    delta = model.k1.delta;
    zeta = model.k1.zeta;
    kappa = model.k1.kappa;

    i=0
    for jj = 1:JJ
        for ii = 1:II
            i = i+1;
            p, z = rr[i], zz[i];
            if kappa == 1.0
                k1a = exp.(p*(-zeta))
                k2a = exp.(p*(1-zeta))
            else
                k1a = (1-zeta + zeta*exp.(p*(1-kappa))).^(1/(kappa-1));
                k2a = ((1-zeta)*exp.(p*(kappa-1)) + zeta).^(1/(kappa-1));
            end
            c = alpha - d1[i]*k1a - d2[i]*k2a; 
            misspecification_penalty_term = (1-gamma)*(h1[i]^2 + h2[i]^2 + hz[i]^2)/2;
            if rho == 1.0
                uu[i] = (delta*log(c) + misspecification_penalty_term + mu_1[i]);
            else
                uu[i] = (delta/(1-rho)*(c.^(1-rho)*exp.((rho-1)*V[i])-1) + misspecification_penalty_term + mu_1[i]);
            end
        end
    end
    nothing
end


function create_Aval!(Aval, d_, c_, e_, b_, f_, a_1, a_2, II, JJ)

    iter = 1
    Aval[iter] = d_[1]; iter += 1;
    Aval[iter] = e_[1+1]; iter += 1;
    Aval[iter] = f_[II+1]; iter += 1;
    Aval[iter] = a_1[II+1+1]; iter += 1;

    for j=2:II
        Aval[iter] = c_[-1+j]; iter += 1;
        Aval[iter] = d_[j]; iter += 1;
        Aval[iter] = e_[1+j]; iter += 1;
        Aval[iter] = f_[II+j]; iter += 1;
        Aval[iter] = a_1[II+1+j]; iter += 1;
    end

    Aval[iter] = b_[-II+II+1]; iter += 1;
    Aval[iter] = c_[-1+II+1]; iter += 1;
    Aval[iter] = d_[II+1]; iter += 1;
    Aval[iter] = e_[1+II+1]; iter += 1;
    Aval[iter] = f_[II+II+1]; iter += 1;
    Aval[iter] = a_1[II+1+II+1]; iter += 1;

    for j=(II+2):(II*(JJ-1)-1)
        Aval[iter] = a_2[-(II+1)+j]; iter += 1;
        Aval[iter] = b_[-II+j]; iter += 1;
        Aval[iter] = c_[-1+j]; iter += 1;
        Aval[iter] = d_[j]; iter += 1;
        Aval[iter] = e_[1+j]; iter += 1;
        Aval[iter] = f_[II+j]; iter += 1;
        Aval[iter] = a_1[II+1+j]; iter += 1;
    end

    Aval[iter] = a_2[-(II+1)+II*(JJ-1)]; iter += 1;
    Aval[iter] = b_[-II+II*(JJ-1)]; iter += 1;
    Aval[iter] = c_[-1+II*(JJ-1)]; iter += 1;
    Aval[iter] = d_[II*(JJ-1)]; iter += 1;
    Aval[iter] = e_[1+II*(JJ-1)]; iter += 1;
    Aval[iter] = f_[II+II*(JJ-1)]; iter += 1;

    for j=(II*(JJ-1)+1):(II*JJ-1)
        Aval[iter] = a_2[-(II+1)+j]; iter += 1;
        Aval[iter] = b_[-II+j]; iter += 1;
        Aval[iter] = c_[-1+j]; iter += 1;
        Aval[iter] = d_[j]; iter += 1;
        Aval[iter] = e_[1+j]; iter += 1;
    end

    Aval[iter] = a_2[-(II+1)+(II*JJ)]; iter += 1;
    Aval[iter] = b_[-II+(II*JJ)]; iter += 1;
    Aval[iter] = c_[-1+(II*JJ)]; iter += 1;
    Aval[iter] = d_[(II*JJ)]; iter += 1;

    nothing
end

function value_function_twocapitals(gamma::Float64,
                                    rho::Float64,
                                    model::TwoCapitalEconomy,
                                    grid::Grid_rz,
                                    params::FinDiffMethod,
                                    preloadV0::Array{Float64, 2},
                                    preloadcons::Array{Float64, 2},
                                    beta1::Float64)
                                    
      rmin, rmax, II = grid.rmin, grid.rmax, grid.I;
      zmin, zmax, JJ = grid.zmin, grid.zmax, grid.J;

      IJ = II*JJ;
      II_half = convert(Integer, round(II/2));
      JJ_half = convert(Integer, round(JJ/2));

      maxit  = params.maxit;       # max number of iterations in the HJB loop
      crit = params.crit;          # criterion HJB loop
      Delta = params.Delta;        # Delta in HJB algorithm

      r = range(rmin, stop=rmax, length=II);    # capital ratio vector
      dr = (rmax - rmin)/(II-1);
      z = range(zmin, stop=zmax, length=JJ)';   # productivity vector
      dz = (zmax - zmin)/(JJ-1);
      dr2, dz2, drdz = dr*dr, dz*dz, dr*dz;

      println("rmax = ", rmax, ", rmin = ",rmin, ", rlength = ", II)
      println("zmax = ", zmax, ", zmin = ",zmin, ", zlength = ", JJ)

      rr = r * ones(1, JJ);
      zz = ones(II, 1) * z;

      #========================================================================#
      # Storing matrices
      #========================================================================#
      # Value function and forward/backward finite difference matrices
      V, V0 = zeros(II, JJ), zeros(II, JJ);
      Vr_F, Vr_B = zeros(II, JJ), zeros(II, JJ);
      Vz_F, Vz_B = zeros(II, JJ), zeros(II, JJ);
      Vr, Vz = zeros(II, JJ), zeros(II, JJ);

      # Control variables 
      d1_F, d2_F = zeros(II, JJ), zeros(II, JJ);
      d1_B, d2_B = zeros(II, JJ), zeros(II, JJ);
      h1_F, h1_B  = zeros(II, JJ), zeros(II, JJ);
      h2_F, h2_B  = zeros(II, JJ), zeros(II, JJ);
      hz_F, hz_B  = zeros(II, JJ), zeros(II, JJ);
      d1, d2 = zeros(II, JJ), zeros(II, JJ);
      h1, h2, hz = zeros(II, JJ), zeros(II, JJ), zeros(II, JJ);

      # Drifts
      mu_1_F, mu_1_B, mu_1 = zeros(II, JJ), zeros(II, JJ), zeros(II, JJ);
      mu_r_F, mu_r_B = zeros(II, JJ), zeros(II, JJ);
      mu_z_F, mu_z_B = zeros(II, JJ), zeros(II, JJ);

      uu = zeros(II*JJ);
      A = spdiagm( 0 => ones(II*JJ),
                    1 => ones(II*JJ-1),
                    -1 => ones(II*JJ-1),
                    II => ones(II*(JJ-1)),
                    -II => ones(II*(JJ-1)),
                    -II-1 => ones(II*(JJ-1)-1),
                    II+1 => ones(II*(JJ-1)-1));
      Aval = zeros(nnz(A))

      #========================================================================#
      # MODEL PARAMETERS                                                       #
      #========================================================================#
      delta = model.k1.delta;
      zeta = model.k1.zeta;
      kappa = model.k1.kappa;
      beta1 = model.k1.beta;
      beta2 = model.k2.beta;
      s_k1 = model.k1.sigma_k;
      s_k2 = model.k2.sigma_k;
      s_z =  model.k1.sigma_z;
      alpha = model.t1.alpha;
      phi1, phi2 = model.t1.phi, model.t2.phi;

      t1 = dot(s_k2 - s_k1, s_k2 - s_k1)/(2*dr2);
      t2 = dot(s_k2 - s_k1, s_z)/(2*drdz);
      t3 = dot(s_z, s_z)/(2*dz2);

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
      
      if rho == 1.0
        I_delta = sparse((1/Delta + delta)*I, IJ, IJ);
      else
        I_delta = sparse((1/Delta)*I, IJ, IJ);
      end

      for n=1:maxit

          V = copy(v);
          CF = copy(cF)
          CB = copy(cB)
          V_stacked = vec(V);

          # forward diff (last row never used - known value function there)
          # backward diff (first row never used - known value function there)
          # Diff in the z dimension: 1st/last col = 0 imposed
          Vr_B[2:II, :] = Vr_F[1:II-1, :] = (V[2:II, :] - V[1:II-1, :])./dr;
          Vz_B[:, 2:JJ] = Vz_F[:, 1:JJ-1] = (V[:, 2:JJ] - V[:, 1:JJ-1])./dz;

          # Investment-capital ratios
          dstar_twocapitals!(d1_F, d2_F, Vr_F, rr, zz, IJ, rho, V, model, CF);
          dstar_twocapitals!(d1_B, d2_B, Vr_B, rr, zz, IJ, rho, V, model, CB);
          cF = copy(CF)
          cB = copy(CB)
    
          # Drifts
          drifts!(mu_1_F, mu_r_F, mu_z_F, d1_F, d2_F, zz, rr, II, JJ, model);
          drifts!(mu_1_B, mu_r_B, mu_z_B, d1_B, d2_B, zz, rr, II, JJ, model);

          # Upwind transform
          upwind_transform!(Vr, Vr_F, Vr_B, mu_r_F, mu_r_B, IJ);
          upwind_transform!(Vz, Vz_F, Vz_B, mu_z_F, mu_z_B, IJ, 2);
          upwind_transform!(d1, d1_F, d1_B, mu_r_F, mu_r_B, IJ);
          upwind_transform!(d2, d2_F, d2_B, mu_r_F, mu_r_B, IJ);
	      upwind_transform!(mu_1, mu_1_F, mu_1_B, mu_r_F, mu_r_B, IJ);

          # Drifts distortion
          drifts_distortion!(h1, s_k1[1], s_k2[1], s_z[1], IJ, rr, Vr, Vz, model);
          drifts_distortion!(h2, s_k1[2], s_k2[2], s_z[2], IJ, rr, Vr, Vz, model);
          drifts_distortion!(hz, s_k1[3], s_k2[3], s_z[3], IJ, rr, Vr, Vz, model);

          drifts_distortion!(h1_F, s_k1[1],s_k2[1],s_z[1],IJ,rr,Vr_F,Vz_F, model);
          drifts_distortion!(h2_F, s_k1[2],s_k2[2],s_z[2],IJ,rr,Vr_F,Vz_F, model);
          drifts_distortion!(hz_F, s_k1[3],s_k2[3],s_z[3],IJ,rr,Vr_F,Vz_F, model);
          drifts_distortion!(h1_B, s_k1[1],s_k2[1],s_z[1],IJ,rr,Vr_B,Vz_B, model);
          drifts_distortion!(h2_B, s_k1[2],s_k2[2],s_z[2],IJ,rr,Vr_B,Vz_B, model);
          drifts_distortion!(hz_B, s_k1[3],s_k2[3],s_z[3],IJ,rr,Vr_B,Vz_B, model);

	      # FLOW TERMS
          if beta1 == 0.04
            create_uu!(uu, gamma, rho, d1, d2, h1, h2, hz, mu_1, rr, zz, II, JJ, V, Vr, Vz, model);
          else
            create_uu!(uu, gamma, rho, d1_F, d2_F, h1_F, h2_F, hz_F, mu_1_F, rr, zz, II, JJ, V, Vr_F, Vz_F, model);
          end

          #CONSTRUCT MATRIX A
          a_1 = ones(II, JJ)*t2;
          a_2 = ones(II, JJ)*t2;
          b_ = max.(mu_z_F, 0.)/dz .+ t3 .- t2;
          c_ = max.(mu_r_F, 0.)/dr .+ t1 .- t2;
          d_ = (-max.(mu_r_F, 0.)/dr + min.(mu_r_B, 0.)/dr - max.(mu_z_F, 0.)/dz +
                     min.(mu_z_B, 0.)/dz .- 2*(t1 + t3 - t2));
          e_ = -min.(mu_r_B, 0.)/dr .+ t1 .- t2;
          f_ = -min.(mu_z_B, 0.)/dz .+ t3 .- t2;

          ## r upper boundary
          b_[end, :] += a_2[end, :];
          a_2[end, :] .= 0.0;
          d_[end, :] += c_[end, :];
          c_[end, :] .= 0.0;

          ## r lower boundary
          f_[1 , :] += a_1[1, :];
          a_1[1, :] .= 0.0;
          d_[1, :] += e_[1, :];
          e_[1, :] .= 0.0;

          ## z upper boundary
          d_[:, end] += b_[:, end];
          b_[:, end] .= 0.0;
          c_[:, end] += a_2[:, end];
          a_2[:, end] .= 0.0;

          ## z lower boundary
          d_[:, 1] += f_[:, 1];
          f_[:, 1] .= 0.0;
          e_[:, 1] += a_1[:, 1];
          a_1[:, 1] .= 0.0;

          create_Aval!(Aval, d_, c_, e_, b_, f_, a_1, a_2, II, JJ)                      
          A.nzval .= Aval;

          #Cblas function for y:=a*x + y (in our case: uu^n + v^n -> uu^n)
          BLAS.axpy!(IJ, 1/Delta, V_stacked, 1, uu, 1);

          # TIME CONSUMING PART
          V_stacked = (I_delta - A) \ uu

          Vchange = reshape(V_stacked, II, JJ) - v;
          distance[n] = maximum(abs.(Vchange));

          v = reshape(V_stacked, II, JJ);

          println("----------------------------------");
          println("Iteration = ",n);
          println("Distance = ",distance[n]);
          println("v max = ",maximum(v));
          println("v min = ",minimum(v)); 
          println("----------------------------------")
          
          if distance[n]<crit
              val = v[II_half, JJ_half];
              println("    Value Function Converged, Iteration = ", n,
                      " with gamma=", gamma,
                      " with rho=", rho,
                      " and v=", val);
              create_Aval!(Aval, d_, c_, e_, b_, f_, a_1, a_2, II, JJ)
              A.nzval .= Aval;
              break
                elseif n > 2 && distance[n] > distance[n-2]
                Delta = Delta/2;
                println("Delta = ", Delta)
                if rho == 1.0
                    I_delta = sparse((1/Delta + delta)*I, IJ, IJ);
                  else
                    I_delta = sparse((1/Delta)*I, IJ, IJ);
                end
          end
      end

    val = v[II_half, JJ_half];     # Value at (r_0, z_0)  (objective)

    return A, v,val, d1_F, d2_F, d1_B, d2_B, h1_F, h2_F, hz_F, h1_B, h2_B, hz_B,
           mu_1_F, mu_1_B, mu_r_F, mu_r_B, mu_z_F, mu_z_B, V0, Vr, Vr_F, Vr_B, Vz_B, Vz_F, cF, cB, Vz, rr, zz, dr, dz;
end



function stationary_distribution(A::SparseMatrixCSC{Float64,Int64},
                                 grid::Grid_rz)

      #=======================================================#
      # Construct grid
      #=======================================================#
      rmin, rmax, I = grid.rmin, grid.rmax, grid.I;
      zmin, zmax, J = grid.zmin, grid.zmax, grid.J;
      r = range(rmin, stop=rmax, length=I);    # capital ratio vector
      dr = (rmax - rmin)/(I-1);
      z = range(zmin, stop=zmax, length=J)';   # productivity vector
      dz = (zmax - zmin)/(J-1);
      rr = r * ones(1, J);
      p = exp.(rr) ./ (1 .+ exp.(rr));


      b = zeros(I*J);
      AT = copy(A');

      #need to fix one value, otherwise matrix is singular
      i_fix = 1;
      b[i_fix] = .1;
      for j = 1:I*J
        AT[i_fix, j] = 0;
      end
      AT[i_fix, i_fix] = 1;

      #Solve linear system
      gg = AT \ b[:, 1];
      g_sum = gg' * ones(I * J, 1) * dr * dz;
      gg = gg./g_sum;

      g = reshape(gg, I, J);

      return g
end


