using LinearAlgebra
using SparseArrays
using Interpolations
using SuiteSparse
using JuMP, Ipopt

mutable struct Baseline{T}
    a11::T
    sigma_z::Array{T, 1}

    beta::T
    eta::T
    sigma_k::Array{T, 1}

    delta::T
end

mutable struct Technology{T}
    alpha::T
    phi::T
end

mutable struct Robustness{T}
    q::T
    rho1::T
    rho2::T
end

mutable struct OneCapitalEconomy{T}
    k::Baseline{T}
    t::Technology{T}
    robust::Robustness{T}
end

mutable struct Grid_z{T}
  zmin::T
  zmax::T
  I::Integer           # number of z points
end

mutable struct FinDiffMethod
  maxit::Integer      # maximum number of iterations in the HJB loop
  crit::Float64       # criterion HJB loop
  Delta::Float64      # Delta in HJB algorithm
end

function dstar_onecapital!(d::Array{Float64,1},
                            zz::Array{Float64,1},
                            II::Int64,
                            rho::Float64,
                            V::Array{Float64, 1},
                            model::OneCapitalEconomy{Float64},
                            C::Array{Float64, 1})

                            alpha = model.t.alpha; 
                            delta = model.k.delta;
                            phi1 = model.t.phi;

                            for i=1:II
                                z = zz[i]
                            if rho == 1.0
                                d[i] = (alpha - delta)/(delta*phi + 1)
                                C[i] = alpha - d[i]
                            else
                                clowerlim = 0.0001
                                function g(cx)
                                    if cx < clowerlim
                                        cx = clowerlim
                                    end
                                    dx = alpha - cx
                                    return 1/(1+phi*dx) - delta*exp.(V[i]*(rho-1))*cx.^(-rho)
                                end
                                
                                x0 = C[i];
                                croot = find_zero(g, x0, Roots.Order1(), maxiters = 100000000000, xatol = 10e-15, xrtol = 10e-15, atol = 10e-18, rtol = 10e-18, strict = true);
                                if croot < clowerlim
                                    croot = clowerlim
                                end
                                d_root = alpha - croot
                                C[i] = croot;
                                d[i] = d_root;
                            end
                        end
        nothing
end


function robust_control_variables!(s1::Array{Float64,1},
                                    s2::Array{Float64,1},
                                    lambda::Array{Float64,1},
                                    zz::Array{Float64,1},
                                    V::Array{Float64, 1},
                                    Vz::Array{Float64, 1},
                                    II::Int64,
                                    twoparameter::Int64,
                                    model::OneCapitalEconomy{Float64})
    alpha = model.t.alpha; 
    delta = model.k.delta;
    phi = model.t.phi;
    beta = model.k.beta;
    s_k = model.k.sigma_k;
    s_z = model.k.sigma_z;
    betaz = model.k.a11;
    zbar = 0
    q = model.robust.q;
    rho1 = model.robust.rho1;
    rho2 = model.robust.rho2;
    for i = 1:II
        z, lam, vz = zz[i], lambda[i], Vz[i]

        A1 = s_k[1] + s_z[1]*vz
        A2 = s_k[2] + s_z[2]*vz
        C1 = (rho1 + rho2*(z-zbar))*s_z[1] 
        C2 = (rho1 + rho2*(z-zbar))*s_z[2]
        AA = (A1.^2+A2.^2)/2
        BB = (A1.*C1+A2.*C2) - (rho1 + rho2*(z-zbar)) * (A1*s_z[1] + A2*s_z[2])
        CC = -betaz.*(z-zbar).*(rho1 + rho2*(z-zbar)) + rho2.*dot(s_z,s_z)/2-q.^2/2 - (rho1+rho2*(z-zbar)).*(C1*s_z[1]+C2*s_z[2]) + (C1.^2+C2.^2)/2
        lam_inv_1 = (-BB - sqrt(BB.^2-4*AA*CC))./(2*AA)
        lam_inv_2 = (-BB + sqrt(BB.^2-4*AA*CC))./(2*AA)
        if  (z == zbar) && (twoparameter==true)
            s1[i] = 0
            s2[i] = 0
        else
            lambda_1 = 1/lam_inv_1
            s1_1 = -lam_inv_1*A1 - C1
            s2_1 = -lam_inv_1*A2 - C2
            lambda_2 = 1/lam_inv_2
            s1_2 = -lam_inv_2*A1 - C1
            s2_2 = -lam_inv_2*A2 - C2
            if (A1*s1_1 + A2*s2_1 < A1*s1_2 + A2*s2_2) && (lambda_1 > 0)
                lambda[i] = lambda_1
                s1[i] = s1_1
                s2[i] = s2_1
            elseif (A1*s1_1 + A2*s2_1 >= A1*s1_2 + A2*s2_2) && (lambda_2 > 0)
                lambda[i] = lambda_2
                s1[i] = s1_2
                s2[i] = s2_2
            else
                error("All lambda no larger than 0.")
            end
        end
    end
    nothing
end

function upwind_transform!(var::Array{Float64, 1},
                           var_F::Array{Float64, 1},
                           var_B::Array{Float64, 1},
                           drift_F::Array{Float64, 1},
                           drift_B::Array{Float64, 1},
                           II::Int64)

    If = drift_F .>= 0.;
    Ib = .!If .& (drift_B .< 0.);

    If[1] = 1.;    # force to use forward at the first row
    If[end] = 0.;
    Ib[end] = 1.;  # force to use backward at the last row
    Ib[1] = 0.;

    for i=1:II
        I0 = (If[i] + Ib[i]) == 0;  # when (drift_F < 0 and drift_B > 0)
        var[i] = If[i]*var_F[i] + Ib[i]*var_B[i] + I0*var_F[i];
    end

    nothing
end


function drifts!(mu_1::Array{Float64, 1},
                 mu_z::Array{Float64, 1},
                 d::Array{Float64, 1},
                 s1::Array{Float64, 1}, 
                 s2::Array{Float64, 1},
                 zz::Array{Float64, 1},
                 II::Int64,
                 model::OneCapitalEconomy)

    phi = model.t.phi;

    beta = model.k.beta;
    eta = model.k.eta;
    s_k = model.k.sigma_k;
    s_z = model.k.sigma_z;
    betaz = model.k.a11;
    zbar = 0
    for i=1:II
        z, s1p, s2p = zz[i], s1[i], s2[i];
        mu_k =  log(1+d[i]*phi)/phi + beta*z - eta - dot(s_k,s_k)/2;
        mu_z[i] = -betaz*(z-zbar);
        mu_1[i] = mu_k + (s_k[1]*s1p+s_k[2]*s2p);
    end

    nothing

end

function structure_distortion!(mu_z_distorted::Array{Float64, 1},
                               s1::Array{Float64, 1}, 
                               s2::Array{Float64, 1},
                               zz::Array{Float64, 1},
                               II::Int64,
                               model::OneCapitalEconomy)
    s_k = model.k.sigma_k;
    s_z = model.k.sigma_z;

    for i=1:II
        z, s1p, s2p = zz[i], s1[i], s2[i];
        mu_z_distorted[i] = s_z[1]*s1p + s_z[2]*s2p;
    end
    nothing
end

function drifts_distortion!(h::Array{Float64, 1},
                            s_k::Float64,
                            s_z::Float64,
                            II::Int64,
                            Vz::Array{Float64, 1},
                            model::OneCapitalEconomy)

    for i=1:II
        h[i] = s_k+ s_z*Vz[i];
    end

    nothing
end


function create_uu!(uu::Array{Float64, 1},
                    gamma::Float64,
                    rho::Float64,
                    d::Array{Float64, 1},
                    hk::Array{Float64, 1},
                    hz::Array{Float64, 1},
                    mu_1::Array{Float64, 1},
                    mu_z_distorted::Array{Float64, 1},
                    zz::Array{Float64, 1},
                    II::Int64,
                    V::Array{Float64, 1},
                    Vz::Array{Float64, 1},
                    model::OneCapitalEconomy)

    alpha = model.t.alpha;
    delta = model.k.delta;

    for i=1:II
        z = zz[i]
        c = alpha - d[i]
        structure_distorted = Vz[i] * mu_z_distorted[i];
        penalty_term = (1-gamma)*(hk[i]^2 + hz[i]^2)/2;
        if rho == 1.0
            uu[i] = (delta*log(c) + penalty_term + mu_1[i] + structure_distorted);
        else
            uu[i] = (delta/(1-rho)*(c.^(1-rho)*exp.((rho-1)*V[i])-1) + penalty_term + mu_1[i] + structure_distorted);
        end
    end

    nothing
end


function create_Aval!(Aval, d_, c_, e_, II)

    iter = 1
    Aval[iter] = d_[1]; iter += 1;
    Aval[iter] = e_[1+1]; iter += 1;

    for j=2:(II-1)
        Aval[iter] = c_[-1+j]; iter += 1;
        Aval[iter] = d_[j]; iter += 1;
        Aval[iter] = e_[1+j]; iter += 1;
    end

    Aval[iter] = c_[II-1]; iter += 1;
    Aval[iter] = d_[II]; iter += 1;

    nothing
end

function create_Aval_natural!(Aval, d_, c_, e_, c_2, e_2, II)

    iter = 1
    j = 1
    Aval[iter] = d_[j]; iter += 1;
    Aval[iter] = e_[1+j]; iter += 1;
    Aval[iter] = e_2[2+j]; iter += 1;

    j = 2
    Aval[iter] = c_[-1+j]; iter += 1;
    Aval[iter] = d_[j]; iter += 1;
    Aval[iter] = e_[1+j]; iter += 1;
    Aval[iter] = e_2[2+j]; iter += 1;

    for j=3:(II-2)
        Aval[iter] = c_2[-2+j]; iter += 1;
        Aval[iter] = c_[-1+j]; iter += 1;
        Aval[iter] = d_[j]; iter += 1;
        Aval[iter] = e_[1+j]; iter += 1;
        Aval[iter] = e_2[2+j]; iter += 1;
    end

    j = II-1
    Aval[iter] = c_2[-2+j]; iter += 1;
    Aval[iter] = c_[-1+j]; iter += 1;
    Aval[iter] = d_[j]; iter += 1;
    Aval[iter] = e_[1+j]; iter += 1;

    j = II
    Aval[iter] = c_2[-2+j]; iter += 1;
    Aval[iter] = c_[-1+j]; iter += 1;
    Aval[iter] = d_[j]; iter += 1;

    nothing
end

function value_function_onecapital(gamma::Float64,
                                    rho::Float64,
                                    model::OneCapitalEconomy,
                                    grid::Grid_z,
                                    params::FinDiffMethod,
                                    preloadV0::Array{Float64, 1},
                                    preloadcons::Array{Float64, 1},
                                    twoparameter::Int64)
                                    
      zmin, zmax, II = grid.zmin, grid.zmax, grid.I;

      # Derived indexes
      II_half = convert(Integer, round(II/2));
      maxit  = params.maxit;       # max number of iterations in the HJB loop
      crit = params.crit;          # criterion HJB loop
      Delta = params.Delta;        # Delta in HJB algorithm

      zz = collect(range(zmin, stop=zmax, length=II));   
      dz = (zmax - zmin)/(II-1);
      dz2 = dz*dz;

      println("zmax = ", zmax, ", zmin = ",zmin, ", zlength = ", II)

      # Value function and forward/backward finite difference matrices
      V, V0 = zeros(II), zeros(II);

      # These matrices need to compute choices
      Vz_F, Vz_B = zeros(II), zeros(II);
      Vz = zeros(II);

      # Control variables 
      d_F, d_B = zeros(II), zeros(II);
      s1_F, s2_F = zeros(II), zeros(II);
      s1_B, s2_B = zeros(II), zeros(II);
      hk_F, hk_B  = zeros(II), zeros(II);
      hz_F, hz_B  = zeros(II), zeros(II);
      d = zeros(II);
      s1, s2 = zeros(II), zeros(II);
      hk, hz = zeros(II), zeros(II);

      # Lagrange multiplier
      lambda_F, lambda_B, lambda = zeros(II), zeros(II), zeros(II);

      # Drifts
      mu_1_F, mu_1_B, mu_1 = zeros(II), zeros(II), zeros(II);
      mu_z_F, mu_z_B = zeros(II), zeros(II);
      mu_z_distorted_F, mu_z_distorted_B, mu_z_distorted = zeros(II), zeros(II), zeros(II);
      
      uu = zeros(II);
      A = spdiagm( 0 => ones(II),
                    1 => ones(II-1),
                    -1 => ones(II-1));
      Aval = zeros(nnz(A))

      #========================================================================#
      # MODEL PARAMETERS                                                       #
      #========================================================================#
      delta = model.k.delta;
      a11 = model.k.a11;
      eta = model.k.eta;
      beta = model.k.beta;
      s_k = model.k.sigma_k;
      s_z =  model.k.sigma_z;
      alpha = model.t.alpha;
      phi = model.t.phi;

      t2 = dot(s_z, s_z)/(2*dz2);

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
        I_delta = sparse((1/Delta + delta)*I, II, II);
      else
        I_delta = sparse((1/Delta)*I, II, II);
      end

      for n=1:maxit

          V = copy(v);
          CF = copy(cF)
          CB = copy(cB)
          V_stacked = vec(V);

          # forward diff (last row never used - known value function there)
          # backward diff (first row never used - known value function there)
          # Diff in the z dimension: 1st/last col = 0 imposed
          Vz_B[2:II] = Vz_F[1:II-1] = (V[2:II] - V[1:II-1])./dz;

          # Investment-capital ratios
          dstar_onecapital!(d_F, zz, II, rho, V, model, CF);
          dstar_onecapital!(d_B, zz, II, rho, V, model, CB);
          cF = copy(CF)
          cB = copy(CB)
    
          # Drifts
          drifts!(mu_1_F, mu_z_F, d_F, s1_F, s2_F, zz, II, model);
          drifts!(mu_1_B, mu_z_B, d_B, s1_B, s2_B, zz, II, model);

          # Upwind transform
          upwind_transform!(Vz, Vz_F, Vz_B, mu_z_F, mu_z_B, II);
          upwind_transform!(d, d_F, d_B, mu_z_F, mu_z_B, II);
	      upwind_transform!(mu_1, mu_1_F, mu_1_B, mu_z_F, mu_z_B, II);
          
          # Robust control variables
          robust_control_variables!(s1, s2, lambda, zz, V, Vz, II, twoparameter, model);
          robust_control_variables!(s1_F, s2_F, lambda_F, zz, V, Vz_F, II,twoparameter, model);
          robust_control_variables!(s1_B, s2_B, lambda_B, zz, V, Vz_B, II,twoparameter, model);
          structure_distortion!(mu_z_distorted, s1, s2, zz, II, model);
          structure_distortion!(mu_z_distorted_F, s1_F, s2_F, zz, II, model);
          structure_distortion!(mu_z_distorted_B, s1_B, s2_B, zz, II, model);

          # Drifts distortion
          drifts_distortion!(hk, s_k[1], s_z[1], II, Vz, model);
          drifts_distortion!(hz, s_k[2], s_z[2], II, Vz, model);

          drifts_distortion!(hk_F, s_k[1],s_z[1],II,Vz_F, model);
          drifts_distortion!(hz_F, s_k[2],s_z[2],II,Vz_F, model);
          drifts_distortion!(hk_B, s_k[1],s_z[1],II,Vz_B, model);
          drifts_distortion!(hz_B, s_k[2],s_z[2],II,Vz_B, model);

          create_uu!(uu, gamma, rho, d, hk, hz, mu_1, mu_z_distorted, zz, II, V, Vz, model);

          #CONSTRUCT MATRIX A
          c_ = max.(mu_z_F, 0.)/dz .+ t2;
          d_ = -max.(mu_z_F, 0.)/dz + min.(mu_z_B, 0.)/dz .- 2*t2;
          e_ = -min.(mu_z_B, 0.)/dz .+ t2;

          c_2 = zeros(II);
          e_2 = zeros(II);
          
          d_[1] += e_[1];
          e_[1] = 0.0;
          d_[end] += c_[end];
          c_[end] = 0.0;

          create_Aval!(Aval, d_, c_, e_, II)
          A.nzval .= Aval;

          #Cblas function for y:=a*x + y (in our case: uu^n + v^n -> uu^n)
          BLAS.axpy!(II, 1/Delta, V_stacked, 1, uu, 1);

          # TIME CONSUMING PART
          V_stacked = (I_delta - A) \ uu
          #ldiv!(V_stacked, factorize(I_delta - A), uu);
          #lsmr!(V_stacked, I_delta - A, uu, atol=1e-8, btol=1e-8)

          Vchange = reshape(V_stacked, II) - v;
          distance[n] = maximum(abs.(Vchange));

          v = reshape(V_stacked, II);

          println("----------------------------------");
          println("Iteration = ",n);
          println("Distance = ",distance[n]);
          println("v max = ",maximum(v));
          println("v min = ",minimum(v));
          println("----------------------------------")
          
          if distance[n]<crit
            val = v[II_half];
            println("    Value Function Converged, Iteration = ", n,
                    " with gamma=", gamma,
                    " with rho=", rho,
                    " and v=", val);
            create_Aval!(Aval, d_, c_, e_, II)
            A.nzval .= Aval;
            break
          elseif n > 2 && distance[n] > distance[n-2]
              Delta = Delta/2;
              println("Delta = ", Delta)
              if rho == 1.0
                  I_delta = sparse((1/Delta + delta)*I, II, II);
                else
                  I_delta = sparse((1/Delta)*I, II, II);
              end
          
            end
      end

    val = v[II_half];    

    return A, v,val, d_F, d_F, hk_F, hz_F, hk_B, hz_B, s1_F, s2_F, lambda_F, s1_B, s2_B, lambda_B, mu_z_distorted_F, mu_z_distorted_B,
           mu_1_F, mu_1_B, mu_z_F, mu_z_B, V0, Vz_B, Vz_F, cF, cB, Vz, zz, dz, uu;
end



function stationary_distribution(A::SparseMatrixCSC{Float64,Int64},
                                 grid::Grid_z)

      #=======================================================#
      # Construct grid
      #=======================================================#
      zmin, zmax, I = grid.zmin, grid.zmax, grid.I;
      z = range(zmin, stop=zmax, length=I)';   
      dz = (zmax - zmin)/(I-1);

      b = zeros(I);
      AT = copy(A');

      #need to fix one value, otherwise matrix is singular
      i_fix = 1;
      b[i_fix] = .1;
      for j = 1:I
        AT[i_fix, j] = 0;
      end
      AT[i_fix, i_fix] = 1;

      #Solve linear system
      gg = AT \ b[:, 1];
      g_sum = gg' * ones(I, 1) * dz;
      gg = gg./g_sum;

      g = reshape(gg, I);

      return g
end


