using LinearAlgebra
using SparseArrays
using Interpolations
using SuiteSparse

mutable struct Baseline{T}
    a11::T
    a22::T
    beta::T
    eta::T
    sigma_k::Array{T, 1}
    sigma_z::Array{T, 1}
    sigma_y::Array{T, 1}
    delta::T
end

mutable struct Technology{T}
    alpha::T
    phi::T
end

mutable struct OneCapitalEconomy{T}
    k::Baseline{T}
    t::Technology{T}
end

mutable struct Grid_zy{T}
  zmin::T
  zmax::T
  I::Integer           # number of z points
  ymin::T
  ymax::T
  J::Integer           # number of y points
end

mutable struct FinDiffMethod
  maxit::Integer      # maximum number of iterations in the HJB loop
  crit::Float64       # criterion HJB loop
  Delta::Float64      # Delta in HJB algorithm
end

function dstar_onecapital!(d::Array{Float64,2},
                            zz::Array{Float64,2},
                            IJ::Int64,
                            rho::Float64,
                            V::Array{Float64, 2},
                            model::OneCapitalEconomy{Float64},
                            C::Array{Float64, 2})

    # Check for the positivity of the critical term in the quad root formula
    # This might not hold if the technology parameters are weird
    alpha = model.t.alpha; 
    delta = model.k.delta;
    phi = model.t.phi;

    for i=1:IJ
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



function drifts!(mu_k::Array{Float64, 2},
                 d::Array{Float64, 2},
                 zz::Array{Float64, 2},
                 yy::Array{Float64, 2},
                 IJ::Int64,
                 model::OneCapitalEconomy)

    phi = model.t.phi;
    beta = model.k.beta;
    eta = model.k.eta;
    s_k = model.k.sigma_k;

    for i=1:IJ
        z, y = zz[i], yy[i];
        mu_k[i] =  log(1+d[i]*phi)/phi + beta*z - eta - y*dot(s_k,s_k)/2;
    end

    nothing

end


function drifts_distortion!(h::Array{Float64, 2},
                            s_k::Float64,
                            s_z::Float64,
                            s_y::Float64,
                            IJ::Int64,
                            Vz::Array{Float64, 2},
                            Vy::Array{Float64, 2},
                            model::OneCapitalEconomy)

    for i=1:IJ
        h[i] = s_k + s_z*Vz[i] + s_y*Vy[i];
    end

    nothing
end


function create_uu!(uu::Array{Float64, 1},
                    yy::Array{Float64, 2},
                    gamma::Float64,
                    rho::Float64,
                    d::Array{Float64, 2},
                    hk::Array{Float64, 2},
                    hz::Array{Float64, 2},
                    hy::Array{Float64, 2},
                    mu_k::Array{Float64, 2},
                    IJ::Int64,
                    V::Array{Float64, 2},
                    model::OneCapitalEconomy)

    alpha = model.t.alpha;
    delta = model.k.delta;

    for i=1:IJ
        y = yy[i]
        c = alpha - d[i];
        penalty_term = (1-gamma)*y*(hk[i]^2 + hz[i]^2 + hy[i]^2)/2;
        if rho == 1.0
            uu[i] = (delta*log(c) + penalty_term + mu_k[i]);
        else
            uu[i] = (delta/(1-rho)*(c.^(1-rho)*exp.((rho-1)*V[i])-1)+ penalty_term + mu_k[i]);
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

function value_function_onecapital(gamma::Float64,
                                    rho::Float64,
                                    model::OneCapitalEconomy,
                                    grid::Grid_zy,
                                    params::FinDiffMethod,
                                    preloadV0::Array{Float64, 2},
                                    preloadcons::Array{Float64, 2},
                                    ymean::Float64)

      zmin, zmax, II = grid.zmin, grid.zmax, grid.I;
      ymin, ymax, JJ = grid.ymin, grid.ymax, grid.J;

      IJ = II*JJ;
      II_half = convert(Integer, round(II/2));
      JJ_half = convert(Integer, round(JJ/2));

      maxit  = params.maxit;       # max number of iterations in the HJB loop
      crit = params.crit;          # criterion HJB loop
      Delta = params.Delta;        # Delta in HJB algorithm

      z = range(zmin, stop=zmax, length=II); 
      dz = (zmax - zmin)/(II-1);
      y = range(ymin, stop=ymax, length=JJ)'; 
      dy = (ymax - ymin)/(JJ-1);
      dz2, dy2, dzdy = dz*dz, dy*dy, dz*dy;

      println("zmax = ", zmax, ", zmin = ",zmin, ", zlength = ", II)
      println("ymax = ", ymax, ", ymin = ",ymin, ", ylength = ", JJ)

      zz = z * ones(1, JJ);
      yy = ones(II, 1) * y;

      #========================================================================#
      # Storing matrices
      #========================================================================#
      # Value function and forward/backward finite difference matrices
      V, V0 = zeros(II, JJ), zeros(II, JJ);
      Vz_F, Vz_B = zeros(II, JJ), zeros(II, JJ);
      Vy_F, Vy_B = zeros(II, JJ), zeros(II, JJ);
      Vz, Vy = zeros(II, JJ), zeros(II, JJ);

      # Control variables 
      hk_F, hk_B  = zeros(II, JJ), zeros(II, JJ);
      hz_F, hz_B  = zeros(II, JJ), zeros(II, JJ);
      hy_F, hy_B  = zeros(II, JJ), zeros(II, JJ);
      d = zeros(II, JJ);
      hk, hz, hy = zeros(II, JJ), zeros(II, JJ), zeros(II, JJ);

      # Drifts
      mu_k = zeros(II, JJ);
      mu_z = zeros(II, JJ);
      mu_y = zeros(II, JJ);
      
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
      delta = model.k.delta;
      a11 = model.k.a11;
      a22 = model.k.a22;
      eta = model.k.eta;
      beta = model.k.beta;
      s_k = model.k.sigma_k;
      s_z = model.k.sigma_z;
      s_y = model.k.sigma_y;
      alpha = model.t.alpha;
      phi = model.t.phi;

      t1 = dot(s_z, s_z)/(2*dz2);
      t2 = dot(s_z, s_y)/(2*dzdy);
      t3 = dot(s_y, s_y)/(2*dy2);

      #========================================================================#
      # INITIALIZATION                                                         #
      #========================================================================#

      V0 = preloadV0;
      C0 = preloadcons;
      v = copy(V0);
      c = copy(C0);
      distance = zeros(maxit);

      #========================================================================#
      # HAMILTON-JACOBI-BELLMAN EQUATION
      #========================================================================#

      if rho == 1.0
        I_delta = sparse((1/Delta + delta)*I, IJ, IJ);
      else
        I_delta = sparse((1/Delta)*I, IJ, IJ);
      end

      mu_z = - a11 * zz;
      mu_y = - a22 * (yy.-ymean);

      for n=1:maxit

          V = copy(v);
          C = copy(c)
          V_stacked = vec(V);

          # forward diff (last row never used - known value function there)
          # backward diff (first row never used - known value function there)
          # Diff in the z dimension: 1st/last col = 0 imposed
          Vz_B[2:II, :] = Vz_F[1:II-1, :] = (V[2:II, :] - V[1:II-1, :])./dz;
          Vy_B[:, 2:JJ] = Vy_F[:, 1:JJ-1] = (V[:, 2:JJ] - V[:, 1:JJ-1])./dy;

          # Investment-capital/Consumption-capital ratios
          dstar_onecapital!(d, zz, IJ, rho, V, model, C);
          c = copy(C)

          # Drifts
          drifts!(mu_k, d, zz, yy, IJ, model);
        
          # upwind transform
          upwind_transform!(Vz, Vz_F, Vz_B, mu_z, mu_z, IJ);
          upwind_transform!(Vy, Vy_F, Vy_B, mu_y, mu_y, IJ, 2);

          # distorted drift
          drifts_distortion!(hk, s_k[1], s_z[1], s_y[1], IJ, Vz, Vy, model);
          drifts_distortion!(hz, s_k[2], s_z[2], s_y[2], IJ, Vz, Vy, model);
          drifts_distortion!(hy, s_k[3], s_z[3], s_y[3], IJ, Vz, Vy, model);
          drifts_distortion!(hk_F, s_k[1],s_z[1],s_y[1],IJ,Vz_F,Vy_F, model);
          drifts_distortion!(hz_F, s_k[2],s_z[2],s_y[2],IJ,Vz_F,Vy_F, model);
          drifts_distortion!(hy_F, s_k[3],s_z[3],s_y[3],IJ,Vz_F,Vy_F, model);
          drifts_distortion!(hk_B, s_k[1],s_z[1],s_y[1],IJ,Vz_B,Vy_B, model);
          drifts_distortion!(hz_B, s_k[2],s_z[2],s_y[2],IJ,Vz_B,Vy_B, model);
          drifts_distortion!(hy_B, s_k[3],s_z[3],s_y[3],IJ,Vz_B,Vy_B, model);

	      # flow term
          create_uu!(uu, yy, gamma, rho, d, hk, hz, hy, mu_k, IJ, V, model);

          # CONSTRUCT MATRIX A
          a_1 = yy*t2;
          a_2 = yy*t2;
          b_ = max.(mu_y, 0.)/dy .+ (t3 .- t2).*yy;
          c_ = max.(mu_z, 0.)/dz .+ (t1 .- t2).*yy;
          d_ = (-max.(mu_z, 0.)/dz + min.(mu_z, 0.)/dz - max.(mu_y, 0.)/dy +
                 min.(mu_y, 0.)/dy .- 2*(t1 + t3 - t2).*yy);
          e_ = -min.(mu_z, 0.)/dz .+ (t1 .- t2).*yy;
          f_ = -min.(mu_y, 0.)/dy .+ (t3 .- t2).*yy;

          # Adding reflection boundary in I dimension
          f_[1 , :] += a_1[1, :];
          a_1[1, :] .= 0.0;
          b_[end, :] += a_2[end, :];
          a_2[end, :] .= 0.0;

          d_[1, :] += e_[1, :];
          e_[1, :] .= 0.0;
          d_[end, :] += c_[end, :];
          c_[end, :] .= 0.0;

          # Adding reflection boundary in J dimension
          d_[:, 1] += f_[:, 1];
          e_[:, 1] += a_1[:, 1];
          d_[:, end] += b_[:, end];
          c_[:, end] += a_2[:, end];

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
          end
      end

    val = v[II_half, JJ_half];    

    return A, v, val, d, hk_F, hz_F, hy_F, hz_B, hk_B, hy_B,
           mu_k, mu_z, mu_y, V0, Vz, Vz_F, Vz_B, Vy, Vy_B, Vy_F, c, zz, yy, dz, dy;
end



function stationary_distribution(A::SparseMatrixCSC{Float64,Int64},
                                 grid::Grid_zy)

      #=======================================================#
      # Construct grid
      #=======================================================#
      rmin, rmax, I = grid.zmin, grid.zmax, grid.I;
      zmin, zmax, J = grid.ymin, grid.ymax, grid.J;
      r = range(rmin, stop=rmax, length=I);    
      dr = (rmax - rmin)/(I-1);
      z = range(zmin, stop=zmax, length=J)'; 
      dz = (zmax - zmin)/(J-1);
      rr = r * ones(1, J);
      
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
