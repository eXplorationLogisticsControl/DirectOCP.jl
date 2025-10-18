"""Test with CR3BP, continuous problem"""

# using GLMakie     # import manually if runnig script as example
using Ipopt
using JuMP
using LinearAlgebra
using OrdinaryDiffEq
using Test

include(joinpath(@__DIR__, "../src/DirectOCP.jl"))

if !@isdefined get_plot
    get_plot = true
end

mutable struct ControlParams
    μ::Float64
    u::Vector
    function ControlParams(μ::Float64)
        new(μ, zeros(4))
    end
end

function test_continuous_problem(;get_plot::Bool = false)
    # parameters for ODE
    μ = 1.215058560962404e-02
    DU = 389703     # km
    TU = 382981     # sec
    MU = 500.0      # kg
    VU = DU/TU      # km/s

    params = ControlParams(μ)

    function eom!(drv, rv, p, t)
        x, y, z = rv[1:3]
        vx, vy, vz = rv[4:6]
        r1 = sqrt( (x+p.μ)^2 + y^2 + z^2 );
        r2 = sqrt( (x-1+p.μ)^2 + y^2 + z^2 );
        drv[1:3] = rv[4:6]
        # derivatives of velocities
        drv[4] =  2*vy + x - ((1-p.μ)/r1^3)*(p.μ+x) + (p.μ/r2^3)*(1-p.μ-x) + p.u[1]
        drv[5] = -2*vx + y - ((1-p.μ)/r1^3)*y       - (p.μ/r2^3)*y         + p.u[2]
        drv[6] =           - ((1-p.μ)/r1^3)*z       - (p.μ/r2^3)*z         + p.u[3]
        return
    end

    # initial solution
    rv0 = [1.0809931218390707E+00,
        0.0000000000000000E+00,
        -2.0235953267405354E-01,
        1.0157158264396639E-14,
        -1.9895001215078018E-01,
        7.2218178975912707E-15]
    period_0 = 2.3538670417546639E+00

    rvf = [1.1648780946517576,
        0.0,
        -1.1145303634437023E-1,
        0.0,
        -2.0191923237095796E-1,
        0.0]
    period_f = 3.3031221822879884

    # initial & final LPO
    sol_lpo0 = solve(
        ODEProblem(eom!, rv0, [0.0, period_0], params),
        Tsit5(); reltol = 1e-12, abstol = 1e-12
    )
    sol_lpof = solve(
        ODEProblem(eom!, rvf, [0.0, period_f], params),
        Tsit5(); reltol = 1e-12, abstol = 1e-12
    )

    nx, nu = 6, 4
    N = 50
    tf = 2.6 
    times = LinRange(0.0, tf, N)

    # thrust = 0.15    # N
    # umax = thrust/MU/1e3 / (VU/TU)
    umax = 0.25

    # create reference solution
    x_along_lpo0 = sol_lpo0(LinRange(0.0, period_0, N))
    x_along_lpof = sol_lpof(LinRange(0.0, period_f, N))
    xbar = zeros(nx,N)
    alphas = LinRange(0,1,N)
    for (i,alpha) in enumerate(alphas)
        xbar[:,i] = (1-alpha)*x_along_lpo0[:,i] + alpha*x_along_lpof[:,i]
    end
    ubar = zeros(nu, N-1)

    # function for control parametrization within dynamics contraints in multiple shooting
    function gfun(t::Tuple{Float64,Float64}, xu::T...) where {T<:Real}
        uhat_k = [xu[i] for i in nx+1:nx+nu-1]
        umag_k = xu[nx+nu]
        return [zeros(T, 3, 3); I(3)] * uhat_k * umag_k
    end

    bc_implicit_initial = nothing
    bc_implicit_final = nothing

    # construct problem
    prob = DirectOCP.ContinuousProblem(
        eom!,
        Ipopt.Optimizer,
        times,
        xbar,
        ubar,
        gfun,
        bc_implicit_initial,
        bc_implicit_final,
        params,
    )

    # boundary conditions (analytical)
    @constraint(prob.model, prob.model[:x][:,1] == rv0)
    @constraint(prob.model, prob.model[:x][:,end] == rvf)

    # max control magnitude
    @constraint(prob.model, max_control_magnitude_constraint[k in 1:N-1], prob.model[:u][4,k] <= umax)

    set_optimizer_attribute(prob.model, "tol", 1e-6)
    set_optimizer_attribute(prob.model, "constr_viol_tol", 1e-10)
    set_optimizer_attribute(prob.model, "max_iter", 100)
    if get_plot
        set_optimizer_attribute(prob.model, "print_level", 5)
    else
        set_optimizer_attribute(prob.model, "print_level", 0)   # at test, we set Ipopt to be silent
    end

    # solve
    optimize!(prob.model)
    xs_opt, us_opt = value.(prob.model[:x]), value.(prob.model[:u])
    residuals_dynamics = DirectOCP.get_dynamics_residuals(prob, times, xs_opt, us_opt)

    @test termination_status(prob.model) == LOCALLY_SOLVED
    @test maximum(abs.(residuals_dynamics)) <= 1e-8
    @test objective_value(prob.model) ≈ 4.068582679757887 atol = 1e-5

    if get_plot
        # construct trajectory initial guess
        sols_ig = DirectOCP.get_trajectory(prob, times, xbar, ubar)

        # plot
        fig = Figure(size=(600,500))
        Label(fig[0,1:2], text = "Objective = $(objective_value(prob.model))", fontsize=18)
        ax3d = Axis3(fig[1,1]; aspect=:data)
        lines!(Array(sol_lpo0)[1,:], Array(sol_lpo0)[2,:], Array(sol_lpo0)[3,:], color=:grey)
        lines!(Array(sol_lpof)[1,:], Array(sol_lpof)[2,:], Array(sol_lpof)[3,:], color=:grey)
        for _sol in sols_ig
            lines!(Array(_sol)[1,:], Array(_sol)[2,:], Array(_sol)[3,:], color=:skyblue)
        end
        
        xs_opt, us_opt = value.(prob.model[:x]), value.(prob.model[:u])
        sols_opt = DirectOCP.get_trajectory(prob, times, xs_opt, us_opt)
        for _sol in sols_opt
            lines!(Array(_sol)[1,:], Array(_sol)[2,:], Array(_sol)[3,:], color=:midnightblue)
        end

        # plot control impulses
        ax_u = Axis(fig[1,2])
        stairs!(ax_u, times, [us_opt[4,:]; 0.0], label="||u||", linewidth=2.0, color = :black, step=:post)
        for i in 1:3
            stairs!(ax_u, times, [us_opt[i,:] .* us_opt[4,:]; 0.0], label="u[$i]", linewidth=1.0, step=:post)
        end
        hlines!(ax_u, [-umax, umax], color=:grey, linestyle=:dash)
        axislegend(ax_u, position=:cc)
        display(fig)
    end
end

test_continuous_problem(;get_plot=get_plot)