"""Impulsive optimal control problem"""


mutable struct ImpulsiveProblem <: OptimalControlProblem
    N::Int
    nx::Int
    nu::Int
    eom!::Function
    model::Model
    times::Union{Vector,LinRange}
    base_ode::ODEProblem
    gfun::Function
    ode_params
    ode_method
    ode_reltol::Float64
    ode_abstol::Float64
end


function get_shooting_func(prob::ImpulsiveProblem, tspan, control_parametrization::Symbol)

    if control_parametrization == :DirMag
        function shooting_constraint(xu::T...) where {T<:Real}
            _ode_modified = remake(
                prob.base_ode,
                u0 = [xu[i] for i in 1:prob.nx] + prob.gfun(tspan, xu...),
                tspan=tspan,
            )
            sol = solve(_ode_modified, prob.ode_method; reltol=prob.ode_reltol, abstol=prob.ode_abstol)
            return sol.u[end]
        end

    else
        @error "Unsupported `control_parametrization`: $control_parametrization"
    end
    return shooting_constraint
end


function ImpulsiveProblem(
    eom!::Function,
    optimizer,
    times::Union{Vector,LinRange},
    xbar::Matrix,
    ubar::Matrix,
    gfun::Function,
    bc_implicit_initial::Union{Function, Nothing},
    bc_implicit_final::Union{Function, Nothing},
    ode_params = nothing;
    ode_method = Tsit5(),
    ode_reltol::Float64 = 1e-12,
    ode_abstol::Float64 = 1e-12,
    control_parametrization = :DirMag,
)
    # extract sizes
    N = length(times)
    nx, _Nx = size(xbar)
    nu, _Nu = size(ubar)
    @assert _Nx == _Nu == N

    # base ODE problem
    base_ode = ODEProblem(
        eom!, ones(nx), (0.0, 1.0), ode_params
    )

    # instantiate problem struct
    prob = ImpulsiveProblem(
        N,
        nx,
        nu,
        eom!, 
        Model(optimizer),
        times,
        base_ode,
        gfun,
        ode_params,
        ode_method,
        ode_reltol,
        ode_abstol,
    )

    # poopulate JuMP with variables
    @variable(prob.model, x[i=1:nx, k=1:N], start = xbar[i,k])
    @variable(prob.model, u[i=1:nu, k=1:N], start = ubar[i,k])

    # memoize dynamics function
    for k in 1:N-1
        shooting_constraint = get_shooting_func(prob, (times[k], times[k+1]), control_parametrization)
        memoized_shooting_fun = memoize(shooting_constraint, nx)   # second argument is number of outputs

        # add nonlinear operator for dynamics
        for i in 1:nx
            op = add_nonlinear_operator(
                prob.model, nx+nu, memoized_shooting_fun[i];
                name = Symbol("dynamics_k$(k)_i$(i)"),
            )
            @constraint(prob.model, x[i,k+1] - op([x[:,k]; u[:,k]]...)== 0)
        end
    end

    # constraints on control variables for numerical stability
    if control_parametrization == :DirMag
        @constraint(prob.model, constraint_dir_unit_vector[k=1:N], u[1,k]^2 + u[2,k]^2 + u[3,k]^2 == 1.0)
        @constraint(prob.model, constraint_dir_bounds[i=1:nu-1, k=1:N], -1.0 <= u[i,k] <= 1.0)
        @constraint(prob.model, constraint_mag_lower_bound[k=1:N], u[4,k] >= 0.0)
    end

    # append boundary conditions
    if !isnothing(bc_implicit_initial)
        # TODO
    end
    if !isnothing(bc_implicit_final)
        # TODO
    end

    # objective
    @objective(prob.model, Min, sum(u[4,:]))
    return prob
end


"""Extract trajectory from impulsive problem"""
function get_trajectory(prob::ImpulsiveProblem, times, xs, us)
    sols = []
    for (k,t) in enumerate(times[1:end-1])
        _ode_modified = remake(
            prob.base_ode,
            u0=xs[:,k] + prob.gfun((t,times[k+1]), [xs[:,k]; us[:,k]]...),
            tspan=(t, times[k+1]),
        )
        sol = solve(_ode_modified, prob.ode_method; reltol=prob.ode_reltol, abstol=prob.ode_abstol)
        push!(sols, sol)
    end
    return sols
end


function get_dynamics_residuals(prob::ImpulsiveProblem, times, xs, us)
    g_dynamics = zeros(prob.nx, prob.N-1)
    for (k,t) in enumerate(times[1:end-1])
        _ode_modified = remake(
            prob.base_ode,
            u0=xs[:,k] + prob.gfun((t,times[k+1]), [xs[:,k]; us[:,k]]...),
            tspan=(t, times[k+1]),
        )
        sol = solve(_ode_modified, prob.ode_method; reltol=prob.ode_reltol, abstol=prob.ode_abstol)
        g_dynamics[:,k] = xs[:,k+1] - sol.u[end]
    end
    return g_dynamics
end