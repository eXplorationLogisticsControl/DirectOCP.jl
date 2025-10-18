"""Impulsive optimization problem"""


mutable struct ImpulsiveProblem <: DirectNonlinearOptimalControlProblem
    N::Int
    nx::Int
    nu::Int
    eom_stm!::Function
    model::Model
    times::Union{Vector,LinRange}
    base_ode::ODEProblem
    dfdu::Function
    ode_params
    ode_method
    ode_reltol
    ode_abstol
end


function get_shooting_func(prob::ImpulsiveProblem, tspan, control_parametrization::Symbol)

    if control_parametrization == :DirMag
        function shooting_constraint(xu::T...) where {T<:Real}
            x_k = [xu[i] for i in 1:prob.nx]
            uhat_k = [xu[i] for i in prob.nx+1:prob.nx+prob.nu-1]
            umag_k = xu[prob.nx+prob.nu]
            _ode_modified = remake(
                prob.base_ode,
                u0 = x_k + prob.dfdu(x_k, tspan[1]) * uhat_k * umag_k,
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
    eom_stm!::Function,
    optimizer,
    times::Union{Vector,LinRange},
    xbar::Matrix,
    ubar::Matrix,
    dfdu::Function = (x,t) -> [zeros(3,3); I(3)],
    ode_params = nothing;
    ode_method = Tsit5(),
    ode_reltol = 1e-12,
    ode_abstol = 1e-12,
    control_parametrization = :DirMag,
)
    # extract sizes
    nx, _Nx = size(xbar)
    nu, _Nu = size(ubar)
    @assert _Nx == _Nu == length(times)

    # base ODE problem
    base_ode = ODEProblem(
        eom_stm!, ones(nx), (0.0, 1.0), ode_params
    )

    # instantiate problem struct
    N = length(times)
    prob = ImpulsiveProblem(
        N,
        nx,
        nu,
        eom_stm!, 
        Model(optimizer),
        times,
        base_ode,
        dfdu,
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
        memoized_shooting_func = memoize(shooting_constraint, nx)   # second argument is number of outputs

        # add nonlinear operator for dynamics
        for i in 1:nx
            op = add_nonlinear_operator(
                prob.model, nx+nu, memoized_shooting_func[i];
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

    # objective
    @objective(prob.model, Min, sum(u[4,:]))
    return prob
end


"""Append boundary conditions to the problem"""
function append_boundary_conditions!(prob::ImpulsiveProblem, x0, xf)
    @assert length(x0) == length(xf) == prob.nx
    @constraint(prob.model, prob.model[:x][:,1] == x0)
    @constraint(prob.model, prob.model[:x][:,end] + prob.dfdu(prob.model[:x][:,end], prob.times[end]) * prob.model[:u][1:prob.nu-1,end] * prob.model[:u][prob.nu,end] == xf)
end


"""Extract trajectory from impulsive problem"""
function get_trajectory(prob::ImpulsiveProblem, times, xs, us)
    sols = []
    for (k,t) in enumerate(times[1:end-1])
        _ode_modified = remake(
            prob.base_ode,
            u0=xs[:,k] + prob.dfdu(xs[:,k], t) * us[1:prob.nu-1,k] * us[prob.nu,k],
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
            u0=xs[:,k] + prob.dfdu(xs[:,k], t) * us[1:prob.nu-1,k] * us[prob.nu,k],
            tspan=(t, times[k+1]),
        )
        sol = solve(_ode_modified, prob.ode_method; reltol=prob.ode_reltol, abstol=prob.ode_abstol)
        g_dynamics[:,k] = xs[:,k+1] - sol.u[end]
    end
    return g_dynamics
end