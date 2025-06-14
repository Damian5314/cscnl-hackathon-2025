using DifferentialEquations, Plots, Parameters

# -------------------------------
# 1. Parameters
# -------------------------------
function make_parameters()
    g = 9.81
    L = 5.0
    N = 200
    x = range(0.0, L, length=N) |> collect
    dx = x[2] - x[1]
    zb = zeros(N)
    tspan = (0.0, 2.0)
    return (; g, x, dx, N, zb, tspan)
end

# -------------------------------
# 2. Beginconditie (symmetrisch of asymmetrisch)
# -------------------------------
function initial_conditions(params; symmetric=true)
    @unpack x, zb = params
    xc = symmetric ? 2.5 : 1.0  # bult in het midden of links
    h = 0.1 .* exp.(-100 .* (x .- xc).^2) .+ 10.0
    q = zeros(length(x))
    return vcat(h, q)
end

# -------------------------------
# 3. RHS voor ODE systeem (1D SWE)
# -------------------------------
function shallow_water_rhs!(du, u, p, t)
    @unpack g, x, dx, N, zb = p

    h = @view u[1:N]
    q = @view u[N+1:2N]
    dhdt = @view du[1:N]
    dqdt = @view du[N+1:2N]

    h_ext = vcat(h[end], h, h[1])
    q_ext = vcat(q[end], q, q[1])
    zb_ext = vcat(zb[end], zb, zb[1])
    zeta = h_ext .+ zb_ext

    for i in 1:N
        iL = i
        iR = i + 2

        FhL = q_ext[iL]
        FqL = q_ext[iL]^2 / h_ext[iL] + 0.5 * g * h_ext[iL]^2

        FhR = q_ext[iR]
        FqR = q_ext[iR]^2 / h_ext[iR] + 0.5 * g * h_ext[iR]^2

        dhdt[i] = - (FhR - FhL) / (2dx)
        dzdx = (zeta[iR] - zeta[iL]) / (2dx)
        dqdt[i] = - (FqR - FqL) / (2dx) - g * h[i] * dzdx
    end
end

# -------------------------------
# 4. Oplossen in de tijd
# -------------------------------
function solve_problem(; symmetric=true)
    params = make_parameters()
    u0 = initial_conditions(params; symmetric=symmetric)
    prob = ODEProblem(shallow_water_rhs!, u0, params.tspan, params)
    sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-8)
    return sol, params
end

# -------------------------------
# 5. Maak animatie
# -------------------------------
function make_animation(sol, params, filename)
    N = params.N
    anim = @animate for i in 1:10:length(sol.t)
        h = sol.u[i][1:N]
        plot(params.x, h,
            ylim=(9.9, 10.2),
            xlabel="x", ylabel="Waterhoogte",
            title="t = $(round(sol.t[i], digits=2)) s",
            legend=false)
    end
    gif(anim, filename, fps=10)
end

# -------------------------------
# 6. Run symmetrisch
# -------------------------------
sol_sym, params_sym = solve_problem(symmetric=true)
make_animation(sol_sym, params_sym, "animatie_sym.gif")

# -------------------------------
# 7. Run asymmetrisch
# -------------------------------
sol_asym, params_asym = solve_problem(symmetric=false)
make_animation(sol_asym, params_asym, "animatie_asym.gif")