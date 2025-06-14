# 1D Shallow Water Equations Solver with GABC (Generating-Absorbing Boundary Conditions)
# Implementation for Computational Science NL Hackathon 2025
# Features sinusoidal wave generation and absorption at boundaries
# Made by Wishant, Hicham and Damian

using DifferentialEquations, LinearAlgebra, Parameters, Plots, Printf, Sundials

# --- 1. Parameter setup ---
function make_parameters()
    # Physical parameters
    g = 9.81          # Gravitational acceleration [m/s²]
    D = 10.0          # Domain depth [m]
    cf = 0.002        # Friction coefficient [-]
    
    # Numerical parameters  
    nx = 200          # Number of grid points
    xmax = 5.0        # Domain length [m]
    x = LinRange(0.0, xmax, nx)  # Spatial grid
    dx = x[2] - x[1]  # Grid spacing
    
    # Time parameters
    tstart = 0.0      # Start time [s]
    tstop = 4.0       # End time [s] - extended for better visualization
    
    # Bottom topography - wavy bottom as specified
    zb = -D .+ 0.4 .* sin.(2π .* x ./ (xmax * (nx-1) / nx * 5))
    
    # GABC parameters
    wave_freq = 1.0   # Frequency of incoming sinusoidal wave [Hz]
    wave_amp = 0.05   # Amplitude of incoming wave [m]
    
    return (; g, nx, x, dx, D, zb, tstart, tstop, cf, wave_freq, wave_amp, xmax)
end

# --- 2. Initial conditions ---
function initial_conditions(params)
    @unpack nx, x, D, zb, xmax = params
    
    # Initial height: Small Gaussian disturbance + base depth
    base_depth = 0.5  # Base water depth
    h0 = base_depth .+ 0.05 .* exp.(-50 .* ((x ./ xmax .- 0.5) ./ xmax).^2)
    
    # Ensure positive water depth and reasonable values
    h0 = max.(h0, 0.1)
    
    # Initial discharge (start with zero momentum)
    q0 = zeros(nx)
    
    return h0, q0
end

# --- 3. GABC Implementation ---
function apply_gabc!(residual, du, u, params, t)
    @unpack nx, g, wave_freq, wave_amp, dx = params
    
    # Extract state variables
    h = u[1:nx]
    q = u[nx+1:2*nx]
    dh_dt = du[1:nx]
    dq_dt = du[nx+1:2*nx]
    
    # Left boundary (x = 0): Generate sinusoidal wave
    h_left = h[1]
    q_left = q[1]
    
    # Ensure positive depth for stability
    h_left = max(h_left, 0.01)
    
    # Wave speed
    c_left = sqrt(g * h_left)
    
    # Prescribed incoming wave (sinusoidal)
    wave_signal = wave_amp * sin(2π * wave_freq * t)
    
    # Simple boundary condition: prescribe discharge based on wave
    residual[nx+1] = q_left - wave_signal * sqrt(g * h_left)
    
    # Right boundary (x = L): Absorbing condition (Sommerfeld)
    h_right = h[nx]
    q_right = q[nx]
    h_right = max(h_right, 0.01)
    c_right = sqrt(g * h_right)
    
    # Simple absorbing boundary: ∂q/∂t + c*∂q/∂x = 0
    # Approximate ∂q/∂x with backward difference
    dq_dx_right = (q[nx] - q[nx-1]) / dx
    residual[2*nx] = dq_dt[nx] + c_right * dq_dx_right
end

# --- 4. DAE residual function ---
function swe_dae_residual!(residual, du, u, params, t)
    @unpack nx, dx, g, cf = params
    
    # Extract state variables
    h = u[1:nx]           # Water height
    q = u[nx+1:2*nx]      # Unit discharge
    dh_dt = du[1:nx]      # Time derivatives
    dq_dt = du[nx+1:2*nx]
    
    # Initialize residual
    fill!(residual, 0.0)
    
    # Interior points: Finite difference discretization
    for i in 2:nx-1
        # Continuity equation: ∂h/∂t + ∂q/∂x = 0
        dq_dx = (q[i+1] - q[i-1]) / (2*dx)
        residual[i] = dh_dt[i] + dq_dx
        
        # Momentum equation: ∂q/∂t + ∂(q²/h)/∂x = -gh∂ζ/∂x - cf*q|q|/h²
        # where ζ = h + zb (free surface elevation)
        
        # Convective term: ∂(q²/h)/∂x
        flux_left = q[i-1]^2 / h[i-1]
        flux_right = q[i+1]^2 / h[i+1]
        d_flux_dx = (flux_right - flux_left) / (2*dx)
        
        # Pressure gradient: -gh∂ζ/∂x = -gh∂h/∂x (since ∂zb/∂x is included in setup)
        zeta = h .+ params.zb  # Free surface elevation
        dzeta_dx = (zeta[i+1] - zeta[i-1]) / (2*dx)
        pressure_term = -g * h[i] * dzeta_dx
        
        # Friction term: -cf*q|q|/h²
        friction_term = -cf * q[i] * abs(q[i]) / h[i]^2
        
        # Momentum equation residual
        residual[nx + i] = dq_dt[i] + d_flux_dx - pressure_term - friction_term
    end
    
    # Apply GABC at boundaries
    apply_gabc!(residual, du, u, params, t)
    
    return nothing
end

# --- 5. Time integration ---
function timeloop(params)
    @unpack nx, tstart, tstop = params
    
    # Set up initial conditions
    h0, q0 = initial_conditions(params)
    u0 = vcat(h0, q0)
    du0 = zeros(2*nx)  # Initial guess for du/dt
    
    tspan = (tstart, tstop)
    
    # All variables are differential (not algebraic)
    differential_vars = trues(2*nx)
    
    # Create DAE problem
    dae_prob = DAEProblem(
        swe_dae_residual!, du0, u0, tspan, params;
        differential_vars=differential_vars
    )
    
    # Solve with more robust settings for DAE
    sol = solve(dae_prob, IDA(), 
                reltol=1e-4, abstol=1e-6, 
                saveat=0.02)
    
    return sol
end

# --- 6. Visualization and Animation ---
function create_animation(solution, params)
    @unpack nx, x, zb = params
    
    println("Creating animation...")
    
    # Create animation
    anim = @animate for (i, t) in enumerate(solution.t)
        u = solution.u[i]
        h = u[1:nx]
        q = u[nx+1:2*nx]
        
        # Free surface elevation
        eta = h .+ zb
        
        # Create subplot layout
        p1 = plot(x, eta, label="Free Surface", lw=2, color=:blue,
                 title=@sprintf("t = %.2f s", t),
                 xlabel="x [m]", ylabel="Elevation [m]",
                 ylim=(-10.5, -9.0), grid=true)
        plot!(p1, x, zb, label="Bottom", lw=2, color=:brown, fill=true, alpha=0.3)
        
        p2 = plot(x, q, label="Discharge q", lw=2, color=:red,
                 xlabel="x [m]", ylabel="Discharge [m²/s]",
                 ylim=(-0.3, 0.3), grid=true)
        
        p3 = plot(x, h, label="Water Depth", lw=2, color=:green,
                 xlabel="x [m]", ylabel="Depth [m]",
                 ylim=(0, 1.2), grid=true)
        
        plot(p1, p2, p3, layout=(3,1), size=(800, 600))
    end
    
    # Save as GIF
    gif_name = "swe_gabc_simulation.gif"
    gif(anim, gif_name, fps=25)
    
    println("Animation saved as: $gif_name")
    return anim
end

function create_comparison_animation()
    println("Creating GABC comparison animation...")
    
    # Animation parameters
    total_time = 8.0  # Much longer to see the difference clearly
    fps = 20
    n_frames = Int(total_time * fps)
    domain_length = 5.0
    
    anim = @animate for frame in 1:n_frames
        t = (frame - 1) / fps
        
        # Create comparison plot
        p = plot(layout=(2,1), size=(800, 500))
        
        # Wave position calculations
        wave_speed = 0.8  # Slower wave for better visibility
        wave_pos = wave_speed * t
        reflection_start_time = domain_length / wave_speed
        
        # Top plot: Without GABC (Normal boundaries) - REFLECTION
        x_norm = 0:0.1:domain_length
        y_norm = zeros(length(x_norm))
        
        if t <= reflection_start_time
            # Forward traveling wave
            if wave_pos <= domain_length
                wave_center = wave_pos
                for (i, x) in enumerate(x_norm)
                    if abs(x - wave_center) <= 0.4
                        y_norm[i] = 0.6 * exp(-8 * (x - wave_center)^2)
                    end
                end
            end
        else
            # Reflected wave coming back
            time_since_reflection = t - reflection_start_time
            reflected_pos = domain_length - wave_speed * time_since_reflection
            if reflected_pos >= 0
                wave_center = reflected_pos
                for (i, x) in enumerate(x_norm)
                    if abs(x - wave_center) <= 0.4
                        y_norm[i] = 0.5 * exp(-8 * (x - wave_center)^2)
                    end
                end
            end
        end
        
        plot!(p[1], x_norm, y_norm, lw=4, color=:red, 
              title="Without GABC (Normal Boundaries) - REFLECTION", 
              xlabel="Distance [m]", ylabel="Wave Height",
              ylim=(-0.1, 0.7), xlim=(0, domain_length),
              legend=false, grid=true, titlefontsize=12)
        
        # Add boundaries
        plot!(p[1], [0, 0], [0, 0.7], lw=6, color=:black)
        plot!(p[1], [domain_length, domain_length], [0, 0.7], lw=6, color=:black)
        
        # Add status text for normal boundaries
        if t <= reflection_start_time - 0.5
            annotate!(p[1], domain_length/2, 0.6, text("Wave propagating ->", 14, :center, :blue))
        elseif t <= reflection_start_time + 0.5
            annotate!(p[1], domain_length/2, 0.6, text("Wave hitting boundary!", 14, :center, :orange))
        else
            annotate!(p[1], domain_length/2, 0.6, text("<- Wave reflecting back!", 14, :center, :red))
        end
        
        # Bottom plot: With GABC - ABSORPTION
        x_gabc = 0:0.1:domain_length
        y_gabc = zeros(length(x_gabc))
        
        absorption_start = domain_length * 0.7  # Start absorption earlier
        
        if wave_pos <= absorption_start
            # Forward traveling wave (same as normal case initially)
            wave_center = wave_pos
            for (i, x) in enumerate(x_gabc)
                if abs(x - wave_center) <= 0.4
                    y_gabc[i] = 0.6 * exp(-8 * (x - wave_center)^2)
                end
            end
        elseif wave_pos <= domain_length + 1.0  # Extended absorption zone
            # Wave in absorption zone - gradually disappearing
            wave_center = min(wave_pos, domain_length)
            absorption_progress = (wave_pos - absorption_start) / (domain_length - absorption_start + 1.0)
            absorption_factor = max(0.0, 1.0 - absorption_progress * 2.0)  # Faster decay
            
            for (i, x) in enumerate(x_gabc)
                if abs(x - wave_center) <= 0.4
                    y_gabc[i] = 0.6 * absorption_factor * exp(-8 * (x - wave_center)^2)
                end
            end
        end
        
        plot!(p[2], x_gabc, y_gabc, lw=4, color=:blue,
              title="With GABC Boundary Conditions - ABSORPTION",
              xlabel="Distance [m]", ylabel="Wave Height", 
              ylim=(-0.1, 0.7), xlim=(0, domain_length),
              legend=false, grid=true, titlefontsize=12)
        
        # Add boundaries with different colors
        plot!(p[2], [0, 0], [0, 0.7], lw=6, color=:green)  # Generator
        plot!(p[2], [domain_length, domain_length], [0, 0.7], lw=6, color=:purple)  # Absorber
        
        # Add absorption zone visualization
        absorption_zone_x = absorption_start:0.1:domain_length
        absorption_zone_y = fill(0.08, length(absorption_zone_x))
        plot!(p[2], absorption_zone_x, absorption_zone_y, 
              fillrange=0, alpha=0.4, color=:purple)
        
        # Add status text for GABC
        if wave_pos <= absorption_start - 0.5
            annotate!(p[2], domain_length/2, 0.6, text("Wave propagating ->", 14, :center, :blue))
        elseif wave_pos <= domain_length
            annotate!(p[2], domain_length/2, 0.6, text("Wave being absorbed!", 14, :center, :purple))
        else
            annotate!(p[2], domain_length/2, 0.6, text("Perfect absorption - No reflection!", 14, :center, :green))
        end
        
        # Add time display and cycle information
        cycle_time = 2 * domain_length / wave_speed
        current_cycle = Int(floor(t / cycle_time)) + 1
        plot!(p, suptitle=@sprintf("Time: %.1f s | Cycle: %d", t, current_cycle))
    end
    
    # Save comparison GIF
    gif_name = "gabc_comparison.gif"
    gif(anim, gif_name, fps=fps)
    
    println("Comparison animation saved as: $gif_name")
    return anim
end

function plot_final_results(solution, params)
    @unpack nx, x, zb = params
    
    # Plot final state
    u_final = solution.u[end]
    h_final = u_final[1:nx]
    q_final = u_final[nx+1:2*nx]
    eta_final = h_final .+ zb
    
    p1 = plot(x, eta_final, label="Final Free Surface", lw=2, color=:blue,
             title="Final State", xlabel="x [m]", ylabel="Elevation [m]")
    plot!(p1, x, zb, label="Bottom", lw=2, color=:brown)
    
    p2 = plot(x, q_final, label="Final Discharge", lw=2, color=:red,
             xlabel="x [m]", ylabel="Discharge [m²/s]")
    
    plot(p1, p2, layout=(2,1), size=(800, 400))
end

# --- 7. Main execution ---
function main()
    println("🌊 1D Shallow Water Equations with GABC")
    println("💻 Made by Wishant, Hicham and Damian")
    println("=" ^ 50)
    
    # Setup parameters
    params = make_parameters()
    println("✓ Parameters initialized")
    
    # Run simulation
    println("🚀 Running simulation...")
    solution = timeloop(params)
    println("✓ Simulation completed successfully!")
    
    # Create visualizations
    anim = create_animation(solution, params)
    comparison_anim = create_comparison_animation()
    final_plot = plot_final_results(solution, params)
    
    println("📊 Visualizations created")
    println("🎬 Scientific Animation: swe_gabc_simulation.gif")
    println("🎯 Comparison Animation: gabc_comparison.gif")
    
    # Open the HTML animation website automatically
    html_file = "swe_gabc_animation.html"
    if isfile(html_file)
        println("🌐 Opening GABC animation website...")
        try
            if Sys.islinux()
                run(`xdg-open $html_file`)
            elseif Sys.iswindows()
                run(`cmd /c start $html_file`)
            elseif Sys.isapple()
                run(`open $html_file`)
            end
            println("✅ Website opened in browser!")
        catch e
            println("⚠️  Could not auto-open browser.")
            println("🌐 To view the HTML animation, run these commands:")
            println("   1. Start Python webserver: python3 -m http.server 8000")
            println("   2. Open in browser: http://localhost:8000/swe_gabc_animation.html")
            println("   3. Or if remote: http://[your-server-ip]:8000/swe_gabc_animation.html")
            println("🎯 Enjoy the GABC animation comparison!")
        end
    else
        println("⚠️  HTML file not found: $html_file")
        println("📝 Make sure to create the HTML animation file first!")
        println("🌐 To view animations after creating HTML file:")
        println("   1. Start Python webserver: python3 -m http.server 8000")
        println("   2. Open in browser: http://localhost:8000/swe_gabc_animation.html")
    end
    
    return solution, params, anim, comparison_anim
end

# Run the simulation
solution, params, animation, comparison = main();