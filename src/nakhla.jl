const use_return  = haskey(ENV, "USE_RETURN" ) ? parse(Bool, ENV["USE_RETURN"] ) : false
const USE_GPU     = haskey(ENV, "USE_GPU"    ) ? parse(Bool, ENV["USE_GPU"]    ) : false
const do_viz      = haskey(ENV, "DO_VIZ"     ) ? parse(Bool, ENV["DO_VIZ"]     ) : true
const do_save     = haskey(ENV, "DO_SAVE"    ) ? parse(Bool, ENV["DO_SAVE"]    ) : false
#const nx          = haskey(ENV, "NX"         ) ? parse(Int , ENV["NX"]         ) : 128 - 1
#const ny          = haskey(ENV, "NY"         ) ? parse(Int , ENV["NY"]         ) : 128 - 1
###
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, LinearAlgebra

@parallel function smooth!(A2::Data.Array, A::Data.Array, fact::Data.Number)
    @inn(A2) = @inn(A) + 1.0/4.1/fact*(@d2_xi(A) + @d2_yi(A))
    return
end

@parallel function compute_maxloc!(ητ::Data.Array, η::Data.Array)
    @inn(ητ) = @maxloc(η)
    return
end

@parallel function compute_iter_params!(dτ_Rho::Data.Array, Gdτ::Data.Array, ητ::Data.Array, Vpdτ::Data.Number, Re::Data.Number, r::Data.Number, max_lxy::Data.Number)
    @all(dτ_Rho) = Vpdτ*max_lxy/Re/@all(ητ)
    @all(Gdτ)    = Vpdτ^2/@all(dτ_Rho)/(r+2.0)
    return
end

@parallel function compute_η!(η::Data.Array, χ::Data.Array, η0::Data.Number)
    @all(η) = η0 / ((1-@all(χ)) * (1-@all(χ)) * (1-@all(χ)))
    return
end

@parallel function compute_ρ̄!(ρ̄::Data.Array, ρx0::Data.Number, ρm0::Data.Number, χ::Data.Array)
    @all(ρ̄) = (1-@all(χ))*ρm0 + @all(χ)*ρx0
    return
end

@parallel function compute_Δwx!(Δwx::Data.Array, η::Data.Array, ρ̄::Data.Array, ρx0::Data.Number, d0::Data.Number, g0::Data.Number)
    @inn(Δwx) = - (ρx0 - @all(ρ̄)) * g0 * d0*d0 / @all(η)
    return
end

@parallel function compute_massfluxes!(χVx::Data.Array, χVy::Data.Array, χ::Data.Array, Vx::Data.Array, Vy::Data.Array, Δwx::Data.Array)
    @all(χVx)  = @av_xi(χ)* @all(Vx)
    @all(χVy)  = @av_yi(χ)*(@all(Vy)+@all(Δwx))
    return
end

@parallel function compute_dχdt!(dχdt::Data.Array, χVx::Data.Array, χVy::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(dχdt) = - @d_xa(χVx)/dx - @d_ya(χVy)/dy
    return
end

@parallel function compute_χ!(χ::Data.Array, χ_old::Data.Array, dχdt::Data.Array, dχdt_old::Data.Array, dt::Data.Number)
    @inn(χ) = @inn(χ_old) + (@all(dχdt) + @all(dχdt_old))*dt/2
    return
end

@parallel function compute_P!(∇V::Data.Array, Pt::Data.Array, Vx::Data.Array, Vy::Data.Array, Gdτ::Data.Array, r::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    @all(Pt)  = @all(Pt) - r*@all(Gdτ)*@all(∇V)
    return
end

@parallel function compute_τ!(τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, Vx::Data.Array, Vy::Data.Array, η::Data.Array, Gdτ::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(τxx) = (@all(τxx) + 2.0*@all(Gdτ)*@d_xa(Vx)/dx)/(@all(Gdτ)/@all(η) + 1.0)
    @all(τyy) = (@all(τyy) + 2.0*@all(Gdτ)*@d_ya(Vy)/dy)/(@all(Gdτ)/@all(η) + 1.0)
    @all(τxy) = (@all(τxy) + 2.0*@av(Gdτ)*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)))/(@av(Gdτ)/@av(η) + 1.0)
    return
end

@parallel function compute_dV!(Rvx::Data.Array, Rvy::Data.Array, dVx::Data.Array, dVy::Data.Array, Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, ρ̄::Data.Array, ρref::Data.Number, dτ_Rho::Data.Array, dx::Data.Number, dy::Data.Number, g0::Data.Number)
    @all(Rvx)  = @d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx
    @all(Rvy)  = @d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy - (@all(ρ̄)-ρref)*g0
    @all(dVx)  = @av_xi(dτ_Rho)*@all(Rvx)
    @all(dVy)  = @av_yi(dτ_Rho)*@all(Rvy)
    return
end

@parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, dVx::Data.Array, dVy::Data.Array)
    @inn(Vx) = @inn(Vx) + @all(dVx)
    @inn(Vy) = @inn(Vy) + @all(dVy)
    return
end

@parallel_indices (ix) function bc_y0!(A::Data.Array)
    A[ix, 1  ] = 0
    A[ix, end] = 0
    return
end

@parallel_indices (iy) function bc_x!(A::Data.Array)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@parallel_indices (ix) function bc_y!(A::Data.Array)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end

@parallel function assign!(A_old::Data.Array, A::Data.Array)
    @all(A_old) = @all(A)
    return
end 

@views function Stokes2D_()
    # Physics
    lx, ly    = 10.0, 10.0  # domain extent
    d0        = 0.001       # crystal size
    ρm0       = 2500.0      # rel. background density
    ρx0       = 3000.0      # rel. inclusion density
    g0        = 10.0        # gravity
    η0        = 100.0       # matrix viscosity
    χ0        = 0.1         # background cRvystallinity
    χi        = 0.11        # inclusion cRvystallinity
    # Numerics
    stepMax   = 20          # maximum number of time steps
    tend      = 100         # stopping time
    iterMax   = 1e5         # maximum number of pseudo-transient iterations
    nout      = 500         # error checking frequency
    ε         = 1e-8        # nonlinear absolute tolerence
    CFL       = 0.9/sqrt(2) # Courant-Friedrich-Lewy number
    Re        = 5π          # ???
    r         = 1.0         # relative step size
    nx, ny    = 1*128-1, 1*128-1    # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    # Derived numerics
    dx, dy    = lx/nx, ly/ny # cell sizes
    max_lxy   = max(lx,ly)   # max cell size
    dt        = 0.025        # time step
    Vpdτ      = min(dx,dy)*CFL # ???
    xc, yc, xv, yv = LinRange(dx/2, lx - dx/2, nx), LinRange(dy/2, ly - dy/2, ny), LinRange(0, lx, nx+1), LinRange(0, ly, ny+1)
    # Array allocations
    ∇V        = @zeros(nx  ,ny  )
    τxx       = @zeros(nx  ,ny  )
    τyy       = @zeros(nx  ,ny  )
    τxy       = @zeros(nx-1,ny-1)
    Rvx       = @zeros(nx-1,ny-2)
    Rvy       = @zeros(nx-2,ny-1)
    dVx       = @zeros(nx-1,ny-2)
    dVy       = @zeros(nx-2,ny-1)
    χ2        = @zeros(nx  ,ny  )
    ητ        = @zeros(nx  ,ny  )
    Gdτ       = @zeros(nx  ,ny  )
    dτ_Rho    = @zeros(nx  ,ny  )
    χVx       = @zeros(nx-1,ny-2)
    χVy       = @zeros(nx-2,ny-1)
    dχdt      = @zeros(nx-2,ny-2)
    dχdt_old  = copy(dχdt)
    # Initial conditions
    Pt        = @zeros(nx  ,ny  )
    Vx        = @zeros(nx+1,ny  )
    Vy        = @zeros(nx  ,ny+1)
    Δwx       = @zeros(nx  ,ny+1)
    Vy        = @zeros(nx  ,ny+1)
    Rad2      = @zeros(nx  ,ny  )
    Rad2     .= [((ix-1)*dx +0.5*dx -0.5*lx)^2 + ((iy-1)*dy +0.5*dy -0.5*ly)^2 for ix=1:size(Rad2,1), iy=1:size(Rad2,2)]
     χ         = χ0 * ones(nx,ny)
    χ[Rad2.<1.0] .= χi
    χ         = Data.Array( χ )
    χ2       .= χ
    for ism=1:10
        @parallel smooth!(χ2, χ, 1.0)
        χ, χ2 = χ2, χ
    end
    χ_old     = copy(χ)
    ρ̄         =   @zeros(nx,ny)
    η         = η0*@ones(nx,ny)
    ητ       .= η

    # initialise material coefficients
    @parallel compute_η!(η, χ, η0)
    @parallel compute_ρ̄!(ρ̄, ρx0, ρm0, χ)
    @parallel compute_Δwx!(Δwx, η, ρ̄, ρx0, d0, g0)
    @parallel (1:size(Δwx,2)) bc_x!(Δwx)
    @parallel (1:size(Δwx,1)) bc_y0!(Δwx)

    # initialise iterative parameters
    @parallel compute_maxloc!(ητ, η)
    @parallel (1:size(ητ,2)) bc_x!(ητ)
    @parallel (1:size(ητ,1)) bc_y!(ητ)
    @parallel compute_iter_params!(dτ_Rho, Gdτ, ητ, Vpdτ, Re, r, max_lxy)
    
    ρref = mean(ρ̄)
    show(ρref)

    # visualise initial condition
    if do_viz
        p1 = heatmap(xc, yv, Array(Δwx)', aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yv), c=:viridis, title="Δwx")
        p2 = heatmap(xc, yv, Array(Vy)' , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yv), c=:viridis, title="Vy")
        p3 = heatmap(xc, yc, Array(χ)'  , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="χ")
        p4 = heatmap(xc, yc, Array(χ-χ_old)'  , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="χ-χ_old")
        #p4 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
        display(plot(p1, p2, p3, p4))
    end

    # time loop
    step = 1; time = 0
    while time < tend && step <= stepMax
        @printf("\n*****  step = %d, time = %1.3e, dt = %1.3e \n", step, time, dt)

        @parallel assign!(χ_old,χ)
        @parallel assign!(dχdt_old,dχdt)

        # iterative solution loop
        err=2*ε; iter=1; err_evo1=[]; err_evo2=[]
        while err > ε && iter <= iterMax
            if (iter==1)  global wtime0 = Base.time()  end

            if iter % nout == 0
                # update TC-solution
                @parallel compute_massfluxes!(χVx, χVy, χ, Vx, Vy, Δwx)
                @parallel compute_dχdt!(dχdt, χVx, χVy, dx, dy)
                @parallel compute_χ!(χ, χ_old, dχdt, dχdt_old, dt)
                @parallel (1:size(χ,1)) bc_y!(χ)
                @parallel (1:size(χ,2)) bc_x!(χ)

                # update material coefficients
                @parallel compute_η!(η, χ, η0)
                @parallel compute_ρ̄!(ρ̄, ρx0, ρm0, χ)
                @parallel compute_Δwx!(Δwx, η, ρ̄, ρx0, d0, g0)
                @parallel (1:size(Δwx,2)) bc_x!(Δwx)
                @parallel (1:size(Δwx,1)) bc_y0!(Δwx)

                # update iterative parameters
                @parallel compute_maxloc!(ητ, η)
                @parallel (1:size(ητ,2)) bc_x!(ητ)
                @parallel (1:size(ητ,1)) bc_y!(ητ)
                @parallel compute_iter_params!(dτ_Rho, Gdτ, ητ, Vpdτ, Re, r, max_lxy)
            end

            # update FM-solution
            @parallel compute_P!(∇V, Pt, Vx, Vy, Gdτ, r, dx, dy)

            @parallel compute_τ!(τxx, τyy, τxy, Vx, Vy, η, Gdτ, dx, dy)
            @parallel compute_dV!(Rvx, Rvy, dVx, dVy, Pt, τxx, τyy, τxy, ρ̄, ρref, dτ_Rho, dx, dy, g0)
            @parallel compute_V!(Vx, Vy, dVx, dVy)
            @parallel (1:size(Vx,1)) bc_y!(Vx)
            @parallel (1:size(Vy,2)) bc_x!(Vy)

            if iter % nout == 0
                # check residuals
                @parallel compute_dV!(Rvx, Rvy, dVx, dVy, Pt, τxx, τyy, τxy, ρ̄, ρref, dτ_Rho, dx, dy, g0)
                Vmin, Vmax  = minimum(Vx), maximum(Vx)
                Pmin, Pmax  = minimum(Pt), maximum(Pt)
                norm_Rvx    = norm(Rvx)/sqrt(length(Rvx))/((Vmax-Vmin)/lx)
                norm_Rvy    = norm(Rvy)/sqrt(length(Rvy))/((Vmax-Vmin)/lx)
                norm_∇V     = norm(∇V )/sqrt(length(∇V ))/((Pmax-Pmin)/lx)
                err         = maximum([norm_Rvx, norm_Rvy, norm_∇V])
                push!(err_evo1, maximum([norm_Rvx, norm_Rvy, norm_∇V])); push!(err_evo2,iter)
                @printf("  --- iter = %d, err = %1.3e [norm_Rvx=%1.3e, norm_Rvy=%1.3e, norm_∇V=%1.3e] \n", iter, err, norm_Rvx, norm_Rvy, norm_∇V)
            end
            
            iter += 1
        end
        # Performance
        wtime    = Base.time() - wtime0
        A_eff    = (3*2)/1e9*nx*ny*sizeof(Data.Number)  # Effective main memoRvy access per iteration [GB] (Lower bound of required memoRvy access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
        wtime_it = wtime/(iter-10)                      # Execution time per iteration [s]
        T_eff    = A_eff/wtime_it                       # Effective memoRvy throughput [GB/s]
        @printf("      Total steps = %d, err = %1.3e, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", iter-1, err, wtime, round(T_eff, sigdigits=2))
        time += dt
        step += 1
    end
    
    # Visualisation
    if do_viz
        p1 = heatmap(xc, yv, Array(Δwx)', aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yv), c=:viridis, title="Δwx")
        p2 = heatmap(xc, yv, Array(Vy)' , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yv), c=:viridis, title="Vy")
        p3 = heatmap(xc, yc, Array(χ)'  , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="χ")
        p4 = heatmap(xc, yc, Array(χ-χ_old)'  , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="χ-χ_old")
        #p4 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
        display(plot(p1, p2, p3, p4))
    end
    if do_save
        !ispath("../output") && mkdir("../output")
        open("../output/out_Stokes2D.txt","a") do io
            println(io, "$(nx) $(ny) $(iter)")
        end
    end
    return xc, yc, Pt
end

if use_return
    xc, yc, P = Stokes2D_();
else
    Stokes2D = begin Stokes2D_(); return; end
end
