const use_return  = haskey(ENV, "USE_RETURN" ) ? parse(Bool, ENV["USE_RETURN"] ) : false
const USE_GPU     = haskey(ENV, "USE_GPU"    ) ? parse(Bool, ENV["USE_GPU"]    ) : false
const do_viz      = haskey(ENV, "DO_VIZ"     ) ? parse(Bool, ENV["DO_VIZ"]     ) : true
const do_save     = haskey(ENV, "DO_SAVE"    ) ? parse(Bool, ENV["DO_SAVE"]    ) : false
#const nx          = haskey(ENV, "NX"         ) ? parse(Int , ENV["NX"]         ) : 128 - 1
#const ny          = haskey(ENV, "NY"         ) ? parse(Int , ENV["NY"]         ) : 128 - 1
###
using Random
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, LinearAlgebra

@parallel function smooth!(A::Data.Array, A_old::Data.Array)
    @inn(A) = @inn(A_old) + 1.0/4.1*(@d2_xi(A) + @d2_yi(A))
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

@parallel function compute_ρ!(ρ::Data.Array, ρx0::Data.Number, ρm0::Data.Number, χ::Data.Array)
    @all(ρ) = (1-@all(χ))*ρm0 + @all(χ)*ρx0
    return
end

@parallel function compute_Δwx!(Δwx::Data.Array, η::Data.Array, ρ::Data.Array, ρx0::Data.Number, d0::Data.Number, g0::Data.Number)
    @inn(Δwx) = - (ρx0 - @all(ρ)) * g0 * d0*d0 / @all(η)
    return
end

# @parallel function compute_eql!(χ::Data.Array, T::Data.Array, Tsol::Data.Number, Tliq::Data.Number)
#     @all(χ) = 1-(@all(T)-Tsol)/(Tliq-Tsol)
#     return
# end
@parallel function compute_eql!(χq::Data.Array, cxq::Data.Array, cmq::Data.Array, c::Data.Array, T::Data.Array, Tc0::Data.Number, Tc1::Data.Number, dc)
    @all(cxq) = (@all(T)-Tc0)./(Tc1-Tc0).*(1-dc)
    @all(cmq) =  @all(cxq) + dc
    @all(χq)  =  max(0,min(1,(@all(c)-@all(cmq))/(@all(cxq)-@all(cmq))))
    return
end

@parallel function compute_mixtflux!(qAx::Data.Array, qAy::Data.Array, A::Data.Array, Am::Data.Array, Ax::Data.Array, χ::Data.Array, Vx::Data.Array, Vy::Data.Array, Δwx::Data.Array, k::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(qAx)  = (1-@av_xi(χ)) * @av_xi(Am) * @all(Vx) + @av_xi(χ) * @av_xi(Ax) * (@all(Vx)            ) - k * @d_xi(A)/dx
    @all(qAy)  = (1-@av_xi(χ)) * @av_xi(Am) * @all(Vy) + @av_xi(χ) * @av_xi(Ax) * (@all(Vy) + @all(Δwx)) - k * @d_yi(A)/dy
    return
end

@parallel function compute_phaseflux!(qAx::Data.Array, qAy::Data.Array, A::Data.Array, Vx::Data.Array, Vy::Data.Array, Δwx::Data.Array, k::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(qAx)  = @av_xi(A)*(@all(Vx)          ) + k * @d_xi(A)/dx
    @all(qAy)  = @av_yi(A)*(@all(Vy)+@all(Δwx)) + k * @d_yi(A)/dy
    return
end

@parallel function compute_dAdt!(dAdt::Data.Array, qAx::Data.Array, qAy::Data.Array, QA::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(dAdt) = - @d_xa(qAx)/dx - @d_ya(qAy)/dy + @inn(QA)
    return
end

@parallel function compute_A!(A::Data.Array, A_old::Data.Array, dAdt::Data.Array, dAdt_old::Data.Array, dt::Data.Number)
    @inn(A) = @inn(A_old) + (@all(dAdt) + @all(dAdt_old))*dt/2
    return
end

@parallel function compute_react!(Gamma::Data.Array, Gammai::Data.Array, χ::Data.Array, χq::Data.Array, dt::Data.Number)
    @all(Gamma) = (@all(Gammai) + (@all(χq) - @all(χ))/(5*dt))/2
    return
end

@parallel function compute_latheat!(LatHeat::Data.Array, Gamma::Data.Array, LH::Data.Number)
    @all(LatHeat) = -@all(Gamma) * LH
    return
end

@parallel function compute_bndA!(bndA::Data.Array, A::Data.Array, bnd::Data.Array, Aw::Data.Number, tw::Data.Number)
    @all(bndA) = (Aw-@all(A))/tw * @all(bnd)
    return
end

@parallel function compute_P!(DivV::Data.Array, Pt::Data.Array, Vx::Data.Array, Vy::Data.Array, Gdτ::Data.Array, r::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(DivV) = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    @all(Pt)   = @all(Pt) - r*@all(Gdτ)*@all(DivV)
    return
end

@parallel function compute_τ!(τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, Vx::Data.Array, Vy::Data.Array, η::Data.Array, Gdτ::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(τxx) = (@all(τxx) + 2.0*@all(Gdτ)*@d_xa(Vx)/dx)/(@all(Gdτ)/@all(η) + 1.0)
    @all(τyy) = (@all(τyy) + 2.0*@all(Gdτ)*@d_ya(Vy)/dy)/(@all(Gdτ)/@all(η) + 1.0)
    @all(τxy) = (@all(τxy) + 2.0*@av(Gdτ)*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)))/(@av(Gdτ)/@av(η) + 1.0)
    return
end

@parallel function compute_dV!(Rvx::Data.Array, Rvy::Data.Array, dVx::Data.Array, dVy::Data.Array, Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, ρ::Data.Array, ρref::Data.Number, dτ_Rho::Data.Array, dx::Data.Number, dy::Data.Number, g0::Data.Number)
    @all(Rvx)  = @d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx
    @all(Rvy)  = @d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy - (@all(ρ)-ρref)*g0
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
    LH        = -30.0       # latent heat (normalised by cP)
    T0        = 1125        # background temperature
    Ti        = T0          # inclusion temperature
    Ta        = 1           # random perturbation amplitude
    c0        = 0.3         # background composition
    ci        = c0          # inclusion composition
    ca        = 0.001       # random perturbation amplitude
    Tc1       = 800.0       # lowest melting T
    Tc0       = 1200.0      # highest melting T
    dc        = 0.2         # solidus-liquidus composition step
    kT        = 1e-6        # thermal diffusivity
    kχ        = 1e-7        # phase diffusivity
    kc        = 1e-7        # component diffusivity
    Tw        = 300.0       # wall rock T
    tw        = 1000.0      # wall equilibration time
    # Numerics
    stepMax   = 1000        # maximum number of time steps
    tend      = 1000        # stopping time
    iterMax   = 1e5         # maximum number of pseudo-transient iterations
    nup       = 500         # interval for updating TC, residuals
    nop       = 5          # interval for output generation
    ε         = 1e-6        # nonlinear absolute tolerence
    CFL       = 0.9/sqrt(2) # Courant-Friedrich-Lewy number
    Re        = 5π          # ???
    r         = 1.0         # relative step size
    nx, ny    = 1*128-1, 1*128-1    # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    # Derived numerics
    dx, dy    = lx/nx, ly/ny # cell sizes
    max_lxy   = max(lx,ly)   # max domain extent
    max_dxy   = max(dx,dy)   # max cell size
    dt        = 0.05         # time step
    Vpdτ      = min(dx,dy)*CFL # ???
    xc, yc, xv, yv = LinRange(dx/2, lx - dx/2, nx), LinRange(dy/2, ly - dy/2, ny), LinRange(0, lx, nx+1), LinRange(0, ly, ny+1)
    # Array allocations
    DivV      = @zeros(nx  ,ny  )
    τxx       = @zeros(nx  ,ny  )
    τyy       = @zeros(nx  ,ny  )
    τxy       = @zeros(nx-1,ny-1)
    Rvx       = @zeros(nx-1,ny-2)
    Rvy       = @zeros(nx-2,ny-1)
    dVx       = @zeros(nx-1,ny-2)
    dVy       = @zeros(nx-2,ny-1)
    ητ        = @zeros(nx  ,ny  )
    Gdτ       = @zeros(nx  ,ny  )
    dτ_Rho    = @zeros(nx  ,ny  )
    qχx       = @zeros(nx-1,ny-2)
    qχy       = @zeros(nx-2,ny-1)
    Gamma     = @zeros(nx  ,ny  )
    Gammai    = copy(Gamma)
    dχdt      = @zeros(nx-2,ny-2)
    dχdt_old  = copy(dχdt)
    qTx       = @zeros(nx-1,ny-2)
    qTy       = @zeros(nx-2,ny-1)
    LatHeat   = @zeros(nx  ,ny  )
    bndT      = @zeros(nx  ,ny  )
    dTdt      = @zeros(nx-2,ny-2)
    dTdt_old  = copy(dTdt)
    qcx       = @zeros(nx-1,ny-2)
    qcy       = @zeros(nx-2,ny-1)
    dcdt      = @zeros(nx-2,ny-2)
    dcdt_old  = copy(dcdt)
    # Initial conditions
    Pt        = @zeros(nx  ,ny  )
    Vx        = @zeros(nx+1,ny  )
    Vy        = @zeros(nx  ,ny+1)
    Δwx       = @zeros(nx  ,ny+1)
    Vy        = @zeros(nx  ,ny+1)
    χq        = @zeros(nx  ,ny  )
    cxq       = @zeros(nx  ,ny  )
    cmq       = @zeros(nx  ,ny  )
    
    Rad2      = @zeros(nx  ,ny  )
    Rad2     .= [((ix-1)*dx +0.5*dx -0.5*lx)^2 + ((iy-1)*dy +0.5*dy -0.5*ly)^2 for ix=1:size(Rad2,1), iy=1:size(Rad2,2)]
    
    rng       = MersenneTwister(1524)
    dr        = randn(rng, (nx,ny))
    dr_old    = copy(dr)
    for ism=1:15
        @parallel assign!(dr_old, dr)
        @parallel smooth!(dr, dr_old)
        @parallel (1:size(dr,1)) bc_y!(dr)
        @parallel (1:size(dr,2)) bc_x!(dr)
    end
    dr       .= dr./(maximum(dr)-minimum(dr))
    
    T         = T0 * ones(nx,ny)
    T[Rad2.<1.0] .= Ti
    T        .= T .+ Ta.*dr
    T         = Data.Array( T )
    T_old     = copy(T)

    c         = c0 * ones(nx,ny)
    c[Rad2.<1.0] .= ci
    c        .= c .+ ca.*dr
    c         = Data.Array( c )
    c_old     = copy(c)

    bnd       = exp.(-yc'.*ones(nx)./dy) .+ exp.(-(ly.-yc)'.*ones(nx)./dy)
    @parallel compute_bndA!(bndT,T,bnd,Tw,tw)

    # @parallel compute_eql!(χq, T, Tsol, Tliq)
    @parallel compute_eql!(χq, cxq, cmq, c, T, Tc0, Tc1, dc)
    χ         = copy(χq)
    χ_old     = copy(χ)
    @parallel assign!(Gammai, Gamma)
    @parallel compute_react!(Gamma, Gammai, χ, χq, dt)
    ρ         =   @zeros(nx,ny)
    η         = η0*@ones(nx,ny)
    ητ       .= η

    # initialise material coefficients
    @parallel compute_η!(η, χ, η0)
    @parallel compute_ρ!(ρ, ρx0, ρm0, χ)
    @parallel compute_Δwx!(Δwx, η, ρ, ρx0, d0, g0)
    @parallel (1:size(Δwx,2)) bc_x!(Δwx)
    @parallel (1:size(Δwx,1)) bc_y0!(Δwx)

    # initialise iterative parameters
    @parallel compute_maxloc!(ητ, η)
    @parallel (1:size(ητ,2)) bc_x!(ητ)
    @parallel (1:size(ητ,1)) bc_y!(ητ)
    @parallel compute_iter_params!(dτ_Rho, Gdτ, ητ, Vpdτ, Re, r, max_lxy)
    
    ρref = mean(ρ)

    # visualise initial condition
    if do_viz
        p1 = heatmap(xc, yc, Array(T)'    , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="T")
        p2 = heatmap(xc, yc, Array(c)'    , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="c")
        p3 = heatmap(xc, yc, Array(χ)'    , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="χ")
        p4 = heatmap(xc, yc, Array(Gamma)', aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="Γ")
        display(plot(p1, p2, p3, p4))
    end

    # time loop
    step = 1; time = 0
    while time < tend && step <= stepMax
        @printf("\n*****  step = %d, time = %1.3e, dt = %1.3e \n", step, time, dt)

        # store previous solution
        @parallel assign!(T_old,T)
        @parallel assign!(dTdt_old,dTdt)
        @parallel assign!(c_old,c)
        @parallel assign!(dcdt_old,dcdt)
        @parallel assign!(χ_old,χ)
        @parallel assign!(dχdt_old,dχdt)

        # iterative solution loop
        err=2*ε; iter=1; err_evo1=[]; err_evo2=[]
        while err > ε && iter <= iterMax
            if (iter==1)  global wtime0 = Base.time()  end

            if iter % nup == 0
                if iter>0; dt = max_dxy/4/max(maximum(abs.(Vx.+1e-16)),maximum(abs.(Vy.+Δwx.+1e-16))); end

                # update TC-solution
                @parallel compute_bndA!(bndT,T,bnd,Tw,tw)
                @parallel compute_latheat!(LatHeat, Gamma, LH)
                @parallel compute_mixtflux!(qTx, qTy, T, T, T, χ, Vx, Vy, Δwx, kT, dx, dy)
                @parallel compute_dAdt!(dTdt, qTx, qTy, LatHeat + bndT, dx, dy)
                @parallel compute_A!(T, T_old, dTdt, dTdt_old, dt)
                @parallel (1:size(T,1)) bc_y!(T)
                @parallel (1:size(T,2)) bc_x!(T)

                @parallel compute_mixtflux!(qcx, qcy, c, cmq, cxq, χ, Vx, Vy, Δwx, kc, dx, dy)
                @parallel compute_dAdt!(dcdt, qcx, qcy, @zeros(nx,ny), dx, dy)
                @parallel compute_A!(c, c_old, dcdt, dcdt_old, dt)
                @parallel (1:size(c,1)) bc_y!(c)
                @parallel (1:size(c,2)) bc_x!(c)

                @parallel compute_eql!(χq, cxq, cmq, c, T, Tc0, Tc1, dc)

                @parallel assign!(Gammai, Gamma)
                @parallel compute_react!(Gamma, Gammai, χ, χq, dt)
                @parallel compute_phaseflux!(qχx, qχy, χ, Vx, Vy, Δwx, kχ, dx, dy)
                @parallel compute_dAdt!(dχdt, qχx, qχy, Gamma, dx, dy)
                @parallel compute_A!(χ, χ_old, dχdt, dχdt_old, dt)
                @parallel (1:size(χ,1)) bc_y!(χ)
                @parallel (1:size(χ,2)) bc_x!(χ)

                # p1 = heatmap(xc, yc, Array(T)'    , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="T")
                # p2 = heatmap(xc, yc, Array(c)'    , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="c")
                # p3 = heatmap(xc, yc, Array(χ)'    , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="χ")
                # p4 = heatmap(xc, yc, Array(Gamma)', aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="Γ")
                # display(plot(p1, p2, p3, p4))        

                # update material coefficients
                @parallel compute_η!(η, χ, η0)
                @parallel compute_ρ!(ρ, ρx0, ρm0, χ)
                @parallel compute_Δwx!(Δwx, η, ρ, ρx0, d0, g0)
                @parallel (1:size(Δwx,2)) bc_x!(Δwx)
                @parallel (1:size(Δwx,1)) bc_y0!(Δwx)

                # update iterative parameters
                @parallel compute_maxloc!(ητ, η)
                @parallel (1:size(ητ,2)) bc_x!(ητ)
                @parallel (1:size(ητ,1)) bc_y!(ητ)
                @parallel compute_iter_params!(dτ_Rho, Gdτ, ητ, Vpdτ, Re, r, max_lxy)
            end

            # update FM-solution
            @parallel compute_P!(DivV, Pt, Vx, Vy, Gdτ, r, dx, dy)

            @parallel compute_τ!(τxx, τyy, τxy, Vx, Vy, η, Gdτ, dx, dy)
            @parallel compute_dV!(Rvx, Rvy, dVx, dVy, Pt, τxx, τyy, τxy, ρ, ρref, dτ_Rho, dx, dy, g0)
            @parallel compute_V!(Vx, Vy, dVx, dVy)
            @parallel (1:size(Vx,1)) bc_y!(Vx)
            @parallel (1:size(Vy,2)) bc_x!(Vy)

            if iter % nup == 0 || iter==iterMax
                # check residuals
                @parallel compute_dV!(Rvx, Rvy, dVx, dVy, Pt, τxx, τyy, τxy, ρ, ρref, dτ_Rho, dx, dy, g0)
                Vmin, Vmax  = minimum(Vx), maximum(Vx)
                Pmin, Pmax  = minimum(Pt), maximum(Pt)
                norm_Rvx    = norm(Rvx)/sqrt(length(Rvx))/((Pmax-Pmin)/lx)
                norm_Rvy    = norm(Rvy)/sqrt(length(Rvy))/((Pmax-Pmin)/lx)
                norm_DivV   = norm(DivV )/sqrt(length(DivV ))/((Vmax-Vmin)/lx)
                err         = maximum([norm_Rvx, norm_Rvy, norm_DivV])
                push!(err_evo1, maximum([norm_Rvx, norm_Rvy, norm_DivV])); push!(err_evo2,iter)
                @printf("  --- iter = %d, err = %1.3e [norm_Rvx=%1.3e, norm_Rvy=%1.3e, norm_DivV=%1.3e] \n", iter, err, norm_Rvx, norm_Rvy, norm_DivV)
            end
            iter += 1
        end
        # Performance
        wtime    = Base.time() - wtime0
        A_eff    = (3*2)/1e9*nx*ny*sizeof(Data.Number)  # Effective main memoRvy access per iteration [GB] (Lower bound of required memoRvy access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
        wtime_it = wtime/(iter-10)                      # Execution time per iteration [s]
        T_eff    = A_eff/wtime_it                       # Effective memoRvy throughput [GB/s]
        @printf("      Total steps = %d, err = %1.3e, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", iter-1, err, wtime, round(T_eff, sigdigits=2))

        # Visualisation
        if do_viz && step % nop == 0 || step==stepMax
            p1 = heatmap(xc, yc, Array(T)'    , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="T")
            p2 = heatmap(xc, yc, Array(c)'    , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="c")
            p3 = heatmap(xc, yc, Array(χ)'    , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="χ")
            p4 = heatmap(xc, yc, Array(Gamma)', aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="Γ")
            #p4 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
            display(plot(p1, p2, p3, p4))
        end
        time += dt
        step += 1
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
