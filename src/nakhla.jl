const use_return  = haskey(ENV, "USE_RETURN" ) ? parse(Bool, ENV["USE_RETURN"] ) : false
const USE_GPU     = haskey(ENV, "USE_GPU"    ) ? parse(Bool, ENV["USE_GPU"]    ) : false
const do_viz      = haskey(ENV, "DO_VIZ"     ) ? parse(Bool, ENV["DO_VIZ"]     ) : true
const do_save     = haskey(ENV, "DO_SAVE"    ) ? parse(Bool, ENV["DO_SAVE"]    ) : false
const nx          = haskey(ENV, "NX"         ) ? parse(Int , ENV["NX"]         ) : 128 - 1
const ny          = haskey(ENV, "NY"         ) ? parse(Int , ENV["NY"]         ) : 128 - 1
const nop         = haskey(ENV, "NOP"        ) ? parse(Int , ENV["NOP"]        ) : 10

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

@parallel function compute_ρ!(ρ::Data.Array, ρm::Data.Array, ρx::Data.Array, T::Data.Array, cm::Data.Array, cx::Data.Array, χ::Data.Array, ρx0::Data.Number, ρm0::Data.Number, αT::Data.Number, γc::Data.Number)
    @all(ρm) = ρm0 * (1 - αT*(T-T0) - γc*(cm-c0))
    @all(ρx) = ρx0 * (1 - αT*(T-T0) - γc*(cx-c0))
    @all(ρ)  = (1-@all(χ))*@all(ρm) + @all(χ)*@all(ρx)
    return
end

@parallel function compute_segr!(Δwχ::Data.Array, wχ::Data.Array, χ::Data.Array, η::Data.Array, ρ::Data.Array, ρx0::Data.Number, d0::Data.Number, g0::Data.Number)
    @inn_y(Δwχ) = - (ρx0 - @av_ya(ρ)) * g0 * d0*d0 / @av_ya(η)
    @inn_y( wχ) = @av_ya(χ) * @inn_y(Δwχ)
    return
end

@parallel function compute_eql!(χq::Data.Array, cxq::Data.Array, cmq::Data.Array, c::Data.Array, T::Data.Array, Tc0::Data.Number, Tc1::Data.Number, dc)
    @all(cxq) = (@all(T)-Tc0)./(Tc1-Tc0).*(1-dc)
    @all(cmq) =  @all(cxq) + dc
    @all(χq)  =  max(0,min(1,(@all(c)-@all(cmq))/(@all(cxq)-@all(cmq))))
    return
end

@parallel function compute_mixtflux!(qAx::Data.Array, qAy::Data.Array, A::Data.Array, Am::Data.Array, Ax::Data.Array, χ::Data.Array, U::Data.Array, W::Data.Array, Δwχ::Data.Array, k::Data.Number, dx::Data.Number, dy::Data.Number)
    @inn_x(qAx)  = (1-@av_xa(χ)) * @av_xa(Am) * @inn_x(U) + @av_xa(χ) * @av_xa(Ax) * (@inn_x(U)              ) - k * @d_xa(A)/dx
    @inn_y(qAy)  = (1-@av_ya(χ)) * @av_ya(Am) * @inn_y(W) + @av_ya(χ) * @av_ya(Ax) * (@inn_y(W) + @inn_y(Δwχ)) - k * @d_ya(A)/dy
    return
end

@parallel function compute_phaseflux!(qAx::Data.Array, qAy::Data.Array, A::Data.Array, U::Data.Array, W::Data.Array, Δwχ::Data.Array, k::Data.Number, dx::Data.Number, dy::Data.Number)
    @inn_x(qAx)  = @av_xa(A)*(@inn_x(U)            ) - k * @d_xa(A)/dx
    @inn_y(qAy)  = @av_ya(A)*(@inn_y(W)+@inn_y(Δwχ)) - k * @d_ya(A)/dy
    return
end

@parallel function compute_dAdt!(dAdt::Data.Array, qAx::Data.Array, qAy::Data.Array, QA::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(dAdt) = - (@d_xa(qAx)/dx + @d_ya(qAy)/dy) + @all(QA)
    return
end

@parallel function compute_A!(A::Data.Array, A_old::Data.Array, dAdt::Data.Array, dAdt_old::Data.Array, dt::Data.Number)
    @all(A) = @all(A_old) + (@all(dAdt) + @all(dAdt_old))*dt/2
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

@parallel function compute_P!(RPt::Data.Array, Pt::Data.Array, DivV::Data.Array, U::Data.Array, W::Data.Array, wχ::Data.Array, Gdτ::Data.Array, r::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(DivV ) = @d_xa(U)/dx + @d_ya(W )/dy
    @all(RPt)   = @all(DivV)  + @d_ya(wχ)/dy
    @all(Pt)    = @all(Pt) - r * @all(Gdτ) * @all(RPt)
    return
end

@parallel function compute_τ!(τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, U::Data.Array, W::Data.Array, η::Data.Array, Gdτ::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(τxx) = (@all(τxx) + 2.0*@all(Gdτ)*@d_xa(U)/dx)/(@all(Gdτ)/@all(η) + 1.0)
    @all(τyy) = (@all(τyy) + 2.0*@all(Gdτ)*@d_ya(W)/dy)/(@all(Gdτ)/@all(η) + 1.0)
    @inn(τxy) = (@inn(τxy) + 2.0*@av(Gdτ)*(0.5*(@d_yi(U)/dy + @d_xi(W)/dx)))/(@av(Gdτ)/@av(η) + 1.0)
    return
end

@parallel function compute_dV!(RU::Data.Array, RW::Data.Array, dU::Data.Array, dW::Data.Array, Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, ρ::Data.Array, ρref::Data.Number, dτ_Rho::Data.Array, dx::Data.Number, dy::Data.Number, g0::Data.Number)
    @all(RU)  = @d_xa(τxx)/dx + @d_yi(τxy)/dy - @d_xa(Pt)/dx
    @all(RW)  = @d_ya(τyy)/dy + @d_xi(τxy)/dx - @d_ya(Pt)/dy - (@av_ya(ρ)-ρref)*g0
    @all(dU)  = @av_xa(dτ_Rho)*@all(RU)
    @all(dW)  = @av_ya(dτ_Rho)*@all(RW)
    return
end

@parallel function compute_V!(U::Data.Array, W::Data.Array, dU::Data.Array, dW::Data.Array)
    @inn_x(U) = @inn_x(U) + @all(dU)
    @inn_y(W) = @inn_y(W) + @all(dW)
    return
end

@parallel_indices (iy) function bc_x0!(A::Data.Array)
    A[1  , iy] = 0
    A[end, iy] = 0
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

@parallel_indices (ix) function bc_yns!(τxy::Data.Array,U::Data.Array)
    τxy[ix, 1  ] = -2*U[ix, 1  ]
    τxy[ix, end] = -2*U[ix, end]
    return
end

# @parallel_indices (iy) function bc_xns!(A::Data.Array)
#     A[1  , iy] = -A[2    , iy]
#     A[end, iy] = -A[end-1, iy]
#     return
# end

# @parallel_indices (ix) function bc_yns!(A::Data.Array)
#     A[ix, 1  ] = -A[ix, 2    ]
#     A[ix, end] = -A[ix, end-1]
#     return
# end

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
    T0        = 1125.0      # background temperature
    Ti        = T0          # inclusion temperature
    Ta        = 1.0         # random perturbation amplitude
    c0        = 0.35        # background composition
    ci        = c0          # inclusion composition
    ca        = -0.001      # random perturbation amplitude
    Tc1       = 800.0       # lowest melting T
    Tc0       = 1200.0      # highest melting T
    dc        = 0.3         # solidus-liquidus composition step
    kT        = 1e-6        # thermal diffusivity
    kχ        = 1e-7        # phase diffusivity
    kc        = 1e-7        # component diffusivity
    Tw        = 300.0       # wall rock T
    tw        = 6.e3        # wall equilibration time
    # Numerics
    stepMax   = 1e4         # maximum number of time steps
    tend      = 1e4         # stopping time
    iterMax   = 1e5         # maximum number of pseudo-transient iterations
    nup       = 500         # interval for updating TC, residuals
    #nop       = 5           # interval for output generation
    ε         = 1e-6        # nonlinear absolute tolerence
    CFL       = 0.9/sqrt(2) # Courant-Friedrich-Lewy number
    Re        = 5π          # ???
    r         = 1.0         # relative step size
    #nx, ny    = 1*128-1, 1*128-1    # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    # Derived numerics
    dx, dy    = lx/nx, ly/ny # cell sizes
    max_lxy   = max(lx,ly)   # max domain extent
    max_dxy   = max(dx,dy)   # max cell size
    dt        = 0.05         # time step
    Vpdτ      = min(dx,dy)*CFL # ???
    xc, yc, xv, yv = LinRange(dx/2, lx - dx/2, nx), LinRange(dy/2, ly - dy/2, ny), LinRange(0, lx, nx+1), LinRange(0, ly, ny+1)
    # Array allocations
    RPt       = @zeros(nx  ,ny  )
    DivV      = @zeros(nx  ,ny  )
    τxx       = @zeros(nx  ,ny  )
    τyy       = @zeros(nx  ,ny  )
    τxy       = @zeros(nx+1,ny+1)
    RU        = @zeros(nx-1,ny  )
    RW        = @zeros(nx  ,ny-1)
    dU        = @zeros(nx-1,ny  )
    dW        = @zeros(nx  ,ny-1)
    ητ        = @zeros(nx  ,ny                                                                                        )
    Gdτ       = @zeros(nx  ,ny  )
    dτ_Rho    = @zeros(nx  ,ny  )
    qχx       = @zeros(nx+1,ny  )
    qχy       = @zeros(nx  ,ny+1)
    Gamma     = @zeros(nx  ,ny  )
    Gammai    = copy(Gamma)
    dχdt      = @zeros(nx  ,ny  )
    dχdt_old  = copy(dχdt)
    qTx       = @zeros(nx+1,ny  )
    qTy       = @zeros(nx  ,ny+1)
    LatHeat   = @zeros(nx  ,ny  )
    bndT      = @zeros(nx  ,ny  )
    dTdt      = @zeros(nx  ,ny  )
    dTdt_old  = copy(dTdt)
    qcx       = @zeros(nx+1,ny  )
    qcy       = @zeros(nx  ,ny+1)
    dcdt      = @zeros(nx  ,ny  )
    dcdt_old  = copy(dcdt)
    # Initial conditions
    Pt        = @zeros(nx  ,ny  )
    U         = @zeros(nx+1,ny  )
    W         = @zeros(nx  ,ny+1)
    Δwχ       = @zeros(nx  ,ny+1)
    wχ        = @zeros(nx  ,ny+1)
    W         = @zeros(nx  ,ny+1)
    χq        = @zeros(nx  ,ny  )
    cxq       = @zeros(nx  ,ny  )
    cmq       = @zeros(nx  ,ny  )
       
    rng       = MersenneTwister(1524)
    dr        = randn(rng, (nx  ,ny  ))
    dr_old    = copy(dr)
    for ism=1:15
        @parallel assign!(dr_old, dr)
        @parallel smooth!(dr, dr_old)
        @parallel (1:size(dr,1)) bc_y!(dr)
        @parallel (1:size(dr,2)) bc_x!(dr)
    end
    dr       .= dr./(maximum(dr)-minimum(dr))
    
    T         = T0 * ones(nx  ,ny  )
    T        .= T .+ Ta.*dr
    T         = Data.Array( T )
    T_old     = copy(T)

    c         = c0 * ones(nx  ,ny  )
    c        .= c .+ ca.*dr
    c         = Data.Array( c )
    c_old     = copy(c)

    bnd       = exp.(-yc'.*ones(nx  )./dy) .+ exp.(-(ly.-yc)'.*ones(nx  )./dy)
    @parallel compute_bndA!(bndT,T,bnd,Tw,tw)

    # @parallel compute_eql!(χq, T, Tsol, Tliq)
    @parallel compute_eql!(χq, cxq, cmq, c, T, Tc0, Tc1, dc)
    χ         = copy(χq)
    χ_old     = copy(χ)
    @parallel assign!(Gammai, Gamma)
    @parallel compute_react!(Gamma, Gammai, χ, χq, dt)
    ρ         =   @zeros(nx,ny)
    ρm        =   @zeros(nx,ny)
    ρx        =   @zeros(nx,ny)
    η         = η0*@ones(nx,ny)
    ητ       .= η

    # initialise material coefficients
    @parallel compute_η!(η, χ, η0)
    @parallel compute_ρ!(ρ, ρm, ρx, T, cmq, cxq, χ, ρx0, ρm0, αT, γc)
    @parallel compute_segr!(Δwχ, wχ, χ, η, ρ, ρx0, d0, g0)
    # @parallel (1:size(Δwχ,2)) bc_x!( Δwχ)
    # @parallel (1:size( wχ,2)) bc_x!(  wχ)

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
                if iter>0; dt = max_dxy/8/max(maximum(abs.(U.+1e-16)),maximum(abs.(W.+Δwχ.+1e-16))); end

                # update TC-solution
                @parallel compute_bndA!(bndT,T,bnd,Tw,tw)
                @parallel compute_latheat!(LatHeat, Gamma, LH)
                @parallel compute_mixtflux!(qTx, qTy, T, T, T, χ, U, W, Δwχ, kT, dx, dy)
                @parallel compute_dAdt!(dTdt, qTx, qTy, LatHeat + bndT, dx, dy)
                @parallel compute_A!(T, T_old, dTdt, dTdt_old, dt)
                # @parallel (1:size(T,1)) bc_y!(T)
                # @parallel (1:size(T,2)) bc_x!(T)

                @parallel compute_mixtflux!(qcx, qcy, c, cmq, cxq, χ, U, W, Δwχ, kc, dx, dy)
                @parallel compute_dAdt!(dcdt, qcx, qcy, @zeros(nx,ny), dx, dy)
                @parallel compute_A!(c, c_old, dcdt, dcdt_old, dt)
                # @parallel (1:size(c,1)) bc_y!(c)
                # @parallel (1:size(c,2)) bc_x!(c)

                @parallel compute_eql!(χq, cxq, cmq, c, T, Tc0, Tc1, dc)

                @parallel assign!(Gammai, Gamma)
                @parallel compute_react!(Gamma, Gammai, χ, χq, dt)
                @parallel compute_phaseflux!(qχx, qχy, χ, U, W, Δwχ, kχ, dx, dy)
                @parallel compute_dAdt!(dχdt, qχx, qχy, Gamma, dx, dy)
                @parallel compute_A!(χ, χ_old, dχdt, dχdt_old, dt)
                # @parallel (1:size(χ,1)) bc_y!(χ)
                # @parallel (1:size(χ,2)) bc_x!(χ)

                # p1 = heatmap(xc, yc, Array(T)'    , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="T")
                # p2 = heatmap(xc, yc, Array(c)'    , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="c")
                # p3 = heatmap(xc, yc, Array(χ)'    , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="χ")
                # p4 = heatmap(xc, yc, Array(Gamma)', aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="Γ")
                # display(plot(p1, p2, p3, p4))        

                # update material coefficients
                @parallel compute_η!(η, χ, η0)
                @parallel compute_ρ!(ρ, ρm, ρx, T, cmq, cxq, χ, ρx0, ρm0, αT, γc)
                @parallel compute_segr!(Δwχ, wχ, χ, η, ρ, ρx0, d0, g0)
                # @parallel (1:size(Δwχ,2)) bc_x!( Δwχ)
                # @parallel (1:size( wχ,2)) bc_x!(  wχ)

                # update iterative parameters
                @parallel compute_maxloc!(ητ, η)
                @parallel (1:size(ητ,2)) bc_x!(ητ)
                @parallel (1:size(ητ,1)) bc_y!(ητ)
                @parallel compute_iter_params!(dτ_Rho, Gdτ, ητ, Vpdτ, Re, r, max_lxy)

                # p1 = heatmap(xv, yc, Array(U)' , aspect_ratio=1, xlims=extrema(xv), ylims=extrema(yc), c=:viridis, title="U")
                # p2 = heatmap(xc, yv, Array(W)' , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yv), c=:viridis, title="W")
                # p3 = heatmap(xc, yv, Array(Δwχ)', aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yv), c=:viridis, title="Δwχ")
                # p4 = heatmap(xc, yv, Array( wχ)', aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yv), c=:viridis, title="wχ")
                # display(plot(p1, p2, p3, p4))
            end

            # update FM-solution
            @parallel compute_P!(RPt, Pt, DivV, U, W, wχ, Gdτ, r, dx, dy)

            @parallel compute_τ!(τxx, τyy, τxy, U, W, η, Gdτ, dx, dy)
            @parallel (1:size(τxy,1)) bc_yns!(τxy,U)
            @parallel compute_dV!(RU, RW, dU, dW, Pt, τxx, τyy, τxy, ρ, ρref, dτ_Rho, dx, dy, g0)
            @parallel compute_V!(U, W, dU, dW)
            # @parallel (1:size(U,1)) bc_y0!(U)
            # @parallel (1:size(W,2)) bc_x0!(W)

            if iter % nup == 0 || iter==iterMax
                # check residuals
                @parallel compute_dV!(RU, RW, dU, dW, Pt, τxx, τyy, τxy, ρ, ρref, dτ_Rho, dx, dy, g0)
                Vmin, Vmax  = minimum(U), maximum(U)
                Pmin, Pmax  = minimum(Pt), maximum(Pt)
                norm_RU    = norm(RU)/sqrt(length(RU))/((Pmax-Pmin)/lx)
                norm_RW    = norm(RW)/sqrt(length(RW))/((Pmax-Pmin)/lx)
                norm_DivV   = norm(RPt)/sqrt(length(RPt))/((Vmax-Vmin)/lx)
                err         = maximum([norm_RU, norm_RW, norm_DivV])
                push!(err_evo1, maximum([norm_RU, norm_RW, norm_DivV])); push!(err_evo2,iter)
                @printf("  --- iter = %d, err = %1.3e [norm_RU=%1.3e, norm_RW=%1.3e, norm_DivV=%1.3e] \n", iter, err, norm_RU, norm_RW, norm_DivV)
            end
            iter += 1
        end
        # Performance
        wtime    = Base.time() - wtime0
        A_eff    = (3*2)/1e9*nx*ny*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
        wtime_it = wtime/(iter-10)                      # Execution time per iteration [s]
        T_eff    = A_eff/wtime_it                       # Effective memory throughput [GB/s]
        @printf("      Total steps = %d, err = %1.3e, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", iter-1, err, wtime, round(T_eff, sigdigits=2))

        # Visualisation
        if do_viz && step % nop == 0 || step==stepMax
            p1 = heatmap(xc, yc, Array(T)'    , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="T")
            p2 = heatmap(xc, yc, Array(c)'    , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="c")
            p3 = heatmap(xc, yc, Array(χ)'    , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="χ")
            p4 = heatmap(xc, yc, Array(Gamma)', aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:viridis, title="Γ")
            #p4 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
            display(plot(p1, p2, p3, p4))

            # p1 = heatmap(xv, yc, Array(U)' , aspect_ratio=1, xlims=extrema(xv), ylims=extrema(yc), c=:viridis, title="U")
            # p2 = heatmap(xc, yv, Array(W)' , aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yv), c=:viridis, title="W")
            # p3 = heatmap(xc, yv, Array(Δwχ)', aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yv), c=:viridis, title="Δwχ")
            # p4 = heatmap(xc, yv, Array( wχ)', aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yv), c=:viridis, title="wχ")
            # display(plot(p1, p2, p3, p4))
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
