using LinearAlgebra

include("CRT_tools_3D_4.jl")


function make_grids(nr::Int, nz::Int, np::Int, r_min::Float64, r_max::Float64, z_min::Float64, z_max::Float64, p_min::Float64, p_max::Float64)

    r_list = exp10.(collect(range(start = log10(r_min), stop = log10(r_max), length = nr)))

    z_poslist = exp10.(collect(range(start = log10(z_min), stop = log10(z_max), length = nz ÷ 2)))
    z_minlist = .- reverse(z_poslist)
    push!(z_minlist, 0)
    z_list = vcat(z_minlist, z_poslist)

    p_list = exp10.(collect(range(start = log10(p_min), stop = log10(p_max), length = np)))

    pu_list = extend_p(p_list) 
    
    return r_list, z_list, p_list, pu_list
end


function make_tau(p_list::Vector{Float64}, z_min::Float64, Dz::Function, Dp::Function, p_dot::Function, v_w::Float64, n::Float64)

    tau_diff = z_min^2 ./ Dz.(1.0, 1.0, p_list) # Time for p to diffuse across z_min
    min_tau_diff = minimum(tau_diff)
    println("Minimum diffusion timescale = ", min_tau_diff, " s")

    tau_adve = (z_min / v_w) .* ones(np) # Time for p to advect across z_min with a wind speed of v_w
    min_tau_adve = minimum(tau_adve)
    println("Minimum advection timescale = ", min_tau_adve, " s")

    tau_loss = (p_list[2:end] .- p_list[1:end-1]) ./ p_dot.(p_list[2:end], n) # Time for p_i+1 to lose momentum to reach p_i
    min_tau_loss = minimum(tau_loss)
    println("Minimum loss timescale = ", min_tau_loss, " s")

    tau_reac = p_list[2:end].^2 ./ Dp.(1.0, 1.0, p_list[1:end-1]) # Time for p_i to gain momentum to reach p_i+1
    min_tau_reac = minimum(tau_reac)
    println("Minimum reacceleration timescale = ", min_tau_reac, " s")


    Dt_min = min(min_tau_diff, min_tau_adve, min_tau_loss, min_tau_reac) / 2.0 # Shortest time / 2

    return min_tau_diff, min_tau_adve, min_tau_loss, min_tau_reac, Dt_min
end


function make_Q(r_list::Vector{Float64}, z_list::Vector{Float64}, p_list::Vector{Float64}, pu_list::Vector{Float64}, Qf::Function)
  
    Q = zeros(length(r_list), length(z_list), length(p_list)) # nr, nz, np
    Qu = zeros(length(r_list), length(z_list), length(pu_list)) # nr, nz, np-1

    for k in eachindex(p_list)
        for j in eachindex(z_list)
            for i in eachindex(r_list)
                Q[i, j, k] = Qf(r_list[i], z_list[j], p_list[k], r_list, z_list) # Q(r_i, z_j, p_k)
            end
        end
    end

    
    for k in eachindex(pu_list)
        for j in eachindex(z_list)
            for i in eachindex(r_list)
                Qu[i, j, k] = Qf(r_list[i], z_list[j], pu_list[k], r_list, z_list) # Q(r_i, z_i, p_k+1)
            end
        end
    end

    return Q, Qu
end


function make_Ar(r_list::Vector{Float64}, z_list::Vector{Float64}, p_list::Vector{Float64}, Dr::Function, Dt::Float64)

    A_nextr = Array{Tridiagonal{Float64, Vector{Float64}}, 2}(undef, length(z_list), length(p_list))
    A_nr = Array{Tridiagonal{Float64, Vector{Float64}}, 2}(undef, length(z_list), length(p_list))

    nr = length(r_list)

    rc_list = r_list[1:end-1] # r_1, r_2, ..., r_M-1
    rl_list, ru_list = extend_r(r_list) # r_-1, r_1, r_2, ..., r_M-2 and r_2, ..., r_M
    dRl_list, dRc_list, dRu_list = delta_r(r_list) 
    # r_1 - r_-1, ..., r_M-1 - r_M-2 and r_2 - r_-1, ..., r_M - r_M-2 and r_2 - r_1, ..., r_M - r_M-1 
        
    Threads.@threads for k in eachindex(p_list)
        @simd for j in eachindex(z_list)

            Dr_list = zeros(nr-1)
            Drl_list = zeros(nr-1)
            Dru_list = zeros(nr-1)

            Dr_list .= Dr.(rc_list, z_list[j], p_list[k])
            Drl_list .= Dr.(rl_list, z_list[j], p_list[k])
            Dru_list .= Dr.(ru_list, z_list[j], p_list[k])

            L = zeros(nr-1)
            C = zeros(nr-1)
            U = zeros(nr-1)

            L .= 2.0 .* Dr_list ./ (dRc_list .* dRl_list)
            L .-= Dr_list ./ (rc_list .* dRc_list)
            L .-= (Dru_list .- Drl_list) ./ (dRc_list .^2)
        
            C .= 2.0 .* Dr_list ./ dRc_list
            C .*= (1.0 ./ dRu_list) .+ (1.0 ./ dRl_list)
        
            U .= 2.0 .* Dr_list ./ (dRc_list .* dRu_list)
            U .+= Dr_list ./ (rc_list .* dRc_list)
            U .+= (Dru_list .- Drl_list) ./ (dRc_list .^2)
            
            next_ldiag = zeros(nr-1)
            n_ldiag = zeros(nr-1)
            next_diag = zeros(nr)
            n_diag = zeros(nr)
            next_udiag = zeros(nr-1)
            n_udiag = zeros(nr-1)

            next_ldiag[1:end-1] .= - (Dt / 2) * L[2:end]
            next_ldiag[end] = 0
            n_ldiag[1:end-1] .= (Dt / 2) * L[2:end]
            n_ldiag[end] = 0

            next_diag[1:end-1] .= 1 .+ ((Dt / 2) * C[1:end])
            next_diag[1] -= (Dt / 2) * L[1]
            next_diag[end] = 1.0
            n_diag[1:end-1] = 1 .- ((Dt / 2) * C[1:end])
            n_diag[1] += (Dt / 2) * L[1]
            n_diag[end] = 1.0

            next_udiag[1:end] = - (Dt / 2) * U[1:end]
            n_udiag[1:end] = (Dt / 2) * U[1:end]

            A_nextr[j, k] = Tridiagonal(next_ldiag, next_diag, next_udiag)
            A_nr[j, k]  = Tridiagonal(n_ldiag, n_diag, n_udiag)
        end
    end
    
    return A_nextr, A_nr
end


function make_Br(Ar::Matrix{Tridiagonal{Float64, Vector{Float64}}}, prev::Array{Float64}, Q_arr::Array{Float64}, Dt::Float64)
    
    nr, nz, np = size(prev)
    B_r = zeros(nz, np, nr)

    Threads.@threads for k in 1:np
        @simd for j in 1:nz
            B_r[j, k, :] = (Ar[j, k] * prev[:, j, k]) + (Q_arr[:, j, k] * Dt)
            B_r[j, k, end] = 0 
        end
    end

    return B_r
end


function make_Az(r_list::Vector{Float64}, z_list::Vector{Float64}, p_list::Vector{Float64}, Dz::Function, v_z::Function, Dt::Float64)
    
    A_nextz = Array{Tridiagonal{Float64, Vector{Float64}}, 2}(undef, length(r_list), length(p_list))
    A_nz = Array{Tridiagonal{Float64, Vector{Float64}}, 2}(undef, length(r_list), length(p_list))

    nz = length(z_list)

    zc_list = z_list[2:end-1] # z_2, ..., z_M-1
    v_list = v_z.(zc_list)
    zl_list, zu_list = extend_z(z_list) # z_1, ..., z_M-2 and z_3, ..., z_M
    dZl_list, dZc_list, dZu_list = delta_z(z_list) 
    # z_2 - z_1, ..., z_M-1 - z_M-2 and z_3 - z_1, ..., z_M - z_M-2 and z_3 - z_2, ..., z_M - z_M-1
        
    Threads.@threads for k in eachindex(p_list)
        @simd for i in eachindex(r_list)
            
            Dz_list = zeros(nz-2)
            Dzl_list = zeros(nz-2)
            Dzu_list = zeros(nz-2)

            Dz_list .= Dz.(r_list[i], zc_list, p_list[k])
            Dzl_list .= Dz.(r_list[i], zl_list, p_list[k])
            Dzu_list .= Dz.(r_list[i], zu_list, p_list[k])

            Lz = zeros(nz-2)
            Cz = zeros(nz-2)
            Uz = zeros(nz-2)
        
            Lz .= 2.0 .* Dz_list ./ (dZc_list .* dZl_list)
            Lz .-= (Dzu_list .- Dzl_list) ./ (dZc_list .^2)
        
            Cz .= 2.0 .* Dz_list ./ dZc_list
            Cz .*= (1.0 ./ dZu_list) .+ (1.0 ./ dZl_list)
        
            Uz .= 2.0 .* Dz_list ./ (dZc_list .* dZu_list)
            Uz .+= (Dzu_list .- Dzl_list) ./ (dZc_list .^2)
        
            La = zeros(nz-2)
            Ca = zeros(nz-2)
            Ua = zeros(nz-2)
            
            La[(nz-2)÷2+2:end] .= v_list[(nz-2)÷2+1:end-1] ./ dZl_list[(nz-2)÷2+2:end]
            Ca[(nz-2)÷2+2:end] .= v_list[(nz-2)÷2+2:end] ./ dZl_list[(nz-2)÷2+2:end]
            Ua[(nz-2)÷2+2:end] .= zeros((nz-2)÷2)
            
            La[1:(nz-2)÷2] .= zeros((nz-2)÷2)
            Ca[1:(nz-2)÷2] .= .- v_list[1:(nz-2)÷2] ./ dZu_list[1:(nz-2)÷2]
            Ua[1:(nz-2)÷2] .= .- v_list[2:(nz-2)÷2+1] ./ dZu_list[1:(nz-2)÷2]
            
            La[(nz-2)÷2+1] = v_list[(nz-2)÷2] / dZc_list[(nz-2)÷2+1]
            Ca[(nz-2)÷2+1] = 0
            Ua[(nz-2)÷2+1] = - v_list[(nz-2)÷2+2] / dZc_list[(nz-2)÷2+1]

            L = Lz + La
            C = Cz + Ca
            U = Uz + Ua

            next_ldiag = zeros(nz-1)
            n_ldiag = zeros(nz-1)
            next_diag = zeros(nz)
            n_diag = zeros(nz)
            next_udiag = zeros(nz-1)
            n_udiag = zeros(nz-1)
        
            next_ldiag[1:end-1] = - (Dt / 2) .* L[1:end]
            next_ldiag[end] = 0 
            n_ldiag[1:end-1] = (Dt / 2) .* L[1:end]
            n_ldiag[end] = 0

            next_diag[2:end-1] = 1 .+ ((Dt / 2) .* C[1:end])
            next_diag[1] = 1.0
            next_diag[end] = 1.0
            n_diag[2:end-1] = 1 .- ((Dt / 2) .* C[1:end])
            n_diag[1] = 1.0
            n_diag[end] = 1.0
        
            next_udiag[2:end] = - (Dt / 2) .* U[1:end]
            next_udiag[1] = 0
            n_udiag[2:end] = (Dt / 2) .* U[1:end]
            n_udiag[1] = 0
        
            A_nextz[i, k] = Tridiagonal(next_ldiag, next_diag, next_udiag)
            A_nz[i, k] = Tridiagonal(n_ldiag, n_diag, n_udiag)
        
        end
    end

    return A_nextz, A_nz
end


function make_Bz(Az::Matrix{Tridiagonal{Float64, Vector{Float64}}}, prev::Array{Float64}, Q_arr::Array{Float64}, Dt::Float64)
    
    nr, nz, np = size(prev)
    B_z = zeros(nr, np, nz)

    Threads.@threads for k in 1:np
        @simd for i in 1:nr
            B_z[i, k, :] = (Az[i, k] * prev[i, :, k]) + (Q_arr[i, :, k] * Dt)
            B_z[i, k, 1] = 0
            B_z[i, k, end] = 0 
        end
    end

    return B_z
end


function make_Ap(r_list::Vector{Float64}, z_list::Vector{Float64}, p_list::Vector{Float64}, P_dot::Function, Dt::Float64)

    A_nextp = Array{Tridiagonal{Float64, Vector{Float64}}, 2}(undef, length(r_list), length(z_list))
    A_np = Array{Tridiagonal{Float64, Vector{Float64}}, 2}(undef, length(r_list), length(z_list))

    np = length(p_list)

    pc_list = p_list[1:end-1] # p_1, ..., p_M-1
    pu_list = extend_p(p_list) # p_2, ..., p_M
    dPu_list = delta_p(p_list) # p_2 - p_1, ..., p_M - p_M-1

    Threads.@threads for j in eachindex(z_list)
        @simd for i in eachindex(r_list)
    
            P_dot_list = zeros(np-1)
            P_dotu_list = zeros(np-1)

            P_dot_list .= P_dot.(r_list[i], z_list[j], pc_list)
            P_dotu_list .= P_dot.(r_list[i], z_list[j], pu_list)

            C = zeros(np-1)
            U = zeros(np-1)
    
            C .= P_dot_list ./ dPu_list
    
            U .= P_dotu_list ./ dPu_list

            next_diag = zeros(np)
            n_diag = zeros(np)
            next_udiag = zeros(np-1)
            n_udiag = zeros(np-1)
    
            next_diag[1:end-1] .= 1 .+ (Dt .* C[1:end])
            next_diag[end] = 1.0
            n_diag[1:end-1] = 1 .- (Dt .* C[1:end])
            n_diag[end] = 1.0
    
            next_udiag[1:end] .= 1 .- (Dt .* U[1:end])
            n_udiag[1:end] .= 1 .+ (Dt .* U[1:end])
    
            A_nextp[i, j] = Tridiagonal(zeros(length(next_udiag)), next_diag, next_udiag)
            A_np[i, j] = Tridiagonal(zeros(length(n_udiag)), n_diag, n_udiag)
    
        end
    end

    return A_nextp, A_np
end


function make_Bp(Ap::Matrix{Tridiagonal{Float64, Vector{Float64}}}, prev::Array{Float64}, Q_arr::Array{Float64}, Qu_arr::Array{Float64}, Dt::Float64)
    
    nr, nz, np = size(prev)
    B_p = zeros(nr, nz, np)
    

    Threads.@threads for j in 1:nz
        @simd for i in 1:nr
            A_part = (Ap[i, j] * prev[i, j, :])
            B_p[i, j, 1:end-1] = (A_part[1:end-1]) + ((Q_arr[i, j, 1:end-1] +  Qu_arr[i, j, :]) * Dt)
            B_p[i, j, end] = 0 
        end
    end

    return B_p
end


function make_Am(r_list::Vector{Float64}, z_list::Vector{Float64}, p_list::Vector{Float64}, Dp::Function, Dt::Float64)

    A_nextm = Array{Tridiagonal{Float64, Vector{Float64}}, 2}(undef, length(r_list), length(z_list))
    A_nm = Array{Tridiagonal{Float64, Vector{Float64}}, 2}(undef, length(r_list), length(z_list))

    np = length(p_list)

    pc_list = p_list[2:end-1] # p_2, ..., p_M-1
    pl_list, pu_list = extend_m(p_list) # p_1, ..., p_M-2 and # p_3, ..., p_M
    dPl_list, dPc_list, dPu_list = delta_m(p_list)
    # p_2 - p_1, ..., p_M-1 - p_M-2 and p_3 - p_1, ..., p_M - p_M-2 and p_3 - p_2, ..., p_M - p_M-1

    Threads.@threads for j in eachindex(z_list)
        @simd for i in eachindex(r_list)

            Dp_list = zeros(np-2)
            Dpl_list = zeros(np-2)
            Dpu_list = zeros(np-2)

            Dp_list .= Dp.(r_list[i], z_list[j], pc_list)
            Dpl_list .= Dp.(r_list[i], z_list[j], pl_list)
            Dpu_list .= Dp.(r_list[i], z_list[j], pu_list)

            L = zeros(np-2)
            C = zeros(np-2)
            U = zeros(np-2)

            L .= - (Dpu_list .- Dpl_list) ./ (dPc_list .^2)
            L .+= (2.0 .* Dp_list ./ (dPc_list .* dPl_list))
            L .+= (2.0 .* Dpl_list ./ (dPl_list .* pl_list))

            C .= (2.0 .* Dp_list ./ dPc_list) .* ((1.0 ./ dPu_list) .+ (1.0 ./ dPl_list))
            C .+= 2.0 .* Dp_list ./ (dPl_list .* pc_list) 

            U .= (Dpu_list .- Dpl_list) ./ (dPc_list .^2)
            U .+= (2.0 .* Dp_list ./ (dPc_list .* dPu_list))

            next_ldiag = zeros(np-1)
            n_ldiag = zeros(np-1)
            next_diag = zeros(np)
            n_diag = zeros(np)
            next_udiag = zeros(np-1)
            n_udiag = zeros(np-1)

            next_ldiag[1:end-1] = - (Dt / 2) .* L[1:end]
            next_ldiag[end] = 0
            n_ldiag[1:end-1] = (Dt / 2) .* L[1:end]
            n_ldiag[end] = 0

            next_diag[2:end-1] = 1 .+ ((Dt / 2) .* C[1:end])
            next_diag[1] = 1.0
            next_diag[end] = 1.0
            n_diag[2:end-1] = 1 .- ((Dt / 2) .* C[1:end])
            n_diag[1] = 1.0
            n_diag[end] = 1.0

            next_udiag[2:end] = - (Dt / 2) .* U[1:end]
            next_udiag[1] -=  (p_list[1] ^2 / p_list[2] ^2)
            n_udiag[2:end] = (Dt / 2) .* U[1:end]
            n_udiag[1] -= (p_list[1] ^2 / p_list[2] ^2)

            A_nextm[i, j] = Tridiagonal(next_ldiag, next_diag, next_udiag)
            A_nm[i, j] = Tridiagonal(n_ldiag, n_diag, n_udiag)

        end
    end

    return A_nextm, A_nm
end


function make_Bm(Am::Matrix{Tridiagonal{Float64, Vector{Float64}}}, prev::Array{Float64}, Q_arr::Array{Float64}, Dt::Float64)
    
    nr, nz, np = size(prev)
    B_m = zeros(nr, nz, np) 

    Threads.@threads for j in 1:nz
        @simd for i in 1:nr
            B_m[i, j, :] = (Am[i, j] * prev[i, j, :]) + (Q_arr[i, j, :] * Dt)
            B_m[i, j, 1] = 0
            B_m[i, j, end] = 0 
        end
    end

    return B_m
end