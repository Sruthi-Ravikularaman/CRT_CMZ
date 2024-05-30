function extend_r(x::Vector)

    nx = length(x)

    xl = Vector{Float64}(undef, nx-1) # r_-1, r_1, ..., r_M-3, r_M-2
    xl[1] = - x[1]
    xl[2:end] .= x[1:end-2]

    xu = Vector{Float64}(undef, nx-1) # r_2, r_3, ..., r_M-1, r_M
    xu[1:end] .= x[2:end]

    return xl, xu
end


function extend_z(x::Vector)

    nx = length(x)

    xl = Vector{Float64}(undef, nx-2) # z_1, z_2, ..., z_M-3, z_M-2
    xl[1:end] .= x[1:end-2]

    xu = Vector{Float64}(undef, nx-2) # z_3, z_4, ..., z_M-1, z_M
    xu[1:end] .= x[3:end]

    return xl, xu
end


function extend_p(x::Vector)

    nx = length(x)

    xu = Vector{Float64}(undef, nx-1) # p_2, p_3, ..., p_M-1, p_M
    xu[1:end] .= x[2:end]

    return xu
end


function extend_m(x::Vector)

    nx = length(x)

    xl = Vector{Float64}(undef, nx-2) # p_1, p_2, ..., p_M-3, p_M-2
    xl[1:end] .= x[1:end-2]

    xu = Vector{Float64}(undef, nx-2) # p_3, p_4, ..., p_M-1, p_M
    xu[1:end] .= x[3:end]

    return xl, xu
end


function delta_r(x::Vector)

    nx = length(x)

    delta_xl = Vector{Float64}(undef, nx-1) # r_1 - r_-1, ..., r_i - r_i-1, ..., r_M-1 - r_M-2
    delta_xl[1] = 2 * x[1] 
    delta_xl[2:end] .= x[2:end-1] .- x[1:end-2]

    delta_xu = Vector{Float64}(undef, nx-1) # r_2 - r_1, ..., r_i+1 - r_i, ..., r_M - r_M-1
    delta_xu[1:end] .= x[2:end] .- x[1:end-1]

    delta_xc = Vector{Float64}(undef, nx-1) # r_2 - r_-1, ..., r_i+1 - r_i-1, ..., r_M - r_M-2
    delta_xc[1] = x[2] + x[1]
    delta_xc[2:end] .= x[3:end] .- x[1:end-2]

    return delta_xl, delta_xc, delta_xu
end


function delta_z(x::Vector)

    nx = length(x)

    delta_xl = Vector{Float64}(undef, nx-2) # z_2 - z_1, ..., z_j - z_j-1, ..., z_M-1 - z_M-2
    delta_xl[1:end] .= x[2:end-1] .- x[1:end-2]

    delta_xu = Vector{Float64}(undef, nx-2) # z_3 - z_2, ..., z_j+1 - z_j, ..., z_M - z_M-1
    delta_xu[1:end] .= x[3:end] .- x[2:end-1]

    delta_xc = Vector{Float64}(undef, nx-2) # z_3 - z_1, ..., z_j+1 - z_j-1, ..., z_M - z_M-2
    delta_xc[1:end] .= x[3:end] .- x[1:end-2]

    return delta_xl, delta_xc, delta_xu 
end


function delta_p(x::Vector)

    nx = length(x)

    delta_xu = Vector{Float64}(undef, nx-1) # p_2 - p_1, ..., p_k+1 - p_k, ..., p_M - p_M-1
    delta_xu[1:end] .= x[2:end] .- x[1:end-1]

    return delta_xu
end


function delta_m(x::Vector)

    nx = length(x)

    delta_xl = Vector{Float64}(undef, nx-2) # p_2 - p_1, ..., p_k - p_k-1, ..., p_M-1 - p_M-2
    delta_xl[1:end] .= x[2:end-1] .- x[1:end-2]

    delta_xu = Vector{Float64}(undef, nx-2) # p_3 - p_2, ..., p_k+1 - p_k, ..., p_M - p_M-1
    delta_xu[1:end] .= x[3:end] .- x[2:end-1]

    delta_xc = Vector{Float64}(undef, nx-2) # p_3 - p_1, ..., p_k+1 - p_k-1, ..., p_M - p_M-2
    delta_xc[1:end] .= x[3:end] .- x[1:end-2]

    return delta_xl, delta_xc, delta_xu
end