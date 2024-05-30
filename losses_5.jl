using DelimitedFiles
using Interpolations
using LinearAlgebra


m_p = 938.272 # in MeV, proton mass
m_e = 0.511 # in MeV, electron mass
c = 3e10 # cm/s, velocity of light
global const sigma_T = 6.652e-25 #cm2, Thomson scattering cross section
global const m_el = 9.1094e-28 #g
global const e = 4.797e-10 # in cm^3/2 g^1/2 s-1
global const r_0 = e^2/(m_el*(c^2)) # cm electron radius
h_bar_cgs = 1.0546e-27 #cm2 g s-1
k_B = 8.6173e-11 # MeV K-1

function velocity(E_kin::Float64, rest_mass::Float64)

    gamma = 1 + (E_kin / rest_mass)
    beta = sqrt(1 - (1 / (gamma^2) ) )
    v = c * beta

    return v, beta, gamma
end


#----------------------PROTONS-------------------------------------

lossp = readdlm("lossfctp.txt", ',')
lp = linear_interpolation(lossp[1:end-4, 1], lossp[1:end-4, 2], extrapolation_bc=Line())


function E_p_dot(E::Float64, n::Float64)

    v, _, _ = velocity(E, m_p)
    loss_rate = (10^( lp(log10(E * 1e6)) )) * 1e-6
    
    return (1/2) * v * n * loss_rate
end

function p_p_dot(p::Float64, n::Float64)

    total_E = sqrt(p^2 + m_p^2)
    E = total_E - m_p
    dp_on_dT = total_E / p
    E_dot = E_p_dot(E, n)
    p_dot = dp_on_dT * E_dot

    return p_dot
end

# Loss time

function time_loss_p(E::Float64, n::Float64)
    return E/E_p_dot(E, n)
end

function tau_loss_p(p::Float64, n::Float64)
    total_E = sqrt(p^2 + m_p^2)
    E = total_E - m_p
    tau = time_loss_p(E, n)
    return tau
end


#---------------------ELECTRONS------------------------------------

#losse = readdlm("lossfcte.txt", ',') # From Padovani 2018 -> ionisation, BS, synchrotron
#le = linear_interpolation(losse[1:end-2, 1], losse[1:end-2, 2], extrapolation_bc=Line())

le_ion_data = readdlm("losse_ion.txt", ',')
le_ion = linear_interpolation(le_ion_data[1:end, 1], le_ion_data[1:end, 2],  extrapolation_bc=Line())

le_bs_data = readdlm("losse_bs.txt", ',')
le_bs = linear_interpolation(le_bs_data[1:end, 1], le_bs_data[1:end, 2],  extrapolation_bc=Line())


function E_e_dot_ion(E::Float64, n::Float64)
    v, _, _ = velocity(E, m_e)
    loss_rate = (10^(le_ion(log10(E * 1e6)))) * 1e-6
    return (1/2) * v * n * loss_rate
end

function p_e_dot_ion(p::Float64, n::Float64)
    total_E = sqrt(p^2 + m_e^2)
    E = total_E - m_e
    dp_on_dT = total_E/p
    E_dot = E_e_dot_ion(E, n)
    p_dot = dp_on_dT*E_dot
    return p_dot
end


function E_e_dot_bs(E::Float64, n::Float64)
    v, _, _ = velocity(E, m_e)
    loss_rate = (10^(le_bs(log10(E * 1e6)))) * 1e-6
    return (1/2) * v * n * loss_rate
end

function p_e_dot_bs(p::Float64, n::Float64)
    total_E = sqrt(p^2 + m_e^2)
    E = total_E - m_e
    dp_on_dT = total_E/p
    E_dot = E_e_dot_bs(E, n)
    p_dot = dp_on_dT*E_dot
    return p_dot
end


function E_e_dot_syn(E::Float64, B_uG::Float64)
    B_G = 1e-6 * B_uG
    B = 1e-4 * B_G
    mu_0 = 4 * pi * 1e-7 # Tm/A 
    B_2_2_mu = B^2 / (2 * mu_0) * 6.242e6 #TA/m = J -> MeV/cm3
    _, beta, gamma = velocity(E, m_e)
    loss_rate = (4/3) * sigma_T * c * (B_2_2_mu) * (beta^2) * (gamma^2)
    return loss_rate
end

function p_e_dot_syn(p::Float64, B_uG::Float64)
    total_E = sqrt(p^2 + m_e^2)
    E = total_E - m_e
    dp_on_dT = total_E/p
    E_dot = E_e_dot_syn(E, B_uG)
    p_dot = dp_on_dT*E_dot
    return p_dot
end

function g_iso(u::Float64)
    a_i = -0.362
    b_i = 0.826
    alpha_i = 0.682
    beta_i = 1.281
    g = 1.0 + ((a_i * (u^alpha_i)) / (1.0 + (b_i * (u^beta_i))))
    return 1 / g
end


function G_iso_0(u::Float64)
    c_iso = 5.68
    num = c_iso * u * log(1.0 + (0.722 * u / c_iso))
    den = 1.0 + (c_iso * u / 0.822)
    return num / den
end


function G_iso(u::Float64)
    return G_iso_0(u) * g_iso(u)
end

function E_e_dot_IC(E::Float64, T::Float64, k_dil::Float64)
    E = E / m_e
    T = k_B * T / m_e
    t = 4 * E * T
    pre_num = 2 * r_0^2 * m_el^3 * c^4 * k_dil * T^2
    pre_den = pi * h_bar_cgs^3 
    pre = pre_num / pre_den
    return pre * G_iso(t) * m_e 
end


function p_e_dot_IC(p::Float64, T::Float64, k_dil::Float64)
    total_E = sqrt(p^2 + m_e^2)
    E = total_E - m_e
    dp_on_dT = total_E/p
    E_dot = E_e_dot_IC(E, T, k_dil)
    p_dot = dp_on_dT*E_dot
    return p_dot
end