
using LabelledArrays
using NamedTupleTools
using DifferentialEquations

function adjust_pop_dist!(p)
  # adjust the population distribution based on population, i0, and s0
  pop = p.population * p.n0; # effective population size
  prop_S = p.s0 * (1 - p.i0); # parameterized in a way that a fraction of S, p.s0, (i.e, pop at risk) is I and therefore, truly susepctible proportion is prop_S
  # for pre-emptive vaccination, the susceptible proportion is reduced by vaccine coverage
  if p.preemptive_vacc
    prop_S = prop_S * (1 - p.vacc_1d_cov); # susceptible proportion
  end
  S = pop * prop_S; # susceptible people
  I = pop * p.s0 * p.i0; # infected people
  prop_u5 = p.prop_u5;
  p.u0.S_1 = S * prop_u5; #
  p.u0.S_2 = S * (1 - prop_u5);
  # I and R states were modelled with two compartments. We simplify and equally divide the pop into two compartments
  prop_R = 1 - p.s0; # since p.s0 includes both I (including E and A) and S compartments 1-p.s0 are recovered people
  
  p.u0.R1_1 = 1/2 * pop * prop_R * prop_u5;
  p.u0.R2_1 = 1/2 * pop * prop_R * prop_u5; 
  p.u0.R1_2 = 1/2 * pop * prop_R * (1 - prop_u5);
  p.u0.R2_2 = 1/2 * pop * prop_R * (1 - prop_u5);
  
  latent_pd = 1/p.epsilon;
  infect_pd = 1/p.gamma;
  fE = latent_pd/(latent_pd + infect_pd); # approximate fraction of E, i.e., fE for every I
  # fi = infect_pd/(latent_pd + infect_pd);

  # putting E in the beginning makes the definition of incidence in the novacc vs. vacc inconsitent
  # because incidence in the no vacc includes E -> I, therefore, initial seeds contribute to the incidence in novacc
  # therefore fE = 0 is added, to nullify coded. Dec 5, 2024
  fE = 0;
  p.u0.E1_1 = 1/2 * I * fE * prop_u5;
  p.u0.E2_1 = 1/2 * I * fE * prop_u5;
  p.u0.E1_2 = 1/2 * I * fE * (1 - prop_u5);
  p.u0.E2_2 = 1/2 * I * fE * (1 - prop_u5);

  p.u0.I1_1 = 1/2 * I * (1-fE) * (1-p.fA) * prop_u5;
  p.u0.I2_1 = 1/2 * I * (1-fE) * (1-p.fA) * prop_u5;
  p.u0.I1_2 = 1/2 * I * (1-fE) * (1-p.fA) * (1 - prop_u5);
  p.u0.I2_2 = 1/2 * I * (1-fE) * (1-p.fA) * (1 - prop_u5);
  
  p.u0.A1_1 = 1/2 * I * (1-fE) * p.fA * prop_u5;
  p.u0.A2_1 = 1/2 * I * (1-fE) * p.fA * prop_u5;
  p.u0.A1_2 = 1/2 * I * (1-fE) * p.fA * (1 - prop_u5);
  p.u0.A2_2 = 1/2 * I * (1-fE) * p.fA * (1 - prop_u5);

  if p.preemptive_vacc
    V = pop * p.vacc_1d_cov * p.s0 * (1 - p.i0); 
    # The individuals in the first two compartments (V1_1, V1_2, V2_1, and V2_2) are susceptible 
    # and therefore, 3rd and 4th compartments are filled for pre-emptive vaccination assuming all vaccinated people have progressed to the 3rd and 4th compartments 
    p.u0.V3_1 = 1/2 * V * prop_u5;
    p.u0.V4_1 = 1/2 * V * prop_u5;
    p.u0.V3_2 = 1/2 * V * (1 - prop_u5);
    p.u0.V4_2 = 1/2 * V * (1 - prop_u5);
  end

  return p    
end 


# function update_params!(params, new_params)
#   is_pop_changed = false;  
#   for k in keys(new_params) # extract the keys  
#     params[k] = new_params[k];
#     if k ∈ [:population, :s0, :i0] && !is_pop_changed # population size changes then change those in each state
#        is_pop_changed = true;
#     end  
#   end
#   if is_pop_changed
#     params = adjust_pop_dist!(params); # if population size, proportion of s0, or i0, adjust the pop distribution 
#   end  # replace the value by key
#   return params # return the updated parameters
# end
# test
# simple function to update the distribution 
# new_p=LVector(R0=1,i0=0.02,s0=0.8);
# for k in keys(new_p)
#   if k ∈ [:population, :s0, :i0]
#   # if k ∈ ["population", "s0", "i0"]  
#     println(k);
#   end
# end

function update_LArray!(la1, la2)
  for k in keys(la2)
    la1[k] = la2[k];
  end
  return la1
end

function initialize_u0(p = nothing)
  u0_1 = LVector(S_1=0.99, E1_1=0.0, E2_1=0.0, I1_1=0.005, I2_1=0.005, A1_1=0.0, A2_1=0.0, R1_1=0.0, R2_1=0.0, CE_1=0.0, CI_1=0.0, V1_1=0.0, V2_1=0.0, V3_1=0.0, V4_1=0.0,
  RS_1=0, VE1_1=0, VI1_1=0, VA1_1=0, VR1_1=0, VE2_1=0, VI2_1=0, VA2_1=0, VR2_1=0, VCI_1=0, UCI_1=0);
  u0_2 = LVector(S_2=0.99, E1_2=0.0, E2_2=0.0, I1_2=0.005, I2_2=0.005, A1_2=0.0, A2_2=0.0, R1_2=0.0, R2_2=0.0, CE_2=0.0, CI_2=0.0, V1_2=0.0, V2_2=0.0, V3_2=0.0, V4_2=0.0,
  RS_2=0, VE1_2=0, VI1_2=0, VA1_2=0, VR1_2=0, VE2_2=0, VI2_2=0, VA2_2=0, VR2_2=0, VCI_2=0, UCI_2=0);
  u0 = [u0_1*0.3; u0_2*0.7];
  if !isnothing(p)
    u0 = update_LArray!(u0, p)
  end
  return u0
end


function initialize_u0_novacc(p = nothing)
  u0_1 = LVector(S_1=0.99, E1_1=0.0, E2_1=0.0, I1_1=0.005, I2_1=0.005, A1_1=0.0, A2_1=0.0, R1_1=0.0, R2_1=0.0, CE_1=0.0, CI_1=0.0);
  u0_2 = LVector(S_2=0.99, E1_2=0.0, E2_2=0.0, I1_2=0.005, I2_2=0.005, A1_2=0.0, A2_2=0.0, R1_2=0.0, R2_2=0.0, CE_2=0.0, CI_2=0.0);
  u0 = [u0_1*0.3; u0_2*0.7];
  if !isnothing(p)
    u0 = update_LArray!(u0, p)
  end
  return u0
end

function initialize_params(p = nothing)
  param_vector = (
    u0 = initialize_u0(),
    s0 = 1.0,
    i0 = 0.01,
    R0 = 3.0,
    n0 = 1.0, # effective proportion of population (population at risk = population * n0)
    population = 1.0,
    prop_detection = 1.0, 
    # baseline population size is critical for the outbreak size
    # and the currently is the the population size of the admin in which the
    # outbreak was reported. Actual population is likely to be a fraction
    # prop_eff_pop = 1 
    # time window over which the number of cases is tracked (i.e., weekly reported)
    # obs_length refers the duration during which data are available in days
    # this is updated for each data set
    tend = 100.0,
    report_freq = 1.0,
    obs_length = 365.0, # 20 weeks
    # this refers to the total simulation days and has to be larger than or equal
    # to the obs_length
    # tau = 0.01, # time step size for numerical integration
    # epsilon = 1/1.4, # mean latent period = 1/epsilon
    # incubation period ~ Erlang(shape=2,rate=1.115044) based on the fitting of Erlang dist to log-normal with median of 1.4 and dispersion of 1.98
    # refer to the blog post: https://www.jonghoonk.com/posts/exp_erlang_lognormal/
    # estimated rate is divided by 2 as the rate in the ODE is implemented by multiplying with 2
    epsilon = 1.115044 / 2, # mean late period is still around 1.4 days, 
    gamma = 1/2, # mean infectious period = 1/gamma
    fA = 0.5, # fraction of asymptomatic state θ
    bA = 0.05, # relative infectiousness of asymptomatic state
    kappa = 575.0, # excretion rate cells per person per day
    # prop_children = 0.193, #
    prop_u5 = 0.193, #
  # exponent used to model sub-exponential growth (0 < expon < 1) (foi=I^{expon}*S/N)
    expon = 1.0, # exponential growth when expo = 1
    xi = 1/21, # mean decay rate of Vibrio cholerae
    K = 10000.0, # half-infective bacteria dose (10,000 cells/ml)
    R0W = 0.0, #
    sigma = 0, # 1/sigma = mean duration of natural immunity
    # 1/sigma_vacc_1d = mean duration of OCV-induced immunity (1st dose)
    # sigma_vacc_1d = 1/(2*365),
    # waiting time in vaccinated state ~ Erlang(shape=2,rate=1.656831/365) based on the fitting of Erlang dist to log-normal with median of 1.4 and dispersion of 1.98
    # refer to the blog post: https://www.jonghoonk.com/posts/vacc_waning_nls_erlang/
    # estimated rate is divided by 2 as the rate in the ODE is implemented by multiplying with 2
    sigma_vacc_1d_u5 = 0, 
    # 1/sigma_vacc_2d = mean duration of OCV-induced immunity (2nd dose)
    sigma_vacc_1d = 0,
    # sigma = 1/(4*365), # 1/sigma = mean duration of natural immunity
    # # 1/sigma_vacc_1d = mean duration of OCV-induced immunity (1st dose)
    # # sigma_vacc_1d = 1/(2*365),
    # # waiting time in vaccinated state ~ Erlang(shape=2,rate=1.656831/365) based on the fitting of Erlang dist to log-normal with median of 1.4 and dispersion of 1.98
    # # refer to the blog post: https://www.jonghoonk.com/posts/vacc_waning_nls_erlang/
    # # estimated rate is divided by 2 as the rate in the ODE is implemented by multiplying with 2
    # sigma_vacc_1d_u5 = (1.656831/365)/2, 
    # # 1/sigma_vacc_2d = mean duration of OCV-induced immunity (2nd dose)
    # sigma_vacc_1d = (1.651868/365)/2,

    # sigma_vacc_2d = 1/(4*365),
    # waiting time in vaccinated state ~ Erlang(shape=2,rate=0.5030361/365) based on the fitting of Erlang dist to log-normal with median of 1.4 and dispersion of 1.98
    # refer to the blog post: https://www.jonghoonk.com/posts/vacc_waning_nls_erlang/
    # estimated rate is divided by 2 as the rate in the ODE is implemented by multiplying with 2
    # sigma_vacc_2d_u5 = (0.5030361/365)/2,
    # sigma_vacc_2d = (0.4960427/365)/2,
    # sigma_vacc_2d_u5 = 0,
    # sigma_vacc_2d = 0,
    # vaccination-campaign associated
    vacc_1d_cov = 0.8,
    # vacc_2d_cov = 0.8,
    campaign_1d_start = 1_000.0,
    # campaign_2d_start = 1_000.0,
    campaign_1d_dur = 7.0, # duration of vaccination campaign in days
    # campaign_2d_dur = 7.0, # delay_until_2nd_campaign = 14 # 30 days of delay between the 1st and the 2nd campaign
    # vacc_1d_eff_1 = 0.30, # younger age group
    # vacc_1d_eff_2 = 0.64, # 
    vacc_1d_eff_1 = 0.2956431,
    vacc_1d_eff_2 = 0.6392458, # mean of the 200 samples used in the static model -- just to be consistent
    # vacc_2d_eff_1 = 0.36, # younger age group
    # vacc_2d_eff_2 = 0.76, # 
    vacc_1d_immunity_rate = 1 / 14, # vaccine-induced immunity develops after 14d on average after the first dose, δ
    # vacc_2d_immunity_rate = 1 / 14, # vaccine-induced immunity develops after 7d on average after the second dose
    # case threshold over which intervention will be implemented
    vacc_1d_times = LVector(start=0.0, stop=0.0),
    # vacc_2d_times = LVector(start=0.0, stop=0.0),
    vacc_rates = LVector(vacc_1d_rate=0.0),
    # vacc_rates = LVector(vacc_1d_rate=0.0, vacc_2d_rate=0.0), # array instead of tuple is used such that vacc_*d_rate can be modified externally
    case_threshold = 10.0,
    alpha = 0.0, # proportional reduction in R0
    # case_track_window = 7.0, 
    preemptive_vacc = false,
    callback = nothing,
    ode = nothing,
    prob_ode = nothing,
    ode_solver = AutoTsit5(Rosenbrock23())
  )
  if !isnothing(p)
    param_vector = merge(param_vector, p)
  end
  return param_vector
end

function initialize_params_novacc(p = nothing)
  param_vector = (
    u0 = initialize_u0_novacc(),
    s0 = 1.0, 
    i0 = 0.01,
    R0 = 3.0,
    n0 = 1.0, # effective proportion of population (population at risk = population * n0)
    population = 1.0,
    prop_detection = 1.0,
    # baseline population size is critical for the outbreak size
    # and the currently is the the population size of the admin in which the
    # outbreak was reported. Actual population is likely to be a fraction
    # prop_eff_pop = 1
    # time window over which the number of cases is tracked (i.e., weekly reported)
    # obs_length refers the duration during which data are available in days
    # this is updated for each data set
    tend = 100.0,
    report_freq = 1.0,
    obs_length = 365.0, # 20 weeks
    # this refers to the total simulation days and has to be larger than or equal
    # to the obs_length
    # tau = 0.01, # time step size for numerical integration
    epsilon = 1/1.4, # mean latent period = 1/epsilon
    gamma = 1/2, # mean infectious period = 1/gamma
    fA = 0.5, # fraction of asymptomatic state
    bA = 0.05, # relative infectiousness of asymptomatic state
    kappa = 575.0, # excretion rate cells per person per day
    # prop_children = 0.193, #
    prop_u5 = 0.193, #
    
  # exponent used to model sub-exponential growth (0 < expon < 1) (foi=I^{expon}*S/N)
    expon = 1.0, # exponential growth when expo = 1
    xi = 1/21, # mean decay rate of Vibrio cholerae
    K = 10000.0, # half-infective bacteria dose (10,000 cells/ml)
    R0W = 0.0, #
    sigma = 1/(4*365), # 1/sigma = mean duration of natural immunity
    # 1/sigma_vacc_1d = mean duration of OCV-induced immunity (1st dose)
    sigma_vacc_1d = 1/(2*365),
    # 1/sigma_vacc_2d = mean duration of OCV-induced immunity (2nd dose)
    sigma_vacc_2d = 1/(4*365),
    # vaccination-campaign associated
    vacc_1d_cov = 0.8,
    vacc_2d_cov = 0.8,
    campaign_1d_start = 1_000.0,
    campaign_2d_start = 1_000.0,
    campaign_1d_dur = 7.0, # duration of vaccination campaign in days
    campaign_2d_dur = 7.0, # delay_until_2nd_campaign = 14 # 30 days of delay between the 1st and the 2nd campaign
    vacc_1d_eff_1 = 0.3, # younger age group
    vacc_1d_eff_2 = 0.6, # 
    vacc_2d_eff_1 = 0.3, # younger age group
    vacc_2d_eff_2 = 0.6, # 
    vacc_1d_immunity_rate = 1/14, # vaccine-induced immunity develops after 14d on average after the first dose
    vacc_2d_immunity_rate = 1/7, # vaccine-induced immunity develops after 7d on average after the second dose
    # case threshold over which intervention will be implemented
    vacc_1d_times = LVector(start=0.0, stop=0.0),
    vacc_2d_times = LVector(start=0.0, stop=0.0),
    vacc_rates = LVector(vacc_1d_rate=0.0, vacc_2d_rate=0.0), # array instead of tuple is used such that vacc_*d_rate can be modified externally
    case_threshold = 10.0,
    alpha = 0.0, # proportional reduction in R0
    case_track_window = 7.0, 
    callback = nothing,
    ode = nothing,
    prob_ode = nothing,
    ode_solver = AutoTsit5(Rosenbrock23())
  )
  if !isnothing(p)
    param_vector = merge(param_vector, p)
  end
  return param_vector
end

