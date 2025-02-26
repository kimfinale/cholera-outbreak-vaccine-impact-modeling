using Printf
function seiarw_2ag_erlang_vacc_two_rounds!(du, u, p, t)
    # ================== LOCAL VARIABLESS ==================================
    # model parameters
    prop_detection = p.prop_detection;
    epsilon = p.epsilon; # 1 / latent period
    kappa = p.kappa; # excretion rate
    xi = p.xi; # decay rate
    K = p.K; # half-infective dose (eg, 10,000 doses/ml)
    gamma = p.gamma; # 1 / recovery period
    sigma = p.sigma; # 1 / duration of natural immunity
    sigma_V = p.sigma_vacc_1d; # 1 / duration of OCV-induced immunity (one dose)
    sigma_T = p.sigma_vacc_2d; # 1 / duration of OCV-induced immunity (two doses)
    # 1 / rate from pre-symptomatic (P) state to infectious (I) states
    fA = p.fA; # fraction asymptomatic
    bA = p.bA; # relative infectivity of A to I
    R0 = p.R0;
    R0W = p.R0W; # transmission rate arising from water
    expon = p.expon; # exponent (0< & <1) to control the exponential-ness of the FOI
    vacc_1d_eff_1 = p.vacc_1d_eff_1; # vaccine efficacy for the younger age group (< 5 yo)
    vacc_1d_eff_2 = p.vacc_1d_eff_2; # vaccine efficacy for the older age group
    vacc_2d_eff_1 = p.vacc_2d_eff_1; # vaccine efficacy for the younger age group (< 5 yo) for the second dose
    vacc_2d_eff_2 = p.vacc_2d_eff_2; # vaccine efficacy for the older age group for the second dose
    vacc_1d_immunity_rate = p.vacc_1d_immunity_rate; # delay for the first dose for the immunity to arise
    vacc_2d_immunity_rate = p.vacc_2d_immunity_rate; # delay for the second dose for the immunity to arise
    # the following vaccination rates are updated based on the callbacks during integration
    vacc_1d_rate = p.vacc_rates.vacc_1d_rate; # vaccination rate for the first round
    vacc_2d_rate = p.vacc_rates.vacc_2d_rate; # vaccination rate for the second round
    # transmission rate based on R0    
    β = R0 / ((bA*fA + (1-fA))/gamma); # R0 = β*(bA*fA+(1-fA))/gamma
    
    # total population size 
    N_1 = u.S_1 + u.E1_1 + u.I1_1 + u.A1_1 + u.R1_1 + u.E2_1 + u.I2_1 + u.A2_1 + u.R2_1 + u.RS_1 +
          u.V1_1 + u.V2_1 + u.V3_1 + u.V4_1 + u.VE1_1 + u.VI1_1 + u.VA1_1 + u.VR1_1 + u.VE2_1 + u.VI2_1 + u.VA2_1 + u.VR2_1 +
          u.T1_1 + u.T2_1 + u.T3_1 + u.T4_1 + u.TE1_1 + u.TI1_1 + u.TA1_1 + u.TR1_1 + u.TE2_1 + u.TI2_1 + u.TA2_1 + u.TR2_1;
    N_2 = u.S_2 + u.E1_2 + u.I1_2 + u.A1_2 + u.R1_2 + u.E2_2 + u.I2_2 + u.A2_2 + u.R2_2 + u.RS_2 +
          u.V1_2 + u.V2_2 + u.V3_2 + u.V4_2 + u.VE1_2 + u.VI1_2 + u.VA1_2 + u.VR1_2 + u.VE2_2 + u.VI2_2 + u.VA2_2 + u.VR2_2 +
          u.T1_2 + u.T2_2 + u.T3_2 + u.T4_2 + u.TE1_2 + u.TI1_2 + u.TA1_2 + u.TR1_2 + u.TE2_2 + u.TI2_2 + u.TA2_2 + u.TR2_2;
    N = N_1 + N_2;

    # infectious population
    isum_1 = u.I1_1 + u.I2_1 + u.VI1_1 + u.VI2_1 + u.TI1_1 + u.TI2_1;
    isum_2 = u.I1_2 + u.I2_2 + u.VI1_2 + u.VI2_2 + u.TI1_2 + u.TI2_2;
    isum = isum_1 + isum_2;
    asum_1 = u.A1_1 + u.A2_1 + u.VA1_1 + u.VA2_1 + u.TA1_1 + u.TA2_1;
    asum_2 = u.A1_2 + u.A2_2 + u.VA1_2 + u.VA2_2 + u.TA1_2 + u.TA2_2;
    asum = asum_1 + asum_2;
 
    foi = β * (isum + bA * asum) / N;

    if isum > 0 && asum > 0
      foi = β * (isum^expon + bA*asum^expon) / N;   
    end
    # @show p.vacc_rate
    # @printf("t = %.4f, vacc rate = %.4f\n", t, p.vacc_rate.first);
    
    # ================== COMPARTMENTAL FLOWS ==================================
    # RS are susceptible with prior infection or vaccination. They have the same susceptibility and infectvitiy as S
    # However, RS will have increased immunity upon vaccination (this will be the second or booster dose) compared to S.
    # One remaining question: how do I set the distribution of RS and S in the beginning of the outbreak?
    
    ## infection ----------------------------------------------
    S_1toE1_1 = u.S_1 * foi;
    RS_1toE1_1 = u.RS_1 * foi; # RS has the same susceptibility and infectivity as S 
    V1_1toVE1_1 = u.V1_1 * foi;
    V2_1toVE1_1 = u.V2_1 * foi;
    V3_1toVE1_1 = u.V3_1 * foi * (1 - vacc_1d_eff_1);
    V4_1toVE1_1 = u.V4_1 * foi * (1 - vacc_1d_eff_1);
    T1_1toTE1_1 = u.T1_1 * foi;
    T2_1toTE1_1 = u.T2_1 * foi;
    T3_1toTE1_1 = u.T3_1 * foi * (1 - vacc_2d_eff_1);
    T4_1toTE1_1 = u.T4_1 * foi * (1 - vacc_2d_eff_1);

    # older age group
    S_2toE1_2 = u.S_2 * foi;
    RS_2toE1_2 = u.RS_2 * foi;
    V1_2toVE1_2 = u.V1_2 * foi;
    V2_2toVE1_2 = u.V2_2 * foi;
    V3_2toVE1_2 = u.V3_2 * foi * (1 - vacc_1d_eff_2);
    V4_2toVE1_2 = u.V4_2 * foi * (1 - vacc_1d_eff_2);
    T1_2toTE1_2 = u.T1_2 * foi;
    T2_2toTE1_2 = u.T2_2 * foi;
    T3_2toTE1_2 = u.T3_2 * foi * (1 - vacc_2d_eff_2);
    T4_2toTE1_2 = u.T4_2 * foi * (1 - vacc_2d_eff_2);

    ## disease progression ----------------------------------------------
    E1_1toE2_1 = u.E1_1 * 2 * epsilon;
    E2_1toI1_1 = u.E2_1 * (1-fA) * 2 * epsilon;
    E2_1toA1_1 = u.E2_1 * fA * 2 * epsilon;
    I1_1toI2_1 = u.I1_1 * 2 * gamma;
    I2_1toR1_1 = u.I2_1 * 2 * gamma;
    A1_1toA2_1 = u.A1_1 * 2 * gamma;
    A2_1toR1_1 = u.A2_1 * 2 * gamma;
    R1_1toR2_1 = u.R1_1 * 2 * sigma;
    R2_1toRS_1 = u.R2_1 * 2 * sigma; # RS to track the past history of infection. A subsequent infection will induce higher immunity (i.e., two-dose efficacy)
    # disease progression for the vaccinated (one-dose)
    VE1_1toVE2_1 = u.VE1_1 * 2 * epsilon;
    VE2_1toVI1_1 = u.VE2_1 * (1-fA) * 2 * epsilon;
    VE2_1toVA1_1 = u.VE2_1 * fA * 2 * epsilon;
    VI1_1toVI2_1 = u.VI1_1 * 2 * gamma;
    VI2_1toVR1_1 = u.VI2_1 * 2 * gamma;
    VA1_1toVA2_1 = u.VA1_1 * 2 * gamma;
    VA2_1toVR1_1 = u.VA2_1 * 2 * gamma;
    VR1_1toVR2_1 = u.VR1_1 * 2 * sigma;
    VR2_1toRS_1 = u.VR2_1 * 2 * sigma; # RS to track vaccination history. Like natural infection, a subsequent vaccination will induce a higher immunity (i.e., two-dose efficacy) 

    # disease progression for the vaccinated for the second time (two-dose)
    TE1_1toTE2_1 = u.TE1_1 * 2 * epsilon;
    TE2_1toTI1_1 = u.TE2_1 * (1-fA) * 2 * epsilon;
    TE2_1toTA1_1 = u.TE2_1 * fA * 2 * epsilon;
    TI1_1toTI2_1 = u.TI1_1 * 2 * gamma;
    TI2_1toTR1_1 = u.TI2_1 * 2 * gamma;
    TA1_1toTA2_1 = u.TA1_1 * 2 * gamma;
    TA2_1toTR1_1 = u.TA2_1 * 2 * gamma;
    TR1_1toTR2_1 = u.TR1_1 * 2 * sigma;
    TR2_1toRS_1 = u.TR2_1 * 2 * sigma; # RS to track vaccination history. Like natural infection, a subsequent vaccination will induce a higher immunity (i.e., two-dose efficacy) 

    # older age group
    E1_2toE2_2 = u.E1_2 * 2 * epsilon;
    E2_2toI1_2 = u.E2_2 * (1-fA) * 2 * epsilon;
    E2_2toA1_2 = u.E2_2 * fA * 2 * epsilon;
    I1_2toI2_2 = u.I1_2 * 2 * gamma;
    A1_2toA2_2 = u.A1_2 * 2 * gamma;
    I2_2toR1_2 = u.I2_2 * 2 * gamma;
    A2_2toR1_2 = u.A2_2 * 2 * gamma;
    R1_2toR2_2 = u.R1_2 * 2 * sigma;
    R2_2toRS_2 = u.R2_2 * 2 * sigma;
    # disease progression for the vaccinated (one-dose)
    VE1_2toVE2_2 = u.VE1_2 * 2 * epsilon;
    VE2_2toVI1_2 = u.VE2_2 * (1-fA) * 2 * epsilon;
    VE2_2toVA1_2 = u.VE2_2 * fA * 2 * epsilon;
    VI1_2toVI2_2 = u.VI1_2 * 2 * gamma;
    VA1_2toVA2_2 = u.VA1_2 * 2 * gamma;
    VI2_2toVR1_2 = u.VI2_2 * 2 * gamma;
    VA2_2toVR1_2 = u.VA2_2 * 2 * gamma;
    VR1_2toVR2_2 = u.VR1_2 * 2 * sigma; # not sigma_V since infected
    VR2_2toRS_2 = u.VR2_2 * 2 * sigma;
    # disease progression for the vaccinated for the second time (two-dose)
    TE1_2toTE2_2 = u.TE1_2 * 2 * epsilon;
    TE2_2toTI1_2 = u.TE2_2 * (1-fA) * 2 * epsilon;
    TE2_2toTA1_2 = u.TE2_2 * fA * 2 * epsilon;
    TI1_2toTI2_2 = u.TI1_2 * 2 * gamma;
    TA1_2toTA2_2 = u.TA1_2 * 2 * gamma;
    TI2_2toTR1_2 = u.TI2_2 * 2 * gamma;
    TA2_2toTR1_2 = u.TA2_2 * 2 * gamma;
    TR1_2toTR2_2 = u.TR1_2 * 2 * sigma; # not sigma_T since infected
    TR2_2toRS_2 = u.TR2_2 * 2 * sigma;

    # vaccination ----------------------------------------------------
    # R may receive vaccine but the status does not chage
    # I does not receive vaccine because of symptoms
    # the number of vaccine doses need to account for the population excluding those who are in the I state
    S_1toV1_1 = u.S_1 * vacc_1d_rate; 
    E1_1toVE1_1 = u.E1_1 * vacc_1d_rate;
    E2_1toVE2_1 = u.E2_1 * vacc_1d_rate;
    A1_1toVA1_1 = u.A1_1 * vacc_1d_rate;
    A2_1toVA2_1 = u.A2_1 * vacc_1d_rate;

    RS_1toT1_1 = u.RS_1 * vacc_2d_rate; 
    V1_1toT1_1 = u.V1_1 * vacc_2d_rate;
    V2_1toT1_1 = u.V2_1 * vacc_2d_rate; # V1 and V2 go to T1. Second dose resets the first dose if immunity has not developed yet.
    # however, this is very small fraction in reality...
    V3_1toT3_1 = u.V3_1 * vacc_2d_rate; # again, V3 and V4 go to T3
    V4_1toT3_1 = u.V4_1 * vacc_2d_rate;
    VE1_1toTE1_1 = u.VE1_1 * vacc_2d_rate;
    VE2_1toTE2_1 = u.VE2_1 * vacc_2d_rate;
    VA1_1toTA1_1 = u.VA1_1 * vacc_2d_rate;
    VA2_1toTA2_1 = u.VA2_1 * vacc_2d_rate;

    # gaining vaccine-derived immunity
    V1_1toV2_1 = u.V1_1 * 2 * vacc_1d_immunity_rate; # delay before immunity development
    V2_1toV3_1 = u.V2_1 * 2 * vacc_1d_immunity_rate; # before immunity development
    T1_1toT2_1 = u.T1_1 * 2 * vacc_2d_immunity_rate; # before immunity development
    T2_1toT3_1 = u.T2_1 * 2 * vacc_2d_immunity_rate; # before immunity development

    # losing vaccine-derived immunity
    V3_1toV4_1 = u.V3_1 * 2 * sigma_V; # partially protected, waning immunity with the rate, sigma_V 
    V4_1toRS_1 = u.V4_1 * 2 * sigma_V; # partially protected
    T3_1toT4_1 = u.T3_1 * 2 * sigma_T; # partially protected, waning immunity with the rate, sigma_T 
    T4_1toRS_1 = u.T4_1 * 2 * sigma_T; # partially protected 
    
    # older age group
    S_2toV1_2 = u.S_2 * vacc_1d_rate;
    E1_2toVE1_2 = u.E1_2 * vacc_1d_rate;
    E2_2toVE2_2 = u.E2_2 * vacc_1d_rate;
    A1_2toVA1_2 = u.A1_2 * vacc_1d_rate;
    A2_2toVA2_2 = u.A2_2 * vacc_1d_rate;
    
    RS_2toT1_2 = u.RS_2 * vacc_2d_rate;
    V1_2toT1_2 = u.V1_2 * vacc_2d_rate;
    V2_2toT1_2 = u.V2_2 * vacc_2d_rate; # V1 and V2 go to T1. Second dose resets the first dose if immunity has not developed yet.
    # however, this is very small fraction in reality...
    V3_2toT3_2 = u.V3_2 * vacc_2d_rate; # again, V3 and V4 go to T3
    V4_2toT3_2 = u.V4_2 * vacc_2d_rate;

    VE1_2toTE1_2 = u.VE1_2 * vacc_2d_rate;
    VE2_2toTE2_2 = u.VE2_2 * vacc_2d_rate;
    VA1_2toTA1_2 = u.VA1_2 * vacc_2d_rate;
    VA2_2toTA2_2 = u.VA2_2 * vacc_2d_rate;

    # gaining vaccine-derived immunity
    V1_2toV2_2 = u.V1_2 * 2 * vacc_1d_immunity_rate; # before immunity development
    V2_2toV3_2 = u.V2_2 * 2 * vacc_1d_immunity_rate; # before immunity development
    T1_2toT2_2 = u.T1_2 * 2 * vacc_2d_immunity_rate; # before immunity development
    T2_2toT3_2 = u.T2_2 * 2 * vacc_2d_immunity_rate; # before immunity development

    # losing vaccine-derived immunity
    V3_2toV4_2 = u.V3_2 * 2 * sigma_V; # partially protected, waning immunity with the rate, sigma 
    V4_2toRS_2 = u.V4_2 * 2 * sigma_V; # partially protected
    T3_2toT4_2 = u.T3_2 * 2 * sigma_T; # partially protected, waning immunity with the rate, sigma 
    T4_2toRS_2 = u.T4_2 * 2 * sigma_T; # partially protected 
    
    # ================== DIFFERENTIAL EQUATIONS ==================================

    du.S_1 = - S_1toE1_1 - S_1toV1_1; # infection and vaccination (1st dose)
    du.E1_1 = + S_1toE1_1 + RS_1toE1_1 - E1_1toE2_1 - E1_1toVE1_1; # infection and vaccination (1st dose)
    du.E2_1 = + E1_1toE2_1 - E2_1toI1_1 - E2_1toA1_1 - E2_1toVE2_1; # disease progression and vaccination
    du.I1_1 = + E2_1toI1_1 - I1_1toI2_1; # disease progression only (symptomatic people do not get vaccinated)
    du.I2_1 = + I1_1toI2_1 - I2_1toR1_1; # disease progresssion 
    du.A1_1 = + E2_1toA1_1 - A1_1toA2_1 - A1_1toVA1_1;
    du.A2_1 = + A1_1toA2_1 - A2_1toR1_1 - A2_1toVA2_1;
    du.R1_1 = + I2_1toR1_1 + A2_1toR1_1 - R1_1toR2_1; # disease progresssion only. Vaccine do not change the states of the recovered people
    du.R2_1 = + R1_1toR2_1 - R2_1toRS_1; # disease progression including immunity waning
    du.RS_1 = + R2_1toRS_1 + VR2_1toRS_1 + V4_1toRS_1 + TR2_1toRS_1 + T4_1toRS_1 - RS_1toE1_1 - RS_1toT1_1; # R2, VR2, V4, TR2, and T4 -> RS

    # vaccinated 
    du.V1_1 = + S_1toV1_1 - V1_1toV2_1 - V1_1toVE1_1 - V1_1toT1_1 ; # vaccination, immunity development, and infection
    du.V2_1 = + V1_1toV2_1 - V2_1toV3_1 - V2_1toVE1_1 - V2_1toT1_1; # immunity development, infection, and vaccination
    du.V3_1 = + V2_1toV3_1 - V3_1toV4_1 - V3_1toVE1_1 - V3_1toT3_1; # immunity waning, infection, and vaccination
    du.V4_1 = + V3_1toV4_1 - V4_1toRS_1 - V4_1toVE1_1 - V4_1toT3_1 ; # immunity waning, infection, and vaccination
    du.VE1_1 = + V1_1toVE1_1 + V2_1toVE1_1 + V3_1toVE1_1 + V4_1toVE1_1 + E1_1toVE1_1 - VE1_1toVE2_1 - VE1_1toTE1_1; # infection, progression, vaccination 
    du.VE2_1 = + VE1_1toVE2_1 - VE2_1toVI1_1 - VE2_1toVA1_1 + E2_1toVE2_1 - VE2_1toTE2_1; # progression, vaccination 
    du.VI1_1 = + VE2_1toVI1_1 - VI1_1toVI2_1;
    du.VI2_1 = + VI1_1toVI2_1 - VI2_1toVR1_1;
    du.VA1_1 = + VE2_1toVA1_1 - VA1_1toVA2_1 + A1_1toVA1_1 - VA1_1toTA1_1; # progression, vaccination 
    du.VA2_1 = + VA1_1toVA2_1 - VA2_1toVR1_1 + A2_1toVA2_1 - VA2_1toTA2_1;
    du.VR1_1 = + VI2_1toVR1_1 + VA2_1toVR1_1 - VR1_1toVR2_1; # R -> VR -> TR vaccination not modeled
    du.VR2_1 = + VR1_1toVR2_1 - VR2_1toRS_1;

    du.T1_1 = + RS_1toT1_1 + V1_1toT1_1 + V2_1toT1_1 - T1_1toT2_1 - T1_1toTE1_1; # vaccination, immunity development, infection
    du.T2_1 = + T1_1toT2_1 - T2_1toT3_1 - T2_1toTE1_1;
    du.T3_1 = + T2_1toT3_1 + V3_1toT3_1 + V4_1toT3_1 - T3_1toT4_1 - T3_1toTE1_1;
    du.T4_1 = + T3_1toT4_1 - T4_1toRS_1 - T4_1toTE1_1;
    du.TE1_1 = + T1_1toTE1_1 + T2_1toTE1_1 + T3_1toTE1_1 + T4_1toTE1_1 + VE1_1toTE1_1 - TE1_1toTE2_1;
    du.TE2_1 = + TE1_1toTE2_1 + VE2_1toTE2_1 - TE2_1toTI1_1 - TE2_1toTA1_1;
    du.TI1_1 = + TE2_1toTI1_1 - TI1_1toTI2_1;
    du.TI2_1 = + TI1_1toTI2_1 - TI2_1toTR1_1;
    du.TA1_1 = + TE2_1toTA1_1 - TA1_1toTA2_1 + VA1_1toTA1_1;
    du.TA2_1 = + TA1_1toTA2_1 - TA2_1toTR1_1 + VA2_1toTA2_1;
    du.TR1_1 = + TI2_1toTR1_1 + TA2_1toTR1_1 - TR1_1toTR2_1;
    du.TR2_1 = + TR1_1toTR2_1 - TR2_1toRS_1;

    du.CE_1 = + S_1toE1_1 + RS_1toE1_1 + V1_1toVE1_1 + V2_1toVE1_1 + V3_1toVE1_1 + V4_1toVE1_1 +
        T1_1toTE1_1 + T2_1toTE1_1 + T3_1toTE1_1 + T4_1toTE1_1; # cumulative infections
    du.CI_1 = + prop_detection * (E2_1toI1_1 + VE2_1toVI1_1 + TE2_1toTI1_1) ; # cumulative reorted cases accounting for detection rate
    
    # older age group
    du.S_2 = - S_2toE1_2 - S_2toV1_2;
    du.E1_2 = + S_2toE1_2 + RS_2toE1_2- E1_2toE2_2 - E1_2toVE1_2 ;
    du.E2_2 = + E1_2toE2_2 - E2_2toI1_2 - E2_2toA1_2 - E2_2toVE2_2;
    du.I1_2 = + E2_2toI1_2 - I1_2toI2_2;
    du.I2_2 = + I1_2toI2_2 - I2_2toR1_2;
    du.A1_2 = + E2_2toA1_2 - A1_2toA2_2 - A1_2toVA1_2;
    du.A2_2 = + A1_2toA2_2 - A2_2toR1_2 - A2_2toVA2_2;
    du.R1_2 = + I2_2toR1_2 + A2_2toR1_2 - R1_2toR2_2;
    du.R2_2 = + R1_2toR2_2 - R2_2toRS_2;
    du.RS_2 = + R2_2toRS_2 + VR2_2toRS_2 + V4_2toRS_2 + TR2_2toRS_2 + T4_2toRS_2 - RS_2toE1_2 - RS_2toT1_2;
 
    # vaccinated 
    du.V1_2 = + S_2toV1_2 - V1_2toV2_2 - V1_2toVE1_2 - V1_2toT1_2;
    du.V2_2 = + V1_2toV2_2 - V2_2toV3_2 - V2_2toVE1_2 - V2_2toT1_2;
    du.V3_2 = + V2_2toV3_2 - V3_2toV4_2 - V3_2toVE1_2 - V3_2toT3_2;
    du.V4_2 = + V3_2toV4_2 - V4_2toRS_2 - V4_2toVE1_2 -V4_2toT3_2;
    du.VE1_2 = + V1_2toVE1_2 + V2_2toVE1_2 + V3_2toVE1_2 + V4_2toVE1_2 + E1_2toVE1_2 - VE1_2toVE2_2 - VE1_2toTE1_2;
    du.VE2_2 = + VE1_2toVE2_2 - VE2_2toVI1_2 + E2_2toVE2_2 - VE2_2toVA1_2 - VE2_2toTE2_2;
    du.VI1_2 = + VE2_2toVI1_2 - VI1_2toVI2_2;
    du.VI2_2 = + VI1_2toVI2_2 - VI2_2toVR1_2;
    du.VA1_2 = + VE2_2toVA1_2 + A1_2toVA1_2 - VA1_2toVA2_2 - VA1_2toTA1_2;
    du.VA2_2 = + VA1_2toVA2_2 + A2_2toVA2_2 - VA2_2toVR1_2 - VA2_2toTA2_2;
    du.VR1_2 = + VI2_2toVR1_2 + VA2_2toVR1_2 - VR1_2toVR2_2;
    du.VR2_2 = + VR1_2toVR2_2 - VR2_2toRS_2;

    du.T1_2 = + RS_2toT1_2 + V1_2toT1_2 + V2_2toT1_2 - T1_2toT2_2 - T1_2toTE1_2 ;
    du.T2_2 = + T1_2toT2_2 - T2_2toT3_2 - T2_2toTE1_2;
    du.T3_2 = + T2_2toT3_2 + V3_2toT3_2 + V4_2toT3_2 - T3_2toT4_2 - T3_2toTE1_2;
    du.T4_2 = + T3_2toT4_2 - T4_2toRS_2 - T4_2toTE1_2;
    du.TE1_2 = + T1_2toTE1_2 + T2_2toTE1_2 + T3_2toTE1_2 + T4_2toTE1_2 + VE1_2toTE1_2 - TE1_2toTE2_2;
    du.TE2_2 = + TE1_2toTE2_2 + VE2_2toTE2_2 - TE2_2toTI1_2  - TE2_2toTA1_2;
    du.TI1_2 = + TE2_2toTI1_2 - TI1_2toTI2_2;
    du.TI2_2 = + TI1_2toTI2_2 - TI2_2toTR1_2;
    du.TA1_2 = + TE2_2toTA1_2 - TA1_2toTA2_2 + VA1_2toTA1_2;
    du.TA2_2 = + TA1_2toTA2_2 - TA2_2toTR1_2 + VA2_2toTA2_2;
    du.TR1_2 = + TI2_2toTR1_2 + TA2_2toTR1_2 - TR1_2toTR2_2;
    du.TR2_2 = + TR1_2toTR2_2 - TR2_2toRS_2;

    du.CE_2 = + S_2toE1_2 + V1_2toVE1_2 + V2_2toVE1_2 + V3_2toVE1_2 + V4_2toVE1_2 +
        T1_2toTE1_2 + T2_2toTE1_2 + T3_2toTE1_2 + T4_2toTE1_2;
    du.CI_2 = + prop_detection * (E2_2toI1_2 + VE2_2toVI1_2 + TE2_2toTI1_2);
end


function seiarw_2ag_erlang_vacc_single_round_old!(du, u, p, t)
    # ================== LOCAL VARIABLESS ==================================
    # model parameters
    prop_detection = p.prop_detection;
    epsilon = p.epsilon; # 1 / latent period
    kappa = p.kappa; # excretion rate
    xi = p.xi; # decay rate
    K = p.K; # half-infective dose (eg, 10,000 doses/ml)
    gamma = p.gamma; # 1 / recovery period
    sigma = p.sigma; # 1 / duration of natural immunity
    sigma_V = p.sigma_vacc_1d; # 1 / duration of OCV-induced immunity (one dose)
    # sigma_T = p.sigma_vacc_2d; # 1 / duration of OCV-induced immunity (two doses)
    # 1 / rate from pre-symptomatic (P) state to infectious (I) states
    fA = p.fA; # fraction asymptomatic
    bA = p.bA; # relative infectivity of A to I
    R0 = p.R0;
    R0W = p.R0W; # transmission rate arising from water
    expon = p.expon; # exponent (0< & <1) to control the exponential-ness of the FOI
    vacc_1d_eff_1 = p.vacc_1d_eff_1; # vaccine efficacy for the younger age group (< 5 yo)
    vacc_1d_eff_2 = p.vacc_1d_eff_2; # vaccine efficacy for the older age group
    vacc_1d_immunity_rate = p.vacc_1d_immunity_rate; # delay for the first dose for the immunity to arise
    # the following vaccination rates are updated based on the callbacks during integration
    vacc_1d_rate = p.vacc_rates.vacc_1d_rate; # vaccination rate for the first round
    # transmission rate based on R0    
    β = R0 / ((bA*fA + (1-fA))/gamma); # R0 = β*(bA*fA+(1-fA))/gamma
    
    # total population size 
    N_1 = u.S_1 + u.E1_1 + u.I1_1 + u.A1_1 + u.R1_1 + u.E2_1 + u.I2_1 + u.A2_1 + u.R2_1 + u.RS_1 +
          u.V1_1 + u.V2_1 + u.V3_1 + u.V4_1 + u.VE1_1 + u.VI1_1 + u.VA1_1 + u.VR1_1 + u.VE2_1 + u.VI2_1 + u.VA2_1 + u.VR2_1;
    N_2 = u.S_2 + u.E1_2 + u.I1_2 + u.A1_2 + u.R1_2 + u.E2_2 + u.I2_2 + u.A2_2 + u.R2_2 + u.RS_2 +
          u.V1_2 + u.V2_2 + u.V3_2 + u.V4_2 + u.VE1_2 + u.VI1_2 + u.VA1_2 + u.VR1_2 + u.VE2_2 + u.VI2_2 + u.VA2_2 + u.VR2_2;
    N = N_1 + N_2;

    # infectious population
    isum_1 = u.I1_1 + u.I2_1 + u.VI1_1 + u.VI2_1;
    isum_2 = u.I1_2 + u.I2_2 + u.VI1_2 + u.VI2_2;
    isum = isum_1 + isum_2;
    asum_1 = u.A1_1 + u.A2_1 + u.VA1_1 + u.VA2_1;
    asum_2 = u.A1_2 + u.A2_2 + u.VA1_2 + u.VA2_2;
    asum = asum_1 + asum_2;
 
    foi = β * (isum + bA * asum) / N;

    if isum > 0 && asum > 0
      foi = β * (isum^expon + bA*asum^expon) / N;   
    end
    # @show p.vacc_rate
    # @printf("t = %.4f, vacc rate = %.4f\n", t, p.vacc_rate.first);
    
    # ================== COMPARTMENTAL FLOWS ==================================
    # RS, representing those who lost immunity after infection or vaccination, are as susceptible and infectious (upon infection) as S. 
    # However, RS will have increased immunity upon vaccination  compared to S because this will be the second or booster dose.
    # One remaining question: how do I set the distribution of RS and S in the beginning of the outbreak?
    # For now, I only model the single-dose vaccination and therefore, putting all in the S compartment will be fine.
    
    ## infection ----------------------------------------------
    S_1toE1_1 = u.S_1 * foi;
    RS_1toE1_1 = u.RS_1 * foi; # RS has the same susceptibility and infectivity as S 
    V1_1toVE1_1 = u.V1_1 * foi; # 3rd and 4th compartments of V have partial immunity
    V2_1toVE1_1 = u.V2_1 * foi;
    V3_1toVE1_1 = u.V3_1 * foi * (1 - vacc_1d_eff_1);
    V4_1toVE1_1 = u.V4_1 * foi * (1 - vacc_1d_eff_1);
    
    ### older age group
    S_2toE1_2 = u.S_2 * foi;
    RS_2toE1_2 = u.RS_2 * foi;
    V1_2toVE1_2 = u.V1_2 * foi;
    V2_2toVE1_2 = u.V2_2 * foi;
    V3_2toVE1_2 = u.V3_2 * foi * (1 - vacc_1d_eff_2);
    V4_2toVE1_2 = u.V4_2 * foi * (1 - vacc_1d_eff_2);
    
    ## disease progression ----------------------------------------------
    E1_1toE2_1 = u.E1_1 * 2 * epsilon;
    E2_1toI1_1 = u.E2_1 * (1-fA) * 2 * epsilon;
    E2_1toA1_1 = u.E2_1 * fA * 2 * epsilon;
    I1_1toI2_1 = u.I1_1 * 2 * gamma;
    I2_1toR1_1 = u.I2_1 * 2 * gamma;
    A1_1toA2_1 = u.A1_1 * 2 * gamma;
    A2_1toR1_1 = u.A2_1 * 2 * gamma;
    R1_1toR2_1 = u.R1_1 * 2 * sigma;
    R2_1toRS_1 = u.R2_1 * 2 * sigma; # RS to track the past history of infection. A subsequent infection will induce higher immunity (i.e., two-dose efficacy)
    # disease progression for the vaccinated (one-dose)
    VE1_1toVE2_1 = u.VE1_1 * 2 * epsilon;
    VE2_1toVI1_1 = u.VE2_1 * (1-fA) * 2 * epsilon;
    VE2_1toVA1_1 = u.VE2_1 * fA * 2 * epsilon;
    VI1_1toVI2_1 = u.VI1_1 * 2 * gamma;
    VI2_1toVR1_1 = u.VI2_1 * 2 * gamma;
    VA1_1toVA2_1 = u.VA1_1 * 2 * gamma;
    VA2_1toVR1_1 = u.VA2_1 * 2 * gamma;
    VR1_1toVR2_1 = u.VR1_1 * 2 * sigma;
    VR2_1toRS_1 = u.VR2_1 * 2 * sigma; 
    # older age group
    E1_2toE2_2 = u.E1_2 * 2 * epsilon;
    E2_2toI1_2 = u.E2_2 * (1-fA) * 2 * epsilon;
    E2_2toA1_2 = u.E2_2 * fA * 2 * epsilon;
    I1_2toI2_2 = u.I1_2 * 2 * gamma;
    A1_2toA2_2 = u.A1_2 * 2 * gamma;
    I2_2toR1_2 = u.I2_2 * 2 * gamma;
    A2_2toR1_2 = u.A2_2 * 2 * gamma;
    R1_2toR2_2 = u.R1_2 * 2 * sigma;
    R2_2toRS_2 = u.R2_2 * 2 * sigma;

    # disease progression for the vaccinated with one dose
    VE1_2toVE2_2 = u.VE1_2 * 2 * epsilon;
    VE2_2toVI1_2 = u.VE2_2 * (1-fA) * 2 * epsilon;
    VE2_2toVA1_2 = u.VE2_2 * fA * 2 * epsilon;
    VI1_2toVI2_2 = u.VI1_2 * 2 * gamma;
    VA1_2toVA2_2 = u.VA1_2 * 2 * gamma;
    VI2_2toVR1_2 = u.VI2_2 * 2 * gamma;
    VA2_2toVR1_2 = u.VA2_2 * 2 * gamma;
    VR1_2toVR2_2 = u.VR1_2 * 2 * sigma; # not sigma_V since infected
    VR2_2toRS_2 = u.VR2_2 * 2 * sigma;

    # vaccination ----------------------------------------------------
    # R may receive vaccine but the status does not chage
    # I does not receive vaccine because of symptoms
    # the number of vaccine doses need to account for the population excluding those who are in the I state
    S_1toV1_1 = u.S_1 * vacc_1d_rate; 
    E1_1toVE1_1 = u.E1_1 * vacc_1d_rate;
    E2_1toVE2_1 = u.E2_1 * vacc_1d_rate;
    A1_1toVA1_1 = u.A1_1 * vacc_1d_rate;
    A2_1toVA2_1 = u.A2_1 * vacc_1d_rate;

    # gaining vaccine-derived immunity
    V1_1toV2_1 = u.V1_1 * 2 * vacc_1d_immunity_rate; # delay before immunity development
    V2_1toV3_1 = u.V2_1 * 2 * vacc_1d_immunity_rate; # before immunity development
   
    # losing vaccine-derived immunity
    V3_1toV4_1 = u.V3_1 * 2 * sigma_V; # partially protected, waning immunity with the rate, sigma_V 
    V4_1toRS_1 = u.V4_1 * 2 * sigma_V; # partially protected
        
    # older age group
    S_2toV1_2 = u.S_2 * vacc_1d_rate;
    E1_2toVE1_2 = u.E1_2 * vacc_1d_rate;
    E2_2toVE2_2 = u.E2_2 * vacc_1d_rate;
    A1_2toVA1_2 = u.A1_2 * vacc_1d_rate;
    A2_2toVA2_2 = u.A2_2 * vacc_1d_rate;
    
    # gaining vaccine-derived immunity
    V1_2toV2_2 = u.V1_2 * 2 * vacc_1d_immunity_rate; # before immunity development
    V2_2toV3_2 = u.V2_2 * 2 * vacc_1d_immunity_rate; # before immunity development

    # losing vaccine-derived immunity
    V3_2toV4_2 = u.V3_2 * 2 * sigma_V; # partially protected, waning immunity with the rate, sigma 
    V4_2toRS_2 = u.V4_2 * 2 * sigma_V; # partially protected
    
    # ================== DIFFERENTIAL EQUATIONS ==================================

    du.S_1 = - S_1toE1_1 - S_1toV1_1; # infection and vaccination (1st dose)
    du.E1_1 = + S_1toE1_1 + RS_1toE1_1 - E1_1toE2_1 - E1_1toVE1_1; # infection and vaccination (1st dose)
    du.E2_1 = + E1_1toE2_1 - E2_1toI1_1 - E2_1toA1_1 - E2_1toVE2_1; # disease progression and vaccination
    du.I1_1 = + E2_1toI1_1 - I1_1toI2_1; # disease progression only (symptomatic people do not get vaccinated)
    du.I2_1 = + I1_1toI2_1 - I2_1toR1_1; # disease progresssion 
    du.A1_1 = + E2_1toA1_1 - A1_1toA2_1 - A1_1toVA1_1;
    du.A2_1 = + A1_1toA2_1 - A2_1toR1_1 - A2_1toVA2_1;
    du.R1_1 = + I2_1toR1_1 + A2_1toR1_1 - R1_1toR2_1; # disease progresssion only. Vaccine do not change the states of the recovered people
    du.R2_1 = + R1_1toR2_1 - R2_1toRS_1; # disease progression including immunity waning
    du.RS_1 = + R2_1toRS_1 + VR2_1toRS_1 + V4_1toRS_1 - RS_1toE1_1; # R2, VR2, V4, TR2, and T4 -> RS

    # vaccinated
    du.V1_1 = + S_1toV1_1 - V1_1toV2_1 - V1_1toVE1_1; # vaccination, immunity development, and infection
    du.V2_1 = + V1_1toV2_1 - V2_1toV3_1 - V2_1toVE1_1; # immunity development, infection, and vaccination
    du.V3_1 = + V2_1toV3_1 - V3_1toV4_1 - V3_1toVE1_1; # immunity waning, infection, and vaccination
    du.V4_1 = + V3_1toV4_1 - V4_1toRS_1 - V4_1toVE1_1; # immunity waning, infection, and vaccination
    du.VE1_1 = + V1_1toVE1_1 + V2_1toVE1_1 + V3_1toVE1_1 + V4_1toVE1_1 + E1_1toVE1_1 - VE1_1toVE2_1; # infection, progression, vaccination 
    du.VE2_1 = + VE1_1toVE2_1 - VE2_1toVI1_1 - VE2_1toVA1_1 + E2_1toVE2_1; # progression, vaccination 
    du.VI1_1 = + VE2_1toVI1_1 - VI1_1toVI2_1;
    du.VI2_1 = + VI1_1toVI2_1 - VI2_1toVR1_1;
    du.VA1_1 = + VE2_1toVA1_1 - VA1_1toVA2_1 + A1_1toVA1_1; # progression, vaccination 
    du.VA2_1 = + VA1_1toVA2_1 - VA2_1toVR1_1 + A2_1toVA2_1;
    du.VR1_1 = + VI2_1toVR1_1 + VA2_1toVR1_1 - VR1_1toVR2_1; # R -> VR
    du.VR2_1 = + VR1_1toVR2_1 - VR2_1toRS_1;

    du.CE_1 = + S_1toE1_1 + RS_1toE1_1 + V1_1toVE1_1 + V2_1toVE1_1 + V3_1toVE1_1 + V4_1toVE1_1; # cumulative infections
    du.CI_1 = + prop_detection * (E2_1toI1_1 + VE2_1toVI1_1); # cumulative reorted cases accounting for detection rate
    
    # older age group
    du.S_2 = - S_2toE1_2 - S_2toV1_2;
    du.E1_2 = + S_2toE1_2 + RS_2toE1_2- E1_2toE2_2 - E1_2toVE1_2 ;
    du.E2_2 = + E1_2toE2_2 - E2_2toI1_2 - E2_2toA1_2 - E2_2toVE2_2;
    du.I1_2 = + E2_2toI1_2 - I1_2toI2_2;
    du.I2_2 = + I1_2toI2_2 - I2_2toR1_2;
    du.A1_2 = + E2_2toA1_2 - A1_2toA2_2 - A1_2toVA1_2;
    du.A2_2 = + A1_2toA2_2 - A2_2toR1_2 - A2_2toVA2_2;
    du.R1_2 = + I2_2toR1_2 + A2_2toR1_2 - R1_2toR2_2;
    du.R2_2 = + R1_2toR2_2 - R2_2toRS_2;
    du.RS_2 = + R2_2toRS_2 + VR2_2toRS_2 + V4_2toRS_2 - RS_2toE1_2;
 
    # vaccinated 
    du.V1_2 = + S_2toV1_2 - V1_2toV2_2 - V1_2toVE1_2;
    du.V2_2 = + V1_2toV2_2 - V2_2toV3_2 - V2_2toVE1_2;
    du.V3_2 = + V2_2toV3_2 - V3_2toV4_2 - V3_2toVE1_2;
    du.V4_2 = + V3_2toV4_2 - V4_2toRS_2 - V4_2toVE1_2;
    du.VE1_2 = + V1_2toVE1_2 + V2_2toVE1_2 + V3_2toVE1_2 + V4_2toVE1_2 + E1_2toVE1_2 - VE1_2toVE2_2;
    du.VE2_2 = + VE1_2toVE2_2 - VE2_2toVI1_2 + E2_2toVE2_2 - VE2_2toVA1_2;
    du.VI1_2 = + VE2_2toVI1_2 - VI1_2toVI2_2;
    du.VI2_2 = + VI1_2toVI2_2 - VI2_2toVR1_2;
    du.VA1_2 = + VE2_2toVA1_2 + A1_2toVA1_2 - VA1_2toVA2_2;
    du.VA2_2 = + VA1_2toVA2_2 + A2_2toVA2_2 - VA2_2toVR1_2;
    du.VR1_2 = + VI2_2toVR1_2 + VA2_2toVR1_2 - VR1_2toVR2_2;
    du.VR2_2 = + VR1_2toVR2_2 - VR2_2toRS_2;

    du.CE_2 = + S_2toE1_2 + V1_2toVE1_2 + V2_2toVE1_2 + V3_2toVE1_2 + V4_2toVE1_2;
    du.CI_2 = + prop_detection * (E2_2toI1_2 + VE2_2toVI1_2);
end

function seiarw_2ag_erlang_vacc_single_round!(du, u, p, t)
  # ================== LOCAL VARIABLESS ==================================
  # model parameters
  prop_detection = p.prop_detection;
  epsilon = p.epsilon; # 1 / latent period
  kappa = p.kappa; # excretion rate
  xi = p.xi; # decay rate
  K = p.K; # half-infective dose (eg, 10,000 doses/ml)
  gamma = p.gamma; # 1 / recovery period
  sigma = p.sigma; # 1 / duration of natural immunity
  sigma_V = p.sigma_vacc_1d; # 1 / duration of OCV-induced immunity (one dose)
  # sigma_T = p.sigma_vacc_2d; # 1 / duration of OCV-induced immunity (two doses)
  # 1 / rate from pre-symptomatic (P) state to infectious (I) states
  fA = p.fA; # fraction asymptomatic
  bA = p.bA; # relative infectivity of A to I
  R0 = p.R0;
  R0W = p.R0W; # transmission rate arising from water
  expon = p.expon; # exponent (0< & <1) to control the exponential-ness of the FOI
  vacc_1d_eff_1 = p.vacc_1d_eff_1; # vaccine efficacy for the younger age group (< 5 yo)
  vacc_1d_eff_2 = p.vacc_1d_eff_2; # vaccine efficacy for the older age group
  vacc_1d_immunity_rate = p.vacc_1d_immunity_rate; # delay for the first dose for the immunity to arise
  # the following vaccination rates are updated based on the callbacks during integration
  vacc_1d_rate = p.vacc_rates.vacc_1d_rate; # vaccination rate for the first round
  # transmission rate based on R0    
  β = R0 / ((bA*fA + (1-fA))/gamma); # R0 = β*(bA*fA+(1-fA))/gamma
  
  # total population size 
  N_1 = u.S_1 + u.E1_1 + u.I1_1 + u.A1_1 + u.R1_1 + u.E2_1 + u.I2_1 + u.A2_1 + u.R2_1 + u.RS_1 +
        u.V1_1 + u.V2_1 + u.V3_1 + u.V4_1 + u.VE1_1 + u.VI1_1 + u.VA1_1 + u.VR1_1 + u.VE2_1 + u.VI2_1 + u.VA2_1 + u.VR2_1;
  N_2 = u.S_2 + u.E1_2 + u.I1_2 + u.A1_2 + u.R1_2 + u.E2_2 + u.I2_2 + u.A2_2 + u.R2_2 + u.RS_2 +
        u.V1_2 + u.V2_2 + u.V3_2 + u.V4_2 + u.VE1_2 + u.VI1_2 + u.VA1_2 + u.VR1_2 + u.VE2_2 + u.VI2_2 + u.VA2_2 + u.VR2_2;
  N = N_1 + N_2;

  # infectious population
  isum_1 = u.I1_1 + u.I2_1 + u.VI1_1 + u.VI2_1;
  isum_2 = u.I1_2 + u.I2_2 + u.VI1_2 + u.VI2_2;
  isum = isum_1 + isum_2;
  asum_1 = u.A1_1 + u.A2_1 + u.VA1_1 + u.VA2_1;
  asum_2 = u.A1_2 + u.A2_2 + u.VA1_2 + u.VA2_2;
  asum = asum_1 + asum_2;

  foi = β * (isum + bA * asum) / N;

  if isum > 0 && asum > 0
    foi = β * (isum^expon + bA*asum^expon) / N;   
  end
  # @show p.vacc_rate
  # @printf("t = %.4f, vacc rate = %.4f\n", t, p.vacc_rate.first);
  
  # ================== COMPARTMENTAL FLOWS ==================================
  # RS, representing those who lost immunity after infection or vaccination, are as susceptible and infectious (upon infection) as S. 
  # However, RS will have increased immunity upon vaccination  compared to S because this will be the second or booster dose.
  # One remaining question: how do I set the distribution of RS and S in the beginning of the outbreak?
  # For now, I only model the single-dose vaccination and therefore, putting all in the S compartment will be fine.
  # Natural immunity does not wane during the period of an outbreak, whereas vaccine-induced immunity might.
  
  ## infection ----------------------------------------------
  S_1toE1_1 = u.S_1 * foi;
  RS_1toE1_1 = u.RS_1 * foi; # RS has the same susceptibility and infectivity as S 
  V1_1toVE1_1 = u.V1_1 * foi; # 3rd and 4th compartments of V have partial immunity
  V2_1toVE1_1 = u.V2_1 * foi;
  V3_1toVE1_1 = u.V3_1 * foi * (1 - vacc_1d_eff_1);
  V4_1toVE1_1 = u.V4_1 * foi * (1 - vacc_1d_eff_1);
  
  ### older age group
  S_2toE1_2 = u.S_2 * foi;
  RS_2toE1_2 = u.RS_2 * foi;
  V1_2toVE1_2 = u.V1_2 * foi;
  V2_2toVE1_2 = u.V2_2 * foi;
  V3_2toVE1_2 = u.V3_2 * foi * (1 - vacc_1d_eff_2);
  V4_2toVE1_2 = u.V4_2 * foi * (1 - vacc_1d_eff_2);
  
  ## disease progression ----------------------------------------------
  E1_1toE2_1 = u.E1_1 * 2 * epsilon;
  E2_1toI1_1 = u.E2_1 * (1-fA) * 2 * epsilon;
  E2_1toA1_1 = u.E2_1 * fA * 2 * epsilon;
  I1_1toI2_1 = u.I1_1 * 2 * gamma;
  I2_1toR1_1 = u.I2_1 * 2 * gamma;
  A1_1toA2_1 = u.A1_1 * 2 * gamma;
  A2_1toR1_1 = u.A2_1 * 2 * gamma;
  R1_1toR2_1 = u.R1_1 * 2 * sigma;
  R2_1toRS_1 = u.R2_1 * 2 * sigma; # RS to track the past history of infection. A subsequent infection will induce higher immunity (i.e., two-dose efficacy)
  # disease progression for the vaccinated (one-dose)
  VE1_1toVE2_1 = u.VE1_1 * 2 * epsilon;
  VE2_1toVI1_1 = u.VE2_1 * (1-fA) * 2 * epsilon;
  VE2_1toVA1_1 = u.VE2_1 * fA * 2 * epsilon;
  VI1_1toVI2_1 = u.VI1_1 * 2 * gamma;
  VI2_1toVR1_1 = u.VI2_1 * 2 * gamma;
  VA1_1toVA2_1 = u.VA1_1 * 2 * gamma;
  VA2_1toVR1_1 = u.VA2_1 * 2 * gamma;
  VR1_1toVR2_1 = u.VR1_1 * 2 * sigma;
  VR2_1toRS_1 = u.VR2_1 * 2 * sigma; 
  # older age group
  E1_2toE2_2 = u.E1_2 * 2 * epsilon;
  E2_2toI1_2 = u.E2_2 * (1-fA) * 2 * epsilon;
  E2_2toA1_2 = u.E2_2 * fA * 2 * epsilon;
  I1_2toI2_2 = u.I1_2 * 2 * gamma;
  A1_2toA2_2 = u.A1_2 * 2 * gamma;
  I2_2toR1_2 = u.I2_2 * 2 * gamma;
  A2_2toR1_2 = u.A2_2 * 2 * gamma;
  R1_2toR2_2 = u.R1_2 * 2 * sigma;
  R2_2toRS_2 = u.R2_2 * 2 * sigma;

  # disease progression for the vaccinated with one dose
  VE1_2toVE2_2 = u.VE1_2 * 2 * epsilon;
  VE2_2toVI1_2 = u.VE2_2 * (1-fA) * 2 * epsilon;
  VE2_2toVA1_2 = u.VE2_2 * fA * 2 * epsilon;
  VI1_2toVI2_2 = u.VI1_2 * 2 * gamma;
  VA1_2toVA2_2 = u.VA1_2 * 2 * gamma;
  VI2_2toVR1_2 = u.VI2_2 * 2 * gamma;
  VA2_2toVR1_2 = u.VA2_2 * 2 * gamma;
  VR1_2toVR2_2 = u.VR1_2 * 2 * sigma; # not sigma_V since infected
  VR2_2toRS_2 = u.VR2_2 * 2 * sigma;

  # vaccination ----------------------------------------------------
  # R may receive vaccine but the status does not chage
  # I does not receive vaccine because of symptoms
  # the number of vaccine doses need to account for the population excluding those who are in the I state
  S_1toV1_1 = u.S_1 * vacc_1d_rate; 
  E1_1toVE1_1 = u.E1_1 * vacc_1d_rate;
  E2_1toVE2_1 = u.E2_1 * vacc_1d_rate;
  A1_1toVA1_1 = u.A1_1 * vacc_1d_rate;
  A2_1toVA2_1 = u.A2_1 * vacc_1d_rate;

  # gaining vaccine-derived immunity
  V1_1toV2_1 = u.V1_1 * 2 * vacc_1d_immunity_rate; # delay before immunity development
  V2_1toV3_1 = u.V2_1 * 2 * vacc_1d_immunity_rate; # before immunity development
 
  # losing vaccine-derived immunity
  V3_1toV4_1 = u.V3_1 * 2 * sigma_V; # partially protected, waning immunity with the rate, sigma_V 
  V4_1toRS_1 = u.V4_1 * 2 * sigma_V; # partially protected
      
  # older age group
  S_2toV1_2 = u.S_2 * vacc_1d_rate;
  E1_2toVE1_2 = u.E1_2 * vacc_1d_rate;
  E2_2toVE2_2 = u.E2_2 * vacc_1d_rate;
  A1_2toVA1_2 = u.A1_2 * vacc_1d_rate;
  A2_2toVA2_2 = u.A2_2 * vacc_1d_rate;
  
  # gaining vaccine-derived immunity
  V1_2toV2_2 = u.V1_2 * 2 * vacc_1d_immunity_rate; # before immunity development
  V2_2toV3_2 = u.V2_2 * 2 * vacc_1d_immunity_rate; # before immunity development

  # losing vaccine-derived immunity
  V3_2toV4_2 = u.V3_2 * 2 * sigma_V; # partially protected, waning immunity with the rate, sigma 
  V4_2toRS_2 = u.V4_2 * 2 * sigma_V; # partially protected
  
  # ================== DIFFERENTIAL EQUATIONS ==================================

  du.S_1 = - S_1toE1_1 - S_1toV1_1; # infection and vaccination (1st dose)
  du.E1_1 = + S_1toE1_1 + RS_1toE1_1 - E1_1toE2_1 - E1_1toVE1_1; # infection and vaccination (1st dose)
  du.E2_1 = + E1_1toE2_1 - E2_1toI1_1 - E2_1toA1_1 - E2_1toVE2_1; # disease progression and vaccination
  du.I1_1 = + E2_1toI1_1 - I1_1toI2_1; # disease progression only (symptomatic people do not get vaccinated)
  du.I2_1 = + I1_1toI2_1 - I2_1toR1_1; # disease progresssion 
  du.A1_1 = + E2_1toA1_1 - A1_1toA2_1 - A1_1toVA1_1;
  du.A2_1 = + A1_1toA2_1 - A2_1toR1_1 - A2_1toVA2_1;
  du.R1_1 = + I2_1toR1_1 + A2_1toR1_1 - R1_1toR2_1; # disease progresssion only. Vaccine do not change the states of the recovered people
  du.R2_1 = + R1_1toR2_1 - R2_1toRS_1; # disease progression including immunity waning
  du.RS_1 = + R2_1toRS_1 + VR2_1toRS_1 + V4_1toRS_1 - RS_1toE1_1; # R2, VR2, V4, TR2, and T4 -> RS

  # vaccinated
  du.V1_1 = + S_1toV1_1 - V1_1toV2_1 - V1_1toVE1_1; # vaccination, immunity development, and infection
  du.V2_1 = + V1_1toV2_1 - V2_1toV3_1 - V2_1toVE1_1; # immunity development, infection, and vaccination
  du.V3_1 = + V2_1toV3_1 - V3_1toV4_1 - V3_1toVE1_1; # immunity waning, infection, and vaccination
  du.V4_1 = + V3_1toV4_1 - V4_1toRS_1 - V4_1toVE1_1; # immunity waning, infection, and vaccination
  du.VE1_1 = + V1_1toVE1_1 + V2_1toVE1_1 + V3_1toVE1_1 + V4_1toVE1_1 + E1_1toVE1_1 - VE1_1toVE2_1; # infection, progression, vaccination 
  du.VE2_1 = + VE1_1toVE2_1 - VE2_1toVI1_1 - VE2_1toVA1_1 + E2_1toVE2_1; # progression, vaccination 
  du.VI1_1 = + VE2_1toVI1_1 - VI1_1toVI2_1;
  du.VI2_1 = + VI1_1toVI2_1 - VI2_1toVR1_1;
  du.VA1_1 = + VE2_1toVA1_1 - VA1_1toVA2_1 + A1_1toVA1_1; # progression, vaccination 
  du.VA2_1 = + VA1_1toVA2_1 - VA2_1toVR1_1 + A2_1toVA2_1;
  du.VR1_1 = + VI2_1toVR1_1 + VA2_1toVR1_1 - VR1_1toVR2_1; # R -> VR
  du.VR2_1 = + VR1_1toVR2_1 - VR2_1toRS_1;

  # infection 
  du.CE_1 = + S_1toE1_1 + RS_1toE1_1 + V1_1toVE1_1 + V2_1toVE1_1 + V3_1toVE1_1 + V4_1toVE1_1; # cumulative infections
  du.VCI_1 = prop_detection * VE2_1toVI1_1 # cumulative incidence in the vaccinated group ignoring waning of vaccine-derived immunity 
  du.UCI_1 = prop_detection * E2_1toI1_1 # cumulative incidence in the unvaccinated group ignoring waning of vaccine-derived immunity 
  du.CI_1 = + prop_detection * (E2_1toI1_1 + VE2_1toVI1_1); # cumulative reorted cases accounting for detection rate
  
  # older age group
  du.S_2 = - S_2toE1_2 - S_2toV1_2;
  du.E1_2 = + S_2toE1_2 + RS_2toE1_2- E1_2toE2_2 - E1_2toVE1_2 ;
  du.E2_2 = + E1_2toE2_2 - E2_2toI1_2 - E2_2toA1_2 - E2_2toVE2_2;
  du.I1_2 = + E2_2toI1_2 - I1_2toI2_2;
  du.I2_2 = + I1_2toI2_2 - I2_2toR1_2;
  du.A1_2 = + E2_2toA1_2 - A1_2toA2_2 - A1_2toVA1_2;
  du.A2_2 = + A1_2toA2_2 - A2_2toR1_2 - A2_2toVA2_2;
  du.R1_2 = + I2_2toR1_2 + A2_2toR1_2 - R1_2toR2_2;
  du.R2_2 = + R1_2toR2_2 - R2_2toRS_2;
  du.RS_2 = + R2_2toRS_2 + VR2_2toRS_2 + V4_2toRS_2 - RS_2toE1_2;

  # vaccinated 
  du.V1_2 = + S_2toV1_2 - V1_2toV2_2 - V1_2toVE1_2;
  du.V2_2 = + V1_2toV2_2 - V2_2toV3_2 - V2_2toVE1_2;
  du.V3_2 = + V2_2toV3_2 - V3_2toV4_2 - V3_2toVE1_2;
  du.V4_2 = + V3_2toV4_2 - V4_2toRS_2 - V4_2toVE1_2;
  du.VE1_2 = + V1_2toVE1_2 + V2_2toVE1_2 + V3_2toVE1_2 + V4_2toVE1_2 + E1_2toVE1_2 - VE1_2toVE2_2;
  du.VE2_2 = + VE1_2toVE2_2 - VE2_2toVI1_2 + E2_2toVE2_2 - VE2_2toVA1_2;
  du.VI1_2 = + VE2_2toVI1_2 - VI1_2toVI2_2;
  du.VI2_2 = + VI1_2toVI2_2 - VI2_2toVR1_2;
  du.VA1_2 = + VE2_2toVA1_2 + A1_2toVA1_2 - VA1_2toVA2_2;
  du.VA2_2 = + VA1_2toVA2_2 + A2_2toVA2_2 - VA2_2toVR1_2;
  du.VR1_2 = + VI2_2toVR1_2 + VA2_2toVR1_2 - VR1_2toVR2_2;
  du.VR2_2 = + VR1_2toVR2_2 - VR2_2toRS_2;

  du.VCI_2 = prop_detection * VE2_2toVI1_2 # cumulative incidence in the vaccinated group ignoring waning of vaccine-derived immunity 
  du.UCI_2 = prop_detection * E2_2toI1_2 # cumulative incidence in the unvaccinated group ignoring waning of vaccine-derived immunity 

  du.CE_2 = + S_2toE1_2 + V1_2toVE1_2 + V2_2toVE1_2 + V3_2toVE1_2 + V4_2toVE1_2;
  du.CI_2 = + prop_detection * (E2_2toI1_2 + VE2_2toVI1_2);
end