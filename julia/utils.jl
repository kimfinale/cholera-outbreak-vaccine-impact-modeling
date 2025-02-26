using Dates
using Printf
using Optimization
using OptimizationBBO
using DifferentialEquations
using Distributions
using LabelledArrays
using NamedTupleTools
using NLsolve

expit(x) = 1/(1+exp(-x));
logit(x) = x < 1.0 && x > 0.0 ? log(x/(1-x)) : error("x must be betwen 0 and 1");
# get parameters r, and p for the negative binomial distribution based on the mean and variance
get_nb_r(m,v) = v > m ? m*m/(v-m) : error("variance v must be larger than the mean m");
get_nb_p(m,r) = r/(m + r)

function final_epidemic_size(R0)
    function f(x) 
        [x[1] - 1 + exp(-R0*x[1])]
    end
    sol = nlsolve(f, [0.5])
    sol.zero
end

function printf_sol(sol)
    @printf("s0 = %.4f, i0 = %.4f, R0 = %.4f\n", expit(sol[1]), expit(sol[2]), exp(sol[3]));
end

function extract_xhat(x)
    p = (s0 = expit(x[1]), i0 = expit(x[2]), R0 = exp(x[3]));
    if length(x) > 3   
        p = merge(p, (n0 = expit(x[4]),));
    end
    if length(x) > 4   
        p = merge(p, (expon = expit(x[5]),));
    end
    return p   
end

function fit_model(p)
   
    prob_opt = OptimizationProblem(p.loss, p.inits.x0, p, lb=p.inits.lower, ub=p.inits.upper);
    sol = solve(prob_opt, BBO_adaptive_de_rand_1_bin_radiuslimited(), NumDimensions=4, PopulationSize=200,
        MaxFuncEvals=20_000);
    return sol
end

function nll_3p(x, p)
    pnew = extract_xhat(x);
    p = merge(p, pnew); 
    p = adjust_pop_dist!(p);
    prob = remake(p.prob_ode; u0=p.u0, p=p);  
    # @show prob
    sol = solve(prob, p.ode_solver, saveat = Int(p.report_freq)); 
    # @show sol
    X = [sol[i].CI_1 + sol[i].CI_2 for i in 2:length(sol.t)] - [sol[i].CI_1 + sol[i].CI_2 for i in 1:(length(sol.t)-1)];
    # make sure all elements of X are positive to use the Poisson
    for i in eachindex(X)
        if X[i] < 1e-6
            X[i] = 1e-6
        end
    end
    - sum(logpdf.(Poisson.(X), p.data));
end

function nll_4p(x, p)
    pnew = (s0 = expit(x[1]), i0 = expit(x[2]), R0 = exp(x[3]), n0 = expit(x[4]));
    p = merge(p, pnew);
    p = adjust_pop_dist!(p);
    prob = ODEProblem(p.ode, p.u0, (0.0, float(p.tend)), p); 
    # @show prob
    sol = solve(prob, p.ode_solver, saveat = Int(p.report_freq)); 
    # @show sol
    X = [sol[i].CI_1 + sol[i].CI_2 for i in 2:length(sol.t)] - [sol[i].CI_1 + sol[i].CI_2 for i in 1:(length(sol.t)-1)];
    # make sure all elements of X are positive to use the Poisson
    for i in eachindex(X)
        if X[i] < 1e-6
            X[i] = 1e-6
        end
    end
    - sum(logpdf.(Poisson.(X), p.data)); # negative log likelihood for Poisson distribution
end

function nll_4p_NB(x, p)
    pnew = (s0 = expit(x[1]), i0 = expit(x[2]), R0 = exp(x[3]), n0 = expit(x[4]));
    p = merge(p, pnew);
    p = adjust_pop_dist!(p);
    prob = ODEProblem(p.ode, p.u0, (0.0, float(p.tend)), p); 
    # @show prob
    sol = solve(prob, p.ode_solver, saveat = Int(p.report_freq)); 
    # @show sol
    X = [sol[i].CI_1 + sol[i].CI_2 for i in 2:length(sol.t)] - [sol[i].CI_1 + sol[i].CI_2 for i in 1:(length(sol.t)-1)];
    # make sure all elements of X are positive to use the Poisson
    for i in eachindex(X)
        if X[i] < 1e-6
            X[i] = 1e-6
        end
    end
    ps = get_nb_p.(X, p.NB_size); # p parameter for Negative Binomial distribution, NB(r,p)
    - sum(logpdf.(NegativeBinomial.(p.NB_size, ps), p.data));
end

function nll_5p(x, p)
    # pnew = extract_xhat(x);
    pnew = (s0 = expit(x[1]), i0 = expit(x[2]), R0 = exp(x[3]), n0 = expit(x[4]), expon = expit(x[5]));
    p = merge(p, pnew);
    p = adjust_pop_dist!(p);
    prob = ODEProblem(p.ode, p.u0, (0.0, float(p.tend)), p); 
    # @show prob
    sol = solve(prob, p.ode_solver, saveat = Int(p.report_freq)); 
    # @show sol
    X = [sol[i].CI_1 + sol[i].CI_2 for i in 2:length(sol.t)] - [sol[i].CI_1 + sol[i].CI_2 for i in 1:(length(sol.t)-1)];
    # make sure all elements of X are positive to use the Poisson
    for i in eachindex(X)
        if X[i] < 1e-6
            X[i] = 1e-6
        end
    end
    - sum(logpdf.(Poisson.(X), p.data));
end

# run the model based on the solution
function run_model(p)
    p = adjust_pop_dist!(p);
    prob = ODEProblem(p.ode, p.u0, (0.0, float(p.tend)), p);
    sol = solve(prob, p.ode_solver, saveat = p.report_freq); 
    X = [sol[i].CI_1 + sol[i].CI_2 for i in 2:length(sol.t)] - [sol[i].CI_1 + sol[i].CI_2 for i in 1:(length(sol.t)-1)];
    # make sure all elements of X are positive to use the Poisson
    # for i in eachindex(X)
    #     if X[i] < 1e-6
    #         X[i] = 1e-6
    #     end
    # end
    return X
end



# run the model based on the solution
function run_model_vacc(p)
    p = set_vaccination_callback(p);
    p = adjust_pop_dist!(p);
    prob = ODEProblem(p.ode, p.u0, (0.0, float(p.tend)), p);
    sol = solve(prob, p.ode_solver, callback=p.callback, 
        tstops = p.vacc_1d_times, saveat=p.report_freq);
    X = [sol[i].CI_1 + sol[i].CI_2 for i in 2:length(sol.t)] - 
        [sol[i].CI_1 + sol[i].CI_2 for i in 1:(length(sol.t)-1)];
    # make sure all elements of X are positive to use the Poisson
    for i in eachindex(X)
        if X[i] < 1e-6
            X[i] = 1e-6
        end
    end
    return X
end

function extract_inc_by_vacc_state(p)
    p = adjust_pop_dist!(p);
    prob = ODEProblem(p.ode, p.u0, (0.0, float(p.tend)), p);
    sol = solve(prob, p.ode_solver, saveat = p.report_freq);
    X_novacc = [sol[i].UCI_1 + sol[i].UCI_2 for i in 2:length(sol.t)] - [sol[i].UCI_1 + sol[i].UCI_2 for i in 1:(length(sol.t)-1)];
    X_vacc = [sol[i].VCI_1 + sol[i].VCI_2 for i in 2:length(sol.t)] - [sol[i].VCI_1 + sol[i].VCI_2 for i in 1:(length(sol.t)-1)];
    # make sure all elements of X are positive to use the Poisson
    # for i in eachindex(X_novacc)
    #     if X_novacc[i] < 1e-6
    #         X_novacc[i] = 1e-6
    #     end
    # end
    # for i in eachindex(X_vacc)
    #     if X_vacc[i] < 1e-6
    #         X_vacc[i] = 1e-6
    #     end
    # end  
    
    X = (inc_novacc = X_novacc, inc_vacc = X_vacc);    
    return X
end


function save_plot(p, pathdir)
    plt = plot(p.TL, p.modeled, label="model", linewidth=2, linecolor=:black);
    scatter!(p.TL, p.data, label="data", fillcolor=:red,
        markershape = :circle,
        markersize = 5,
        markercolor = :red,
        markerstrokewidth = 1,
        markerstrokecolor = :red);
    plot!(title=string("ID: ", p.data_id), xlabel = "", ylabel = "Incidence");
    dt = Dates.now();
    tstamp = Dates.format(dt, dateformat"yyyymmdd");
    png(plt, string(pathdir, "id_", p.data_id, "_", tstamp, ".png"));
end

function plot_incidence(p)
    plt = plot(p.TL, p.modeled, label="model", linewidth=2, linecolor=:black);
    scatter!(p.TL, p.data, label="data", fillcolor=:red,
        markershape = :circle,
        markersize = 5,
        markercolor = :red,
        markerstrokewidth = 1,
        markerstrokecolor = :red);
    plot!(title=string("ID=", p.data_id, " ", p.location), xlabel = "time", ylabel = "Incidence");
    return plt
end


function set_vaccination_callback(p)
    tstop1 = [p.vacc_1d_times.start]
    tstop2 = [p.vacc_1d_times.stop]
    rate = -log(1-p.vacc_1d_cov)/p.campaign_1d_dur

    condition1(u, t, integrator) = t in tstop1
    condition2(u, t, integrator) = t in tstop2
    affect1!(integrator) = integrator.p.vacc_rates.vacc_1d_rate = rate
    affect2!(integrator) = integrator.p.vacc_rates.vacc_1d_rate = 0.0;

    save_positions = (false, false)
    cb = DiscreteCallback(condition1, affect1!, save_positions = save_positions)
    save_positions = (false, false)
    cb2 = DiscreteCallback(condition2, affect2!, save_positions = save_positions)
    cbs = CallbackSet(cb, cb2)
    p = merge(p, (callback = cbs,));
    return p
end

# Vaccine impact metrics
# percent reduction case reduced
function(x1, x2) #x1 and x2 are cumulative number of cases under baseline and vaccine scenarios
    100 * (xl[end] - x2[end]) / x1[end]
end
