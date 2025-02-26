# using Revise
using DifferentialEquations
using Plots
using CSV
using DataFrames
using Random
using LabelledArrays
using Distributions
using NamedTupleTools
using Dates
using Printf
using Optimization
using ForwardDiff
using OptimizationOptimJL
using OptimizationBBO
using JLD2
# load model function, parameter LabelledArrays
path_workspace = "C:\\Users\\jonghoon.kim\\Documents\\CholeraOutbreakModel\\julia"
include(joinpath(path_workspace, "parameters.jl"));
include(joinpath(path_workspace, "seiarw_2ag_erlang_vacc.jl"));
include(joinpath(path_workspace, "utils.jl"));

Random.seed!(42);

dat_ts = DataFrame(CSV.File(joinpath(path_workspace, "data", "outbreak_data_20241121.csv"); header=1, delim=","));
dat_fit = dat_ts;
ids = unique(dat_fit.id_outbreak, dims=1);
ids_df = DataFrame(id_outbreak = ids);
ids_df.data_id = 1:size(ids_df,1);
# just to know the data_id vs. id_outbreak
# CSV.write(string(path_workspace, "\\outputs\\data_id_id_outbreak_20241202.csv"), ids_df);
# n_ids = length(ids);
n_ids = length(unique(ids, dims=1));
dat_fit = innerjoin(dat_fit, ids_df, on = :id_outbreak);

# initial conditions for fitting

x0 = [logit(0.2), logit(0.1), log(2.0), logit(0.5)];
lower = [logit(0.0001), logit(0.0001), log(1.1000), logit(0.0001)];
upper = [logit(0.9999), logit(0.9999), log(30.0000), logit(0.9999)];
inits = LVector(x0=x0, lower=lower, upper=upper);

frac_asymp = [0.1];
for fa in frac_asymp 
    @printf("running fA = %.2f\n", fa);
    df = DataFrame(id_outbreak = "", data_id=zeros(n_ids), fit_id=zeros(n_ids), s0=zeros(n_ids), i0=zeros(n_ids), 
        R0=zeros(n_ids), n0=zeros(n_ids), objective=zeros(n_ids), retcode=zeros(n_ids));     
    fit_params = Vector{Any}(undef, n_ids);
    plotdir = @sprintf("%s\\plots\\4p_fA%03d\\", path_workspace, round(Int, fa*100));
    ispath(plotdir) || mkdir(plotdir)
    outputfilename = @sprintf("%s\\outputs\\fit_4p_fA%03d_", path_workspace, round(Int, fa*100));
    for i in collect(1:10:length(ids))
        @printf("running %d of %d\n", i, length(ids));
        params = initialize_params((fA=fa, loss=nll_4p, ode=seiarw_2ag_erlang_vacc_single_round!, inits=inits));    
        # population size of the admin unit in which the outbreak was observed
        id_pop = dat_fit[dat_fit.id_outbreak .== ids[i], [:pop, :data_id, :id_outbreak]];       
        # update the params.u0 based on the pop
        params = merge(params, (population = id_pop.pop[1],)); 
        params = merge(params, (id_outbreak = id_pop.id_outbreak[1],)); 
        params = merge(params, (data_id = id_pop.data_id[1],)); 
        params = merge(params, (fit_id = i,)); 
        d = dat_ts[dat_ts.id_outbreak .== ids[i], [:s_ch, :week, :tl, :location]];
        params = merge(params, (TL = d.tl,));  
        params = merge(params, (location = d.location[1],));  
        params = merge(params, (report_freq = 7.0,)); # temporal scale is weekly
        y = round.(Int, d.s_ch); # data
        params = merge(params, (data = y,));   
        params = merge(params, (tend = params.report_freq * length(y),));   
       
        try 
            sol = fit_model(params);
        # store results
            # the next two lines can be above the try catch block such that id's are always filled
            df[i, :data_id] = id_pop.data_id[1];
            df[i, :fit_id] = i;

            xhat = extract_xhat(sol);
            params = merge(params, xhat);
            df[i, :s0] = xhat.s0;
            df[i, :i0] = xhat.i0;
            df[i, :R0] = xhat.R0;
            if length(sol.u) > 3
                df[i, :n0] = xhat.n0;
            end    
            df[i, :objective] = sol.objective;
            df[i, :retcode] = Int(sol.retcode); 
            params = merge(params, (sol_optim = sol,));
            x = run_model(params);
            params = merge(params, (modeled = x,));
            fit_params[i] = params;
            save_plot(params, plotdir);
        catch e
            @printf("fitting for i = %d failed\n", i);
            fit_params[i] = nothing;
        end
    end
    
    dt = Dates.now();
    tstamp = Dates.format(dt, dateformat"yyyymmdd\THH");
    save_object(string(outputfilename, tstamp, ".jld2"), fit_params); # rdata, rds
    # jldsave(string(outputfilename, tstamp, ".jld2"); df);
    CSV.write(string(outputfilename, tstamp, ".csv"), df);
end

# fiting from multiple starting points to make sure we arrived at global minimum although the DE algorithm is claimed to be global optimizer. 
# fa asssumed to be 0.5
fa = 0.5
nval = 5
x0s = LinRange.([0.01, 0.0001, 1.1, 0.01], [0.99, 0.9, 20.0, 0.99], nval);

for n in 1:nval 
    x0 = [logit(x0s[1][n]), logit(x0s[2][n]), log(x0s[3][n]), logit(x0s[4][n])];
    inits = LVector(x0=x0, lower=lower, upper=upper);
    @printf("running fA = %.2f\n", fa);
    @printf("x0 = %.2f, %.4f, %.1f, %.2f\n", x0s[1][n], x0s[2][n], x0s[3][n], x0s[4][n]); 
    df = DataFrame(data_id=zeros(n_ids), fit_id=zeros(n_ids), s0=zeros(n_ids), i0=zeros(n_ids), 
        R0=zeros(n_ids), n0=zeros(n_ids), objective=zeros(n_ids), retcode=zeros(n_ids)); 
    fit_params = Vector{Any}(undef, n_ids);
    plotdir = @sprintf("%s\\plots\\4p_init_val_s0%03d\\", path_workspace, round(Int, x0s[1][n]*100));
    ispath(plotdir) || mkdir(plotdir)
    outputfilename = @sprintf("%s\\outputs\\fit_init_val_s0%03d_", path_workspace, round(Int, x0s[1][n]*100));
      for i in eachindex(ids)
        @printf("running %d of %d\n", i, length(ids));
        params = initialize_params((fA=fa, loss=nll_4p, ode=seiarw_2ag_erlang_vacc_two_rounds!, inits=inits));    
        # population size of the admin unit in which the outbreak was observed
        id_pop = dat_fit[dat_fit.ID_outbreak .== ids[i], [:population, :ID]];
        # update the params.u0 based on the pop
        params = merge(params, (population = id_pop.population[1],)); # pop is Vector{Float64}
        params = merge(params, (data_id = id_pop.ID[1],)); 
        params = merge(params, (fit_id = i,)); 
        # update_params!(params, LVector(population = pop[1]));  
        d = dat_ts[dat_ts.ID_outbreak .== ids[i], [:sCh, :temporal_scale, :TL, :location]];
        params = merge(params, (TL = d.TL,));  
        params = merge(params, (location = d.location[1],));  
        # nobs = size(d, 1);
        if d.temporal_scale[1] == "weekly"
            params = merge(params, (report_freq = 7.0,));
        else
            params = merge(params, (report_freq = 1.0,));
        end
        y = round.(Int, d.sCh); # data
        params = merge(params, (data = y,));   
        params = merge(params, (tend = params.report_freq * length(y),));   

        try 
            sol = fit_model(params);
        # store results
            df[i, :data_id] = id_pop.ID[1];
            df[i, :fit_id] = i;
            xhat = extract_xhat(sol);
            params = merge(params, xhat);
            df[i, :s0] = xhat.s0;
            df[i, :i0] = xhat.i0;
            df[i, :R0] = xhat.R0;
            if length(sol.u) > 3
                df[i, :n0] = xhat.n0;
            end    
            df[i, :objective] = sol.objective;
            df[i, :retcode] = Int(sol.retcode); 
            params = merge(params, (sol_optim = sol,));
            x = run_model(params);
            params = merge(params, (modeled = x,));
            fit_params[i] = params;
            save_plot(params, plotdir);
        catch e
            @printf("fitting for i = %d failed\n", i);
        end
    end
    
    dt = Dates.now();
    tstamp = Dates.format(dt, dateformat"yyyymmdd\THH");
    save_object(string(outputfilename, tstamp, ".jld2"), fit_params); # rdata, rds
    CSV.write(string(outputfilename, tstamp, ".csv"), df);
end



# 4 parameters, Negative Binomial distribution
frac_asymp = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5];
for fa in frac_asymp 
    @printf("running fA = %.2f\n", fa);
    plotdir = @sprintf("%s\\plots\\NB_4p_fA%03d\\", path_workspace, round(Int, fa*100));
    ispath(plotdir) || mkdir(plotdir)
    outputfilename = @sprintf("%s\\outputs\\NB_r10_fit_4p_fA%03d_", path_workspace, round(Int, fa*100));
    
    df = DataFrame(data_id=zeros(n_ids), fit_id=zeros(n_ids), s0=zeros(n_ids), i0=zeros(n_ids), 
        R0=zeros(n_ids), n0=zeros(n_ids), objective=zeros(n_ids), retcode=zeros(n_ids)); 
    fit_params = Vector{Any}(undef, n_ids);
    
    for i in eachindex(ids)
        @printf("running %d of %d\n", i, length(ids));   
        params = initialize_params((fA=fa, loss=nll_4p_NB, ode=seiarw_2ag_erlang_vacc_two_rounds!, inits=inits, 
            NB_size=10));     
        # population size of the admin unit in which the outbreak was observed
        id_pop = dat_fit[dat_fit.ID_outbreak .== ids[i], [:population, :ID]];
        # update the params.u0 based on the pop
        params = merge(params, (population = id_pop.population[1],)); # pop is Vector{Float64}
        params = merge(params, (data_id = id_pop.ID[1],)); 
        params = merge(params, (fit_id = i,)); 
        d = dat_ts[dat_ts.ID_outbreak .== ids[i], [:sCh, :temporal_scale, :TL, :location]];
        params = merge(params, (TL = d.TL,));  
        params = merge(params, (location = d.location[1],));  
        # nobs = size(d, 1);
        if d.temporal_scale[1] == "weekly"
            params = merge(params, (report_freq = 7.0,));
        else
            params = merge(params, (report_freq = 1.0,));
        end
        y = round.(Int, d.sCh); # data
        params = merge(params, (data = y,));   
        params = merge(params, (tend = params.report_freq * length(y),));   
        try 
            sol = fit_model(params);
        # store results
            df[i, :data_id] = id_pop.ID[1];
            df[i, :fit_id] = i;
            xhat = extract_xhat(sol);
            params = merge(params, xhat);
            df[i, :s0] = xhat.s0;
            df[i, :i0] = xhat.i0;
            df[i, :R0] = xhat.R0;
            if length(sol.u) > 3
                df[i, :n0] = xhat.n0;
            end    
            df[i, :objective] = sol.objective;
            df[i, :retcode] = Int(sol.retcode); 
            params = merge(params, (sol_optim = sol,));
            x = run_model(params);
            params = merge(params, (modeled = x,));
            fit_params[i] = params;
            save_plot(params, plotdir);
        catch e
            @printf("fitting for i = %d failed\n", i);
        end
    end
    
    dt = Dates.now();
    tstamp = Dates.format(dt, dateformat"yyyymmdd\THHMM");
    save_object(string(outputfilename, tstamp, ".jld2"), fit_params); # rdata, rds
    CSV.write(string(outputfilename, tstamp, ".csv"), df);
end




## 5 parameters including expon

x0 = [logit(0.2), logit(0.1), log(2.0), logit(0.5), logit(0.5)];
lower = [logit(0.1), logit(0.0001), log(1.1), logit(0.1), logit(0.1)];
upper = [logit(0.99), logit(0.9), log(20.0), logit(0.99), logit(0.99)];
inits = LVector(x0=x0, lower=lower, upper=upper);

frac_asymp = [0.05, 0.1, 0.2, 0.3, 0.4];
for fa in frac_asymp 
    df = DataFrame(data_id=zeros(n_ids), fit_id=zeros(n_ids), s0=zeros(n_ids), i0=zeros(n_ids), R0=zeros(n_ids),
    n0=zeros(n_ids), expon=zeros(n_ids), objective=zeros(n_ids), retcode=zeros(n_ids)); 
    fit_params = Vector{Any}(undef, n_ids);
    
    @printf("running fA = %.2f\n", fa);    
    plotdir = @sprintf("%s\\plots\\5p_fA%03d\\", path_workspace, round(Int, fa*100))
    ispath(plotdir) || mkdir(plotdir)
    outputfilename = @sprintf("%s\\outputs\\fit_5p_fA%03d_", path_workspace, round(Int, fa*100))
    parm_init = initialize_params((fA=fa, loss=nll_5p, ode=seiarw_2ag_erlang_vacc_two_rounds!, inits=inits));      
    for i in eachindex(ids)
        @printf("running %d of %d\n", i, length(ids));
        id_pop = dat_fit[dat_fit.ID_outbreak .== ids[i], [:population, :ID]];
        # update the params.u0 based on the pop
        parm = merge(parm_init, (population = id_pop.population[1],)); # pop is Vector{Float64}
        parm = merge(parm, (data_id = id_pop.ID[1],));
        parm = merge(parm, (fit_id = i,));
        d = dat_ts[dat_ts.ID_outbreak .== ids[i], [:sCh, :temporal_scale, :TL, :location]];
        parm = merge(parm, (TL = d.TL,));  
        parm = merge(parm, (location = d.location[1],));  
        # nobs = size(d, 1);
        if d.temporal_scale[1] == "weekly"
            parm = merge(parm, (report_freq = 7.0,));
        else
            parm = merge(parm, (report_freq = 1.0,));
        end
        y = round.(Int, d.sCh); # data
        parm = merge(parm, (data = y,));   
        parm = merge(parm, (tend = parm.report_freq * length(y),));
        try 
            sol = fit_model(parm);
        # store results
            df[i, :data_id] = id_pop.ID[1];
            df[i, :fit_id] = i;
            xhat = extract_xhat(sol);
            parm = merge(parm, xhat);
            df[i, :s0] = xhat.s0;
            df[i, :i0] = xhat.i0;
            df[i, :R0] = xhat.R0;
            if length(sol.u) > 3
                df[i, :n0] = xhat.n0;
            end
            if length(sol.u) > 4
                df[i, :expon] = xhat.expon;
            end
    
            df[i, :objective] = sol.objective;
            df[i, :retcode] = Int(sol.retcode); 
            parm = merge(parm, (sol_optim = sol,));
            x = run_model(parm);
            parm = merge(parm, (modeled = x,));
            fit_params[i] = parm;
            save_plot(parm, plotdir);
        catch e
            @printf("fitting for i = %d failed\n", i);
        end
    end
    
    dt = Dates.now();
    tstamp = Dates.format(dt, dateformat"yyyymmdd\THHMM");
    save_object(string(outputfilename, tstamp, ".jld2"), fit_params); # rdata, rds
    CSV.write(string(outputfilename, tstamp, ".csv"), df);
end

