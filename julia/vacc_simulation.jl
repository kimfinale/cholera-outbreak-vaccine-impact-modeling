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
using JLD2
# load model function, parameter LabelledArrays
path_workspace = "C:\\Users\\jonghoon.kim\\Documents\\CholeraOutbreakModel\\julia"
include(joinpath(path_workspace, "parameters.jl"));
include(joinpath(path_workspace, "seiarw_2ag_erlang_vacc.jl"));
include(joinpath(path_workspace, "utils.jl"));

Random.seed!(42);
# fitted parameter
f = jldopen(string(path_workspace, "\\outputs\\fit_4p_fA010_20241203T18.jld2"), "r");
fit_params = f["single_stored_object"];
# fit_params[708] = 0; # later found to be undef

vacc_start_weeks = 3.0:7.0;
vacc_start_color = cgrad(:inferno, length(vacc_start_weeks), categorical = true);

pathdir = string(path_workspace, "\\plots\\vacc\\");
vacc_fit_params = Vector{Any}(undef, length(fit_params));

for i in 1:length(fit_params)
    @printf("running %d of %d\n", i, length(fit_params));
    p = fit_params[i]; # extract fitted parameter values
    if p == 0
        @printf("empty element\n"); 
        vacc_fit_params[i] = 0;       
    else
        parm = initialize_params();
        # update the parameters based on the fitted results
        parm = merge(parm, (s0=p.s0, i0=p.i0, R0=p.R0, n0=p.n0, tend=p.tend, report_freq=p.report_freq,
            TL=p.TL, data_id=p.data_id, fit_id=p.fit_id, data=p.data, location=p.location, population=p.population));
        # vaccine model with reporting daily
        parm = merge(parm, (ode=seiarw_2ag_erlang_vacc_two_rounds!, report_freq=1.0)); 
        inc_novacc = run_model(parm); # baseline incidence
        parm = merge(parm, (inc_novacc = inc_novacc,));
        plt = plot(inc_novacc, label="no vacc", linewidth=2, linecolor=:black,
            title=string("ID: ", p.data_id, ", loc: ", p.location), xlabel = "Day", ylabel = "Incidence");
        for j in 1:length(vacc_start_weeks)
            vw = vacc_start_weeks[j] * 7.0;
            if (vw + parm.campaign_1d_dur) < parm.tend  
                parm = merge(parm, (vacc_1d_eff_1=0.2, vacc_1d_eff_2=0.6, vacc_1d_cov=0.8,
                    vacc_1d_immunity_rate=1/7.0, campaign_1d_start=vw, campaign_1d_dur=7.0));
                parm = merge(parm, (vacc_1d_times = LVector(start = parm.campaign_1d_start, 
                stop = parm.campaign_1d_start + parm.campaign_1d_dur),));
                inc_v = run_model_vacc(parm);
                vw_int = round(Int, vw);
                if (vw_int < 0)
                    vw_int = 0; # 0 for pre-emptive vaccination
                end
                # C[1-9]T[1-9] represent the coverage proportion (*10) and start timing (weeks) for the vaccination campaign 
                parm = merge(parm, eval(Meta.parse(string("(inc_C8T", vw_int ÷ 7, " = inc_v,)"))));
                plot!(inc_v, color=vacc_start_color[j]);
            end
        end
        vacc_fit_params[i] = parm;
        dt = Dates.now();
        tstamp = Dates.format(dt, dateformat"yyyymmdd\THHMM");
        png(plt, string(pathdir, "fig_", p.fit_id, "_id_", p.data_id, "_", tstamp, ".png"));
    end
end

jldsave(string(path_workspace, "\\outputs\\vacc_4p_", tstamp, ".jld2"); vacc_fit_params);

# prepare a table 
dat_ts = DataFrame(CSV.File(joinpath(path_workspace, "data", "outbreak_data_ts.csv"); header=1, delim=","));
dat = DataFrame(CSV.File(joinpath(path_workspace, "data", "outbreak_data_summary.csv"); header=1, delim=","));
# create ID variables to compare with other data set (e.g., time series data)
dat.ID = 1:size(dat,1); 
dat.ID_outbreak = string.(dat.location, "-", dat.start_date, "-", dat.end_date);
dat.attack_rate_naive = dat.total_suspected_cases ./ dat.population;
# data for the outbreaks affected by OCV
dat_ocv = DataFrame(CSV.File(joinpath(path_workspace, "data", "ocv_long_dataset.csv"); header=1, delim=","));
d = dat[dat.ID_outbreak .∉ Ref(dat_ocv.ID_outbreak), :]; # n=824
ids = d.ID_outbreak;

name_inc = ["novacc"; collect(string.("C8T", 3:7))] # name of vaccination strategies
i = 1;

data_id = Matrix(dat[dat.ID_outbreak .== ids[i], [:ID]])[1];
res = vacc_fit_params[i];
DF = DataFrame(data_id = [data_id for i in 1:res.tend]);
DF[!, :ID_outbreak] .= ids[i];
DF[!, :date] = collect(res.TL[1]:Dates.Day(1):res.TL[1]+Dates.Day(res.tend-1))
missing_vec = Vector{Union{Missing,Float64}}(missing, round(Int, res.tend));
for n in name_inc
    DF[!, string("inc_", n)] = missing_vec
end
string_keys = string.(keys(res));
n_inc = sum(contains.(string_keys, r"inc_*."));
for k in 1:n_inc
    DF[!, string("inc_", name_inc[k])] = eval(Meta.parse(string("res.", string("inc_", name_inc[k]))));
end


for i in 2:length(ids)
    @printf("%d of %d\n", i, length(ids))
    data_id = Matrix(dat[dat.ID_outbreak .== ids[i], [:ID]])[1];
    res = vacc_fit_params[i];
    if res != 0
        df = DataFrame(data_id = [data_id for i in 1:res.tend]);
        df[!, :ID_outbreak] .= ids[i];
        df[!, :date] = collect(res.TL[1]:Dates.Day(1):res.TL[1]+Dates.Day(res.tend-1))
        missing_vec = Vector{Union{Missing,Float64}}(missing, round(Int, res.tend));
        for n in name_inc
            df[!, string("inc_", n)] = missing_vec
        end
        string_keys = string.(keys(res));
        n_inc = sum(contains.(string_keys, r"inc_*."));
        for k in 1:n_inc
            df[!, string("inc_", name_inc[k])] = eval(Meta.parse(string("res.", string("inc_", name_inc[k]))));
        end
        DF = [DF; df];
    end
end

CSV.write(string(path_workspace, "\\outputs\\vacc_4p_", tstamp, ".csv"), DF);

# Model fit summary data
f1 = jldopen(string(path_workspace, "\\outputs\\fit_4p_fA005_20240215T1913.jld2"), "r");
fits_f1 = f1["single_stored_object"]; 
f2 = jldopen(string(path_workspace, "\\outputs\\fit_4p_fA050_20240216T0147.jld2"), "r");
fits_f2 = f2["single_stored_object"]; 
f3 = jldopen(string(path_workspace, "\\outputs\\fit_5p_fA005_20240216T1809.jld2"), "r");
fits_f3 = f3["single_stored_object"];  
f4 = jldopen(string(path_workspace, "\\outputs\\fit_5p_fA050_20240216T1625.jld2"), "r");
fits_f4 = f4["single_stored_object"];

# col_names = ["inc_novacc_4p_fA005", "inc_novacc_4p_fA050", "inc_novacc_5p_fA005", "inc_novacc_5p_fA050"];
name_inc = [collect(string.("inc_novacc_P4fA", ["005","050"])); collect(string.("inc_novacc_P5fA",["005","050"]))]; # na

# realized an undef element and set to zero for programming convenience 
fits_f1[708] = 0;
i = 1;
fit = fits_f1[i];
dlen = length(fit.data);
DF = DataFrame(data_id = [fit.data_id for i in 1:dlen]);
DF[!, :date] = fit.TL;
DF[!, :location] = [fit.location for i in 1:dlen];
DF[!, :inc_data] = fit.data;
DF[!, name_inc[1]] = round.(fit.modeled, digits=2);    
DF[!, name_inc[2]] = round.(fits_f2[i].modeled, digits=2);
DF[!, name_inc[3]] = round.(fits_f3[i].modeled, digits=2); 
DF[!, name_inc[4]] = round.(fits_f4[i].modeled, digits=2);

for i in 2:length(fits_f1)
    @printf("%d of %d\n", i, length(fits_f1))
    fit = fits_f1[i];
    if fit != 0
        dlen = length(fit.data);
        df = DataFrame(data_id = [fit.data_id for i in 1:dlen]);
        df[!, :date] = fit.TL;
        df[!, :location] = [fit.location for i in 1:dlen];
        df[!, :inc_data] = fit.data;
        df[!, name_inc[1]] = round.(fit.modeled, digits=2);    
        df[!, name_inc[2]] = round.(fits_f2[i].modeled, digits=2);
        df[!, name_inc[3]] = round.(fits_f3[i].modeled, digits=2); 
        df[!, name_inc[4]] = round.(fits_f4[i].modeled, digits=2);        
        
        DF = [DF; df];
    end
end

dt = Dates.now();
tstamp = Dates.format(dt, dateformat"yyyymmdd\THHMM");
CSV.write(string(path_workspace, "\\outputs\\fit_inc_", tstamp, ".csv"), DF);

# Model fit summary data
f1 = jldopen(string(path_workspace, "\\outputs\\NB_r10_fit_4p_fA005_20240217T1359.jld2"), "r");
fits_f1 = f1["single_stored_object"]; 
f2 = jldopen(string(path_workspace, "\\outputs\\NB_r10_fit_4p_fA050_20240217T2056.jld2"), "r");
fits_f2 = f2["single_stored_object"]; 
f3 = jldopen(string(path_workspace, "\\outputs\\NB_r50_fit_4p_fA005_20240218T0825.jld2"), "r");
fits_f3 = f3["single_stored_object"];  
f4 = jldopen(string(path_workspace, "\\outputs\\NB_r50_fit_4p_fA050_20240218T1517.jld2"), "r");
fits_f4 = f4["single_stored_object"];

# col_names = ["inc_novacc_4p_fA005", "inc_novacc_4p_fA050", "inc_novacc_5p_fA005", "inc_novacc_5p_fA050"];
name_inc = [collect(string.("inc_novacc_NBr10_P4fA", ["005","050"])); collect(string.("inc_novacc_NBr50_P4fA",["005","050"]))]; # na

# realized an undef element and set to zero for programming convenience 
fits_f1[708] = 0;
i = 1;
fit = fits_f1[i];
dlen = length(fit.data);
DF = DataFrame(data_id = [fit.data_id for i in 1:dlen]);
DF[!, :date] = fit.TL;
DF[!, :location] = [fit.location for i in 1:dlen];
DF[!, :inc_data] = fit.data;
DF[!, name_inc[1]] = round.(fit.modeled, digits=2);    
DF[!, name_inc[2]] = round.(fits_f2[i].modeled, digits=2);
DF[!, name_inc[3]] = round.(fits_f3[i].modeled, digits=2); 
DF[!, name_inc[4]] = round.(fits_f4[i].modeled, digits=2);

for i in 2:length(fits_f1)
    @printf("%d of %d\n", i, length(fits_f1))
    fit = fits_f1[i];
    if fit != 0
        dlen = length(fit.data);
        df = DataFrame(data_id = [fit.data_id for i in 1:dlen]);
        df[!, :date] = fit.TL;
        df[!, :location] = [fit.location for i in 1:dlen];
        df[!, :inc_data] = fit.data;
        df[!, name_inc[1]] = round.(fit.modeled, digits=2);    
        df[!, name_inc[2]] = round.(fits_f2[i].modeled, digits=2);
        df[!, name_inc[3]] = round.(fits_f3[i].modeled, digits=2); 
        df[!, name_inc[4]] = round.(fits_f4[i].modeled, digits=2);        
        
        DF = [DF; df];
    end
end

dt = Dates.now();
tstamp = Dates.format(dt, dateformat"yyyymmdd\THHMM");
CSV.write(string(path_workspace, "\\outputs\\fit_inc_NB", tstamp, ".csv"), DF);

# -----------------------------------------------------------------------------------------------
# pre-emptive vaccination 
dat = DataFrame(CSV.File(joinpath(path_workspace, "data", "outbreak_data_20241121.csv"); header=1, delim=","));
ids = unique(dat.id_outbreak, dims=1);
ids_df = DataFrame(id_outbreak = ids);
ids_df.data_id = 1:size(ids_df,1);
# just to know the data_id vs. id_outbreak
# CSV.write(string(path_workspace, "\\outputs\\data_id_id_outbreak_20241202.csv"), ids_df);
n_ids = length(unique(ids, dims=1))
dat = innerjoin(dat, ids_df, on = :id_outbreak);

age_dist = DataFrame(CSV.File("C:\\Users\\jonghoon.kim\\Documents\\CholeraOutbreakModel\\outputs\\wpp_pop_by_age_20241022.csv"; header=1, delim=","));
dropmissing!(age_dist, :var"ISO3 Alpha-code");
dropmissing!(age_dist, :Year);
pathdir = string(path_workspace, "\\plots\\vacc\\");
vacc_fit_params = Vector{Any}(undef, length(fit_params));

df = DataFrame(data_id = zeros(length(fit_params)));
df[!, :id_outbreak] .= "";
df[!, :CI_novacc] .= zeros(length(fit_params));
df[!, :CI_VC60] .= zeros(length(fit_params));
df[!, :CI_VC75] .= zeros(length(fit_params));
df[!, :CI_VC90] .= zeros(length(fit_params));

mycolors = palette(:tol_bright);
vacc_covs = [0.6, 0.75, 0.9];

for i in 1:length(fit_params)
    @printf("running %d of %d\n", i, length(fit_params));
    # Check if the element exists using isdefined I happend to use undefined (could have used nothing) when fitting the parameters
    if !isassigned(fit_params, i)
        @printf("undefined element at index %d\n", i);
        vacc_fit_params[i] = 0;
        continue
    end
       
    p = fit_params[i]; # Now safe to access since we checked isdefined
    if p == 0
        @printf("empty element\n");
        vacc_fit_params[i] = 0;
        continue
    end

    d = dat[dat.id_outbreak .== p.id_outbreak, [:year, :country]];
    # prop under 5 varies by country and year
    prop_u5 = age_dist[(age_dist.var"ISO3 Alpha-code" .== d.country[1]) .& (age_dist.Year .== d.year[1]), :prop_u5]
    # if prop_u5[1] < 0.15
    #     @printf("prop_u5 is smaller than 0.15:  %.3f\n",  prop_u5[1]);
    # end
    df[i, :data_id] = p.data_id;
    df[i, :id_outbreak] = p.id_outbreak;
    parm = initialize_params();
    # update the parameters based on the fitted results
    parm = merge(parm, (fA=p.fA, prop_children=prop_u5[1], s0=p.s0, i0=p.i0, R0=p.R0, n0=p.n0, tend=p.tend, report_freq=p.report_freq,
        TL=p.TL, data_id=p.data_id, fit_id=p.fit_id, data=p.data, location=p.location, population=p.population));
    # vaccine model with reporting daily
    parm = merge(parm, (ode=seiarw_2ag_erlang_vacc_single_round!, report_freq=7.0)); 
    inc_novacc = run_model(parm); # baseline incidence
    df[i, :CI_novacc] = sum(inc_novacc); 
    parm = merge(parm, (inc_novacc = inc_novacc,));
    plt = plot(parm.TL, inc_novacc, label="no vacc", linewidth=2, linecolor=:black,
        title=string("ID: ", p.data_id), xlabel = "", ylabel = "Incidence");    
    scatter!(parm.TL, parm.data, label="data", fillcolor = :green, markershape = :circle, markersize = 5, markercolor = :red,
        markerstrokewidth = 1, markerstrokecolor = :red);
    # pre-emptive vaccination
    for k in 1:3   # pre-emptive vaccination
        p = fit_params[i]; # extract fitted parameter values
        parm = initialize_params();
        # update the parameters based on the fitted results
        parm = merge(parm, (fA=p.fA, prop_children=prop_u5[1], s0=p.s0, i0=p.i0, R0=p.R0, n0=p.n0, tend=p.tend, report_freq=p.report_freq,
            TL=p.TL, data_id=p.data_id, fit_id=p.fit_id, data=p.data, location=p.location, population=p.population));
        # vaccine model with reporting daily
        parm = merge(parm, (ode=seiarw_2ag_erlang_vacc_single_round!, report_freq=7.0)); 
        parm = merge(parm, (preemptive_vacc = true, vacc_1d_cov=vacc_covs[k]));
        inc_v = run_model(parm); # incidence under pre-emptive vaccination
        df[i, k+3] = sum(inc_v);
        plot!(parm.TL, inc_v, color=mycolors[k], label=string("vacc cov = ", vacc_covs[k]));
    end  
    vacc_fit_params[i] = parm;
    dt = Dates.now();
    tstamp = Dates.format(dt, dateformat"yyyymmdd\THHMM");
    png(plt, string(pathdir, "id_", parm.data_id, "_", tstamp, ".png"));
end

dt = Dates.now();
tstamp = Dates.format(dt, dateformat"yyyymmdd\THH");

CSV.write(string(path_workspace, "\\outputs\\vacc_fit_params_", tstamp, ".csv"), df);
jldsave(string(path_workspace, "\\outputs\\vacc_impact_CI_", tstamp, ".jld2"); vacc_fit_params);


# -----------------------------------------------------------------------------------------------------------------
# checking indiret vaccine effectiveness
dat = DataFrame(CSV.File(joinpath(path_workspace, "data", "outbreak_data_20241121.csv"); header=1, delim=","));
ids = unique(dat.id_outbreak, dims=1);
ids_df = DataFrame(id_outbreak = ids);
ids_df.data_id = 1:size(ids_df,1);
# just to know the data_id vs. id_outbreak
# CSV.write(string(path_workspace, "\\outputs\\data_id_id_outbreak_20241202.csv"), ids_df);
n_ids = length(unique(ids, dims=1))
dat = innerjoin(dat, ids_df, on = :id_outbreak);
age_dist = DataFrame(CSV.File("C:\\Users\\jonghoon.kim\\Documents\\CholeraOutbreakModel\\outputs\\wpp_pop_by_age_20241022.csv"; header=1, delim=","));
dropmissing!(age_dist, :var"ISO3 Alpha-code");
dropmissing!(age_dist, :Year);
pathdir = string(path_workspace, "\\plots\\vacc\\");
vacc_fit_params = Vector{Any}(undef, length(fit_params));
p
df = DataFrame(data_id = zeros(length(fit_params)));
df[!, :id_outbreak] .= "";
df[!, :CI_novacc] .= zeros(length(fit_params));
df[!, :CI_VC60_novacc] .= zeros(length(fit_params));
df[!, :CI_VC60_vacc] .= zeros(length(fit_params));
df[!, :CI_VC75_novacc] .= zeros(length(fit_params));
df[!, :CI_VC75_vacc] .= zeros(length(fit_params));
df[!, :CI_VC90_novacc] .= zeros(length(fit_params));
df[!, :CI_VC90_vacc] .= zeros(length(fit_params));

# mycolors = palette(:tab10);
mycolors = palette(:tol_bright);
vacc_covs = [0.6, 0.75, 0.9];

for i in 1:length(fit_params)
    @printf("running %d of %d\n", i, length(fit_params));
    # Check if the element exists using isdefined I happend to use undefined (could have used nothing) when fitting the parameters
    if !isassigned(fit_params, i)
        @printf("undefined element at index %d\n", i);
        vacc_fit_params[i] = 0;
        continue
    end
       
    p = fit_params[i]; # Now safe to access since we checked isdefined
    if p == 0
        @printf("empty element\n");
        vacc_fit_params[i] = 0;
        continue
    end

    d = dat[dat.id_outbreak .== p.id_outbreak, [:year, :country]];
    # prop under 5 varies by country and year
    prop_u5 = age_dist[(age_dist.var"ISO3 Alpha-code" .== d.country[1]) .& (age_dist.Year .== d.year[1]), :prop_u5]
   
    df[i, :data_id] = p.data_id;
    df[i, :id_outbreak] = p.id_outbreak;
    parm = initialize_params();
    # update the parameters based on the fitted results
    parm = merge(parm, (fA=p.fA, prop_u5=prop_u5[1], s0=p.s0, i0=p.i0, R0=p.R0, n0=p.n0, tend=p.tend, report_freq=p.report_freq,
        TL=p.TL, data_id=p.data_id, fit_id=p.fit_id, data=p.data, location=p.location, population=p.population));
    # vaccine model with reporting daily
    parm = merge(parm, (ode=seiarw_2ag_erlang_vacc_single_round!, report_freq=7.0)); 
    inc_novacc = run_model(parm); # baseline incidence
    df[i, :CI_novacc] = sum(inc_novacc); 
    parm = merge(parm, (inc_novacc = inc_novacc,));
    plt = plot(parm.TL, inc_novacc, label="no vacc", linewidth=2, linecolor=:black,
        title=string("ID: ", p.data_id), xlabel = "", ylabel = "Incidence");    
    scatter!(parm.TL, parm.data, label="data", fillcolor = :green, markershape = :circle, markersize = 5, markercolor = :red,
        markerstrokewidth = 1, markerstrokecolor = :red);
    # pre-emptive vaccination
    for k in 1:3   # pre-emptive vaccination
        p = fit_params[i]; # extract fitted parameter values
        parm = initialize_params();
        # update the parameters based on the fitted results
        parm = merge(parm, (fA=p.fA, prop_u5=prop_u5[1], s0=p.s0, i0=p.i0, R0=p.R0, n0=p.n0, tend=p.tend, report_freq=p.report_freq,
            TL=p.TL, data_id=p.data_id, fit_id=p.fit_id, data=p.data, location=p.location, population=p.population));
        # vaccine model with reporting daily
        parm = merge(parm, (ode=seiarw_2ag_erlang_vacc_single_round!, report_freq=7.0)); 
        parm = merge(parm, (preemptive_vacc = true, vacc_1d_cov=vacc_covs[k]));
        inc_v = extract_inc_by_vacc_state(parm); # incidence under pre-emptive vaccination
        df[i, ((k-1)*2+4)] = sum(inc_v.inc_novacc);
        df[i, ((k-1)*2+5)] = sum(inc_v.inc_vacc);
    end  
    vacc_fit_params[i] = parm;
end

dt = Dates.now();
tstamp = Dates.format(dt, dateformat"yyyymmdd\THH");

CSV.write(string(path_workspace, "\\outputs\\vacc_fit_params_by_vacc_state_", tstamp, ".csv"), df);
jldsave(string(path_workspace, "\\outputs\\vacc_impact_CI_by_vacc_state_", tstamp, ".jld2"); vacc_fit_params);

# -----------------------------------------------------------------------------------------------------------------
# checking indiret vaccine effectiveness 2
# apply sampled direct efficacy
# make sure that p = initialize_params(); p.sigma = 0, p.sigma_v = 0 

vacc_params = DataFrame(CSV.File("C:\\Users\\jonghoon.kim\\Documents\\CholeraOutbreakModel\\outputs\\params_20241017.csv"));

dat = DataFrame(CSV.File(joinpath(path_workspace, "data", "outbreak_data_20241121.csv"); header=1, delim=","));
ids = unique(dat.id_outbreak, dims=1);
ids_df = DataFrame(id_outbreak = ids);
ids_df.data_id = 1:size(ids_df,1);
# just to know the data_id vs. id_outbreak
# CSV.write(string(path_workspace, "\\outputs\\data_id_id_outbreak_20241202.csv"), ids_df);
n_ids = length(unique(ids, dims=1))
dat = innerjoin(dat, ids_df, on = :id_outbreak);
age_dist = DataFrame(CSV.File("C:\\Users\\jonghoon.kim\\Documents\\CholeraOutbreakModel\\outputs\\wpp_pop_by_age_20241022.csv"; header=1, delim=","));
dropmissing!(age_dist, :var"ISO3 Alpha-code");
dropmissing!(age_dist, :Year);
pathdir = string(path_workspace, "\\plots\\vacc\\");
vacc_fit_params = Vector{Any}(undef, length(fit_params));

df = DataFrame(runid = Int64.(zeros(length(fit_params))), data_id = zeros(length(fit_params)));
df[!, :id_outbreak] .= "";
df[!, :CI_novacc] .= zeros(length(fit_params));
df[!, :CI_VC60_novacc] .= zeros(length(fit_params));
df[!, :CI_VC60_vacc] .= zeros(length(fit_params));
df[!, :CI_VC75_novacc] .= zeros(length(fit_params));
df[!, :CI_VC75_vacc] .= zeros(length(fit_params));
df[!, :CI_VC90_novacc] .= zeros(length(fit_params));
df[!, :CI_VC90_vacc] .= zeros(length(fit_params));

# mycolors = palette(:tab10);
mycolors = palette(:tol_bright);
vacc_covs = [0.6, 0.75, 0.9];
nsample = 200;
DF = DataFrame(); # df's will be appended to DF

for n in 1:nsample
    @printf("running sample %d of %d\n", n, nsample);
    df = DataFrame(runid = Int64.(zeros(length(fit_params))), data_id = zeros(length(fit_params)));
    df[!, :id_outbreak] .= "";
    df[!, :CI_novacc] .= zeros(length(fit_params));
    df[!, :CI_VC60_novacc] .= zeros(length(fit_params));
    df[!, :CI_VC60_vacc] .= zeros(length(fit_params));
    df[!, :CI_VC75_novacc] .= zeros(length(fit_params));
    df[!, :CI_VC75_vacc] .= zeros(length(fit_params));
    df[!, :CI_VC90_novacc] .= zeros(length(fit_params));
    df[!, :CI_VC90_vacc] .= zeros(length(fit_params));
    df[!, :runid] .= n;
    for i in 1:length(fit_params)
        @printf("running %d of %d\n", i, length(fit_params));
        # Check if the element exists using isdefined I happend to use undefined (could have used nothing) when fitting the parameters
        if !isassigned(fit_params, i)
            @printf("undefined element at index %d\n", i);
            vacc_fit_params[i] = 0;
            continue
        end
        p = fit_params[i]; # Now safe to access since we checked isdefined
        if p == 0
            @printf("empty element\n");
            vacc_fit_params[i] = 0;
            continue
        end

        d = dat[dat.id_outbreak .== p.id_outbreak, [:year, :country]];
        # prop under 5 varies by country and year
        prop_u5 = age_dist[(age_dist.var"ISO3 Alpha-code" .== d.country[1]) .& (age_dist.Year .== d.year[1]), :prop_u5]
        ve_1d_u5 = vacc_params[n, :vacc_effect_direct_u5];
        ve_1d_5p = vacc_params[n, :vacc_effect_direct_5p];
        df[i, :data_id] = p.data_id;
        df[i, :id_outbreak] = p.id_outbreak;
        parm = initialize_params();
        # update the parameters based on the fitted results
        parm = merge(parm, (fA=p.fA, prop_u5=prop_u5[1], vacc_1d_eff_1 = ve_1d_u5, vacc_1d_eff_2 = ve_1d_5p,
            s0=p.s0, i0=p.i0, R0=p.R0, n0=p.n0, tend=p.tend, 
            report_freq=p.report_freq, TL=p.TL, data_id=p.data_id, fit_id=p.fit_id, 
            data=p.data, location=p.location, population=p.population));
        # vaccine model with reporting daily
        parm = merge(parm, (ode=seiarw_2ag_erlang_vacc_single_round!, report_freq=7.0)); 
        inc_novacc = run_model(parm); # baseline incidence
        df[i, :CI_novacc] = sum(inc_novacc); 
        parm = merge(parm, (inc_novacc = inc_novacc,));
        # pre-emptive vaccination
        for k in 1:3   
            p = fit_params[i]; # extract fitted parameter values
            parm = initialize_params();
            # update the parameters based on the fitted results
            parm = merge(parm, (fA=p.fA, prop_u5=prop_u5[1], vacc_1d_eff_1 = ve_1d_u5, vacc_1d_eff_2 = ve_1d_5p,
            s0=p.s0, i0=p.i0, R0=p.R0, n0=p.n0, tend=p.tend, 
            report_freq=p.report_freq, TL=p.TL, data_id=p.data_id, fit_id=p.fit_id, 
            data=p.data, location=p.location, population=p.population));
            # vaccine model with reporting daily
            parm = merge(parm, (ode=seiarw_2ag_erlang_vacc_single_round!, report_freq=7.0)); 
            parm = merge(parm, (preemptive_vacc = true, vacc_1d_cov=vacc_covs[k]));
            inc_v = extract_inc_by_vacc_state(parm); # incidence under pre-emptive vaccination
            df[i, ((k-1)*2+5)] = sum(inc_v.inc_novacc);
            df[i, ((k-1)*2+6)] = sum(inc_v.inc_vacc);
        end  
    end
    append!(DF, df)
end

dt = Dates.now();
tstamp = Dates.format(dt, dateformat"yyyymmdd\THH");
CSV.write(string(path_workspace, "\\outputs\\vacc_fit_params_by_vacc_state_", tstamp, ".csv"), DF);

                                                                                                      
