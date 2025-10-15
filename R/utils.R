#' Fcalculate_impact_by_subset
#'
#' Returns summary statistics by grouping variable
#'
#' @param group_id group id is a vector of length n, then x[(1+lag):n] / x[1:(n-lag)].
#' @return suitably lagged and iterated fractions

# Fixed version using proper data.table syntax
calculate_impact_by_subset <- function(data,
                                       subset_id,
                                       subset_var = "size_grp",
                                       outcome_var = "pct_reduc_case",
                                       grouping_vars = c("vacc_week", "vacc_cov"),
                                       summary_fun = "median",
                                       quantile_prob = 0.5,
                                       na_rm = TRUE) {

  # Define summary function
  calc_summary <- function(x) {
    switch(summary_fun,
           "median" = median(x, na.rm = na_rm),
           "mean" = mean(x, na.rm = na_rm),
           "quantile" = quantile(x, probs = quantile_prob, na.rm = na_rm),
           stop("summary_fun must be 'median', 'mean', or 'quantile'"))
  }

  # Create the summary column name
  summary_col_name <- paste0(outcome_var, "_", summary_fun)
  if(summary_fun == "quantile") {
    summary_col_name <- paste0(summary_col_name, "_", quantile_prob)
  }

  # Perform the calculation using proper data.table syntax
  result <- data[get(subset_var) >= subset_id,
                 c(list(subset_id = subset_id,
                        subset_var = subset_var,
                        n_outbreaks = .N),
                   setNames(list(calc_summary(get(outcome_var))), summary_col_name)),
                 by = grouping_vars]  # Direct use of grouping_vars

  return(result)
}

#' Fit an exponential decay model to vaccine impact
#'
#' This function estimates the rate of exponential decay (\eqn{k}) in vaccine impact
#' over time and calculates the corresponding halving time ("weeks to 50\%" of the original effect).
#' The decay model assumes that the change in vaccine-attributable case reduction
#' follows an exponential pattern over \code{vacc_week}.
#'
#' @param df A data frame containing at least:
#'   \describe{
#'     \item{\code{pct_reduc_case}}{Numeric vector of proportional case reduction (0–1).}
#'     \item{\code{vacc_week}}{Numeric vector of time in weeks since vaccination.}
#'   }
#' @param min_weeks Minimum number of weekly observations required to fit the model.
#'   Defaults to 3.
#'
#' @details
#' The model fitted is:
#' \deqn{\log(\text{impact}) = \alpha - k \cdot \text{week}}
#' where \eqn{k} is the per-week decay rate. The half-life is calculated as:
#' \deqn{\text{Half-life} = \frac{\log(2)}{k}}
#' If the estimated \eqn{k \le 0}, the half-life is returned as \code{NA}.
#'
#' @return
#' A tibble with two columns:
#' \describe{
#'   \item{\code{k_hat}}{Estimated weekly decay rate in log scale.}
#'   \item{\code{half_life}}{Weeks until the effect is halved. \code{NA} if decay rate is non-positive.}
#' }
#'
#' @examples
#' library(dplyr)
#' df <- data.frame(
#'   vacc_week = 1:6,
#'   pct_reduc_case = c(0.12, 0.10, 0.08, 0.07, 0.05, 0.04)
#' )
#' fit_decay_model(df)
#'
#' @export
# fit_decay_model <- function(df, min_weeks = 3) {
#   # Return NA values if not enough data points
#   if (nrow(df) < min_weeks) {
#     return(tibble(k_hat = NA_real_, half_life = NA_real_))
#   }
#
#   # Avoid taking log(0) by adding a very small constant
#   eps <- 1e-6
#   df <- df %>%
#     mutate(impact_adj = pct_reduc_case + eps)
#
#   # Fit the exponential decay model: log(impact) = intercept - k * week
#   fit <- try(lm(log(impact_adj) ~ vacc_week, data = df), silent = TRUE)
#
#   # Return NA values if model fitting fails
#   if (inherits(fit, "try-error")) {
#     return(tibble(k_hat = NA_real_, half_life = NA_real_))
#   }
#
#   # Extract slope (negative of decay rate)
#   slope <- coef(fit)[["vacc_week"]] # or slope <- coef(fit)[2]
#   k_hat <- -slope
#
#   # Calculate half-life only if k_hat > 0
#   half_life <- ifelse(k_hat > 0, log(2) / k_hat, NA_real_)
#
#   tibble(k_hat = k_hat, half_life = half_life)
# }

#' Fit Exponential Decay Model
#'
#' Fits a simple exponential decay model: log(y) ~ x
#' where y = dependent variable (e.g., vaccine impact)
#'       x = independent variable (e.g., delay week or case trigger)
#'
#' @param df Dataframe containing the data
#' @param y_var Name of the dependent variable (character)
#' @param x_var Name of the independent variable (character)
#' @param min_rows Minimum number of rows required to fit the model
#'
#' @return A tibble with estimated decay rate (k_hat) and half-life
#' @export
fit_decay_model <- function(df, y_var = "pct_reduc_case", x_var = "vacc_week", min_rows = 3) {
  if (nrow(df) < min_rows) {
    return(tibble(k_hat = NA_real_, half_life = NA_real_))
  }

  eps <- 1e-6
  df <- df %>%
    mutate(impact_adj = .data[[y_var]] + eps)

  formula <- as.formula(paste("log(impact_adj) ~", x_var))

  fit <- try(lm(formula, data = df), silent = TRUE)

  if (inherits(fit, "try-error")) {
    return(tibble(k_hat = NA_real_, half_life = NA_real_))
  }

  slope <- coef(fit)[[x_var]]
  k_hat <- -slope
  half_life <- ifelse(k_hat > 0, log(2) / k_hat, NA_real_)

  tibble(k_hat = k_hat, half_life = half_life)
}


#' Fractional change
#'
#' Returns suitably lagged and iterated fractions.
#' An analog of the built-in diff function.
#'
#' @param x Numeric vector.
#' @param lag If x is a vector of length n, then x[(1+lag):n] / x[1:(n-lag)].
#' @return Numeric vector of fractional changes with some value close to 1 (1-eps) for the final value of the input vector
frac_change <- function(x, lag = 1, eps = 1e-6) {
  n <- length(x)
  if (lag >= n) return(rep(NA_real_, n))

  prev <- dplyr::lag(x, lag)
  drop <- 1 - (x / prev)

  # Special cases
  drop[prev == 0 & x == 0] <- 0
  drop[prev == 0 & x  > 0] <- NA_real_

  # Bound inside (0,1) for log safety
  drop <- pmin(pmax(drop, eps), 1 - eps)
}

#' Summarize quantiles and mean for one or more variables
#'
#' This function calculates user-specified quantiles and the mean for one or more
#' numeric variables in a data frame. The output includes a column named
#' \code{variable} indicating the input variable, and columns for the computed
#' quantiles and mean with generic names (e.g., \code{q025}, \code{q250},
#' \code{q500}, \code{q750}, \code{q975}, \code{mean}).
#'
#' @param .data A data frame or tibble.
#' @param ... One or more unquoted variable names (tidy evaluation supported)
#'   for which to compute quantiles and mean.
#' @param probs Numeric vector of probabilities in \[0,1\] at which to compute
#'   quantiles. Defaults to \code{c(0.025, 0.25, 0.5, 0.75, 0.975)}.
#'
#' @details
#' The output is tidy: for each grouping in the input data, and for each
#' requested variable, a single row is returned. The variable name is stored in
#' the \code{variable} column, while quantile and mean values are placed in
#' separate columns with generic names.
#'
#' @return
#' A tibble with one row per variable (within groups if grouped), containing a
#' \code{variable} column and summary columns for the requested quantiles and
#' mean.
#'
#' @examples
#' library(dplyr)
#' df <- tibble(a = rnorm(100), b = runif(100))
#' df %>% group_by() %>% summarize_quantiles(a, b)
#'
#' @export
summarize_quantiles <- function(.data, ...,
                                probs = c(0.025, 0.25, 0.50, 0.75, 0.975)) {
  quos <- rlang::enquos(...)
  q_cols <- paste0("q", gsub("0\\.", "", sprintf("%.3f", probs)))

  purrr::map_dfr(quos, function(var) {
    var_name <- rlang::as_label(var)

    q_exprs <- purrr::map2(
      probs, q_cols,
      ~ rlang::expr(quantile(!!var, !!.x, na.rm = TRUE))
    ) |> rlang::set_names(q_cols)

    mean_expr <- rlang::set_names(
      list(rlang::expr(mean(!!var, na.rm = TRUE))),
      "mean"
    )

    .data %>%
      dplyr::summarise(!!!q_exprs, !!!mean_expr, .groups = "drop") %>%
      dplyr::mutate(variable = var_name, .before = 1)
  })
}

#' Summarize Numeric Variables by Group
#'
#' Computes the mean and specified quantiles for selected numeric variables
#' within a data frame and reshapes the result to a long format with
#' labeled summary statistics.
#'
#' @param df A data frame containing the data to be summarized.
#' @param vars A character vector of variable names to summarize. If NULL,
#'   all numeric variables in `df` will be used.
#' @param group_name A string identifier to label the group (e.g., "short", "long").
#' @param probs A named numeric vector specifying which quantiles to compute.
#'   Default: c(q025 = 0.025, q250 = 0.25, q500 = 0.5, q750 = 0.75, q975 = 0.975)
#'
#' @return A tibble in long format with columns:
#'   - `variable`: name of the variable summarized
#'   - `mean`: the sample mean
#'   - additional columns for each quantile (e.g., `q025`, `q500`, ...)
#'   - `group`: the group label passed to `group_name`
#'
#' @examples
#' summarize_group(df, vars = c("k_hat", "half_life"), group_name = "short")
#'
#' @export
summarize_group <- function(df,
                            vars = NULL,
                            group_name,
                            probs = c(q025 = 0.025, q250 = 0.25,
                                      q500 = 0.5, q750 = 0.75, q975 = 0.975)) {
  # If vars not provided, select all numeric columns
  if (is.null(vars)) {
    vars <- names(df)[sapply(df, is.numeric)]
    if (length(vars) == 0) {
      stop("No numeric variables found in the data frame.")
    }
  }

  # Validate variable names
  if (!all(vars %in% names(df))) {
    stop("Some variables in 'vars' are not present in 'df'.")
  }

  # Validate probabilities
  if (any(probs < 0 | probs > 1)) {
    stop("'probs' must be between 0 and 1.")
  }

  # Base summary functions
  summary_funs <- list(
    # median = ~ median(.x, na.rm = TRUE),
    mean   = ~ mean(.x, na.rm = TRUE)
  )

  # Add quantile functions for each probability
  quantile_funs <- purrr::imap(probs, function(p, name) {
    fn <- function(.x) quantile(.x, p, na.rm = TRUE)
    fn
  })

  names(quantile_funs) <- names(probs)
  all_funs <- c(summary_funs, quantile_funs)

  df %>%
    dplyr::select(dplyr::all_of(vars)) %>%
    dplyr::summarise(
      dplyr::across(
        dplyr::everything(),
        all_funs,
        .names = "{.col}_{.fn}"
      ),
      .groups = "drop"
    ) %>%
    tidyr::pivot_longer(
      cols = matches("_(mean|q025|q250|q500|q750|q975)$"),
      names_to = c("variable", ".value"),
      names_pattern = paste0("^(.*)_(", paste(c("mean", names(probs)), collapse = "|"), ")$")
    ) %>%
    dplyr::mutate(group = group_name)
}


#' Extended Summary of Grouped Data
#'
#' Extends `summarize_group()` by including additional outbreak-level
#' diagnostics such as the proportion of constant-response outbreaks and
#' the number of infinite half-life estimates.
#'
#' @param df A data frame containing outbreak-level results.
#' @param vars A character vector of variable names to summarize. If NULL,
#'   all numeric columns will be summarized.
#' @param group_name A string to identify the group (e.g., "short", "medium").
#' @param probs A named numeric vector of quantiles to compute (default:
#'   c(2.5%, 25%, 50%, 75%, 97.5%)).
#'
#' @return A data frame with one row per variable-statistic combination.
#'   The structure matches `summarize_group()` and adds extra rows for:
#'   - `pct_constant`: Percentage of outbreaks with constant vaccine impact
#'   - `n_inf_half_life`: Number of outbreaks with infinite half-life
#'
#' @seealso \code{\link{summarize_group}} for the core summary method.
#'
#' @examples
#' summarize_group_extended(res_df, vars = c("half_life"), group_name = "short")
#'
#' @export
summarize_group_extended <- function(df,
                                     vars = NULL,
                                     group_name,
                                     probs = c(q025 = 0.025, q250 = 0.25,
                                               q500 = 0.5, q750 = 0.75, q975 = 0.975)) {

  # ---- Step 1: Core numeric summaries using existing summarize_group() ----
  core_summary <- summarize_group(df, vars = vars, group_name = group_name, probs = probs)

  # ---- Step 2: Initialize list to store additional summary rows ----
  extra_rows <- list()

  # ---- Step 3: Compute proportion of constant-impact outbreaks ----
  # Only if 'constant' column is available in the input data
  if ("constant" %in% names(df)) {
    pct_constant <- mean(df$constant, na.rm = TRUE) * 100
    extra_rows[["pct_constant"]] <- tibble(
      variable = "pct_constant",
      mean = pct_constant,
      q025 = NA, q250 = NA, q500 = NA, q750 = NA, q975 = NA,
      group = group_name
    )
  }

  # ---- Step 4: Count outbreaks with infinite half-life values ----
  # Only if 'half_life' column is available in the input data
  if ("half_life" %in% names(df)) {
    n_inf <- sum(is.infinite(df$half_life), na.rm = TRUE)
    extra_rows[["n_inf_half_life"]] <- tibble(
      variable = "n_inf_half_life",
      mean = n_inf,
      q025 = NA, q250 = NA, q500 = NA, q750 = NA, q975 = NA,
      group = group_name
    )
  }

  # ---- Step 5: Combine core and additional summaries into single data frame ----
  bind_rows(core_summary, bind_rows(extra_rows))
}


#' Format Value with interquartile range
#'
#' Works with any probability levels; prints all rows by default
#'
#' @param x data.frame with column names same as the output from summarize_quantiles
#' @param digits Number of decimal places
#' @param row Row numbers (defaults to all rows)
#' @return Character vector with formatted strings for each row
format_median_iqr <- function(x, digits = 1, row = NULL) {
  if (is.null(row)) row <- seq_len(nrow(x))
  paste0(
    format_num(x$q500[row], digits = digits), " [IQR: ",
    format_num(x$q250[row], digits = digits), " - ",
    format_num(x$q750[row], digits = digits), "]"
  )
}

format_median_interval <- function(x, digits = 1, nrow = 1, interval = "95% PI") {
  output <- character(nrow)

  for (i in 1:nrow) {
    if (interval == "IQR") {
      output[i] <- paste0(round(x$q500[i], digits), " [IQR: ",
                          round(x$q250[i], digits), ", ",
                          round(x$q750[i], digits), "]")
    } else if (interval == "95% PI") {
      output[i] <- paste0(round(x$q500[i], digits), " [95% PI: ",
                          round(x$q025[i], digits), ", ",
                          round(x$q975[i], digits), "]")
    }
  }

  return(output)
}

compute_yld <- function(cases, parms=NULL, option=1) {
  if (is.null(parms)) {
    stop("Parameters to compute YLD must be provided")
  }

  # Extract needed parameters
  pmild <- parms[parms$Parameter == "Prop_Mild", "Value"]
  pm <- parms[parms$Parameter == "Prop_Moderate", "Value"]
  ps <- parms[parms$Parameter == "Prop_Severe", "Value"]
  dur <- parms[parms$Parameter == "Duration_Illness", "Value"]
  wa <- parms[parms$Parameter == "Disability_Weight_Asymptomatic", "Value"]
  wmild <- parms[parms$Parameter == "Disability_Weight_Mild", "Value"]
  wm <- parms[parms$Parameter == "Disability_Weight_Moderate", "Value"]
  ws <- parms[parms$Parameter == "Disability_Weight_Severe", "Value"]

  # Calculate weighted disability weight based on option
  dis_wt_tot <- if (option == 1) {
    # Option 1: Assumes that reported cases include only moderate and severe cases only
    pm/(pm+ps)*wm + ps/(pm+ps)*ws  # ~0.2164
  } else {
    # Option 2: Assumes that reported cases include mild, moderate and severe cases
    pmild/(pmild+pm+ps)*wmild + pm/(pmild+pm+ps)*wm + ps/(pmild+pm+ps)*ws  # ~0.1780
  }

  # Calculate YLD
  yld_per_capita <- (dur/365) * dis_wt_tot
  yld_tot <- cases * yld_per_capita

  return(yld_tot)
}

#' Get Life Expectancy
#'
#' Retrieves life expectancy value for a specific country, year and age
#'
#' @param life_exp_data Life expectancy data
#' @param age Age in years
#' @param data Data frame with country and date information
#' @return Life expectancy value
get_life_expectancy <- function(life_exp_data, age, data) {
  life_exp <- life_exp_data %>%
    dplyr::filter(`ISO3 Alpha-code` == data$country,
                  Year == as.character(data.table::year(data$date))) %>%
    dplyr::select(as.character(age)) %>%
    as.numeric()

  return(life_exp)
}

#' Compute Years of Life Lost (YLL)
#'
#' Calculates years of life lost due to premature mortality
#'
#' @param deaths Number of deaths
#' @param life_exps Life expectancy values
#' @param parms Parameter data frame
#' @return Total YLL
compute_yll <- function(deaths, life_exps=NULL, parms=NULL) {
  # Get discount rate
  dr <- parms[parms$Parameter == "Rate_Discount", "Value"]

  # Calculate YLL with discounting
  yll <- deaths * (1/dr) * (1 - exp(-dr * life_exps))

  return(yll)
}

#==============================================================================
# SECTION 8: VACCINE IMPACT SUMMARY FUNCTIONS
#==============================================================================

#' Summarize Vaccine Impact Over Weeks
#'
#' Aggregates weekly impact data to the outbreak level
#'
#' @param d Data frame with weekly impact data
#' @param case_trigger Whether vaccination is triggered by case count
#' @return Data frame with outbreak-level summary
sum_over_week <- function(d, case_trigger=FALSE) {
  # Define grouping variables based on case_trigger
  by_cols <- if (case_trigger) {
    c("runid", "id_outbreak", "vacc_week", "case_trigger", "vacc_cov")
  } else {
    c("runid", "id_outbreak", "vacc_week", "vacc_cov")
  }

  # Aggregate data
  d[, .(
    country = first(country),
    year = first(year),
    pop = first(pop),
    ori_occurred = first(ori_occurred),
    week_delay_to_vacc_effect = first(week_delay_to_vacc_effect),
    # Sum across weeks for weekly variables
    s_ch_tot = sum(s_ch),
    s_ch_averted_tot = sum(s_ch_averted),
    c_ch_tot = sum(c_ch, na.rm=TRUE),
    death_tot = sum(deaths, na.rm=TRUE),
    death_averted_tot = sum(deaths_averted, na.rm=TRUE),
    # Calculate percentage reductions
    pct_reduc_death = 100 * sum(deaths_averted, na.rm=TRUE)/sum(deaths, na.rm=TRUE),
    pct_reduc_case = 100 * sum(s_ch_averted)/sum(s_ch)),
    by = by_cols]
}

#' Calculate Mean Across Model Runs
#'
#' Computes mean impact values across simulation runs
#'
#' @param d Data frame with impact data from multiple runs
#' @param case_trigger Whether vaccination is triggered by case count
#' @return Data frame with mean values
mean_across_runid <- function(d, case_trigger=FALSE) {
  # Define grouping variables based on case_trigger
  by_cols <- if (case_trigger) {
    c("id_outbreak", "case_trigger", "vacc_cov")
  } else {
    c("id_outbreak", "vacc_week", "vacc_cov")
  }

  # Calculate means
  d[, .(
    country = first(country),
    year = first(year),
    pop = first(pop),
    ori_occurred = first(ori_occurred),
    week_delay_to_vacc_effect = first(week_delay_to_vacc_effect),
    # Mean across runids
    c_ch_tot = mean(c_ch_tot, na.rm=TRUE),
    s_ch_tot = mean(s_ch_tot),
    s_ch_averted_tot = mean(s_ch_averted_tot),
    pct_reduc_case = mean(pct_reduc_case),
    death_tot = mean(death_tot, na.rm=TRUE),
    death_averted_tot = mean(death_averted_tot, na.rm=TRUE),
    pct_reduc_death = mean(pct_reduc_death, na.rm=TRUE)),
    by = by_cols]
}

#' Calculate Standard Deviation Across Model Runs
#'
#' Computes standard deviation of impact values across simulation runs
#'
#' @param d Data frame with impact data from multiple runs
#' @param case_trigger Whether vaccination is triggered by case count
#' @return Data frame with standard deviations
sd_across_runid <- function(d, case_trigger=FALSE) {
  # Define grouping variables based on case_trigger
  by_cols <- if (case_trigger) {
    c("id_outbreak", "case_trigger", "vacc_cov")
  } else {
    c("id_outbreak", "vacc_week", "vacc_cov")
  }

  # Calculate standard deviations
  d[, .(
    country = first(country),
    year = first(year),
    pop = first(pop),
    ori_occurred = first(ori_occurred),
    week_delay_to_vacc_effect = first(week_delay_to_vacc_effect),
    # Standard deviation across runs
    c_ch_tot = sd(c_ch_tot, na.rm=TRUE),
    s_ch_tot = sd(s_ch_tot),
    s_ch_averted_tot = sd(s_ch_averted_tot),
    pct_reduc_case = sd(pct_reduc_case),
    death_tot = sd(death_tot, na.rm=TRUE),
    death_averted_tot = sd(death_averted_tot, na.rm=TRUE),
    pct_reduc_death = sd(pct_reduc_death, na.rm=TRUE)),
    by = by_cols]
}

#' Add CEA Variables to Results
#'
#' Adds cost-effectiveness analysis variables to the modeling results
#'
#' @param d Data frame with vaccination impact results
#' @return Data frame with added variables
add_cea_variables <- function(d, mean_age_inf=NULL) {
  # Add life expectancy data
  d <- left_join(
    d,
    life_exp_data[, c("country", "year", as.character(mean_age_inf))],
    by = c("country", "year")
  )

  names(d)[names(d) == as.character(mean_age_inf)] <- "life_exp"
  d$life_exp <- as.numeric(d$life_exp)

  # Add GDP and workforce percentage data
  d <- left_join(
    d,
    gdp_long[, c("country", "year", "gdp")],
    by = c("country", "year")
  ) %>%
    left_join(
      workforce_long[, c("country", "year", "pct_workforce")],
      by = c("country", "year")
    )

  return(d)
}

#' Add Cost-Effectiveness Analysis Results
#'
#' Calculates health and economic outcomes for cost-effectiveness analysis
#'
#' @param d Data frame with vaccination impact results and CEA variables
#' @return Data frame with CEA results
add_cea_results <- function(d) {
  d %>%
    mutate(
      # Health outcomes
      yld = compute_yld(s_ch_tot, parms=parms),
      yld_averted = compute_yld(s_ch_averted_tot, parms=parms),
      yll = compute_yll(death_tot, life_exp, parms),
      yll_averted = compute_yll(death_averted_tot, life_exp, parms),
      daly_averted = yld_averted + yll_averted,

      # Economic outcomes
      coi_averted = s_ch_averted_tot * coi_per_patient,
      cod_averted = death_averted_tot * gdp * life_exp,
      productivity_lost_averted =
        s_ch_averted_tot * gdp * ((patient_workday_lost/365)*(pct_workforce/100) +
                                    (caregiver_workday_lost/365)),

      # Vaccine costs
      vacc_dose = ifelse(ori_occurred, pop * vacc_cov * dose_regimen, 0),
      vacc_cost = (vacc_cost_per_dose + vacc_delivery_cost) * vacc_dose,
      net_cost = vacc_cost - coi_averted - cod_averted - productivity_lost_averted,

      # Cost-effectiveness ratios
      cost_per_daly_averted =
        ifelse(daly_averted > 0, net_cost / daly_averted, NA),
      cost_per_case_averted =
        ifelse(s_ch_averted_tot > 0, net_cost / s_ch_averted_tot, NA),
      cost_per_death_averted =
        ifelse(death_averted_tot > 0, net_cost / death_averted_tot, NA),

      # Impact per 1000 vaccine doses
      case_averted_per_1000_OCV =
        ifelse(vacc_dose > 0, 1000 * s_ch_averted_tot / vacc_dose, NA),
      death_averted_per_1000_OCV =
        ifelse(vacc_dose > 0, 1000 * death_averted_tot / vacc_dose, NA),

      # Classification of cost-effectiveness
      cost_eff_threshold = 3 * gdp,
      ratio_cost_per_daly_averted_to_gdp =
        ifelse(gdp > 0, cost_per_daly_averted / gdp, NA),
      cost_effective =
        ifelse(cost_per_daly_averted > cost_eff_threshold, FALSE, TRUE)
    )
}

#' Summarize Results Across Outbreaks by Run ID
#'
#' Aggregates impact metrics across multiple outbreaks
#'
#' @param d Data frame with outbreak-level results
#' @param case_trigger Whether vaccination is triggered by case count
#' @return Data frame with aggregated metrics by run
sum_over_outbreaks_by_runid <- function(d, ori_occurred=TRUE, case_trigger=FALSE) {
  # Define grouping variables based on case_trigger
  by_cols <- if (case_trigger) {
    c("runid", "case_trigger", "vacc_cov")
  } else {
    c("runid", "vacc_week", "vacc_cov")
  }

  if (ori_occurred) d <- d[ori_occurred == TRUE, ]
  # Aggregate across outbreaks
  d[, .(
    n_outbreaks = .N,                          # rows per group
    pop_tot = sum(pop),
    vacc_dose = sum(ifelse(ori_occurred, pop * vacc_cov, 0)),

    # Sum across outbreaks
    c_ch_tot = sum(c_ch_tot, na.rm=TRUE),
    death_tot = sum(death_tot, na.rm=TRUE),
    death_averted_tot = sum(death_averted_tot, na.rm=TRUE),
    pct_reduc_death = 100 * sum(death_averted_tot, na.rm=TRUE)/sum(death_tot, na.rm=TRUE),
    s_ch_tot = sum(s_ch_tot),
    s_ch_averted_tot = sum(s_ch_averted_tot),
    pct_reduc_case = 100 * sum(s_ch_averted_tot) / sum(s_ch_tot),

    # Calculate impact per 1000 doses
    case_averted_per_1000_OCV =
      1000 * sum(s_ch_averted_tot) / sum(vacc_dose),
    death_averted_per_1000_OCV =
      1000 * sum(death_averted_tot, na.rm=TRUE) /sum(vacc_dose),

    # Calculate cost-effectiveness ratios
    cost_per_daly_averted = sum(net_cost, na.rm=TRUE) / sum(daly_averted, na.rm=TRUE),
    cost_per_case_averted = sum(net_cost, na.rm=TRUE) / sum(s_ch_averted_tot, na.rm=TRUE),
    cost_per_death_averted = sum(net_cost, na.rm=TRUE) / sum(death_averted_tot, na.rm=TRUE)),
    by = by_cols]
}

#' Add CEA Results for All Outbreaks
#'
#' Calculates cost-effectiveness metrics across all outbreaks
#'
#' @param svim Data frame with vaccination impact results
#' @return Data frame with added CEA variables
add_cea_results_all_outbreaks <- function(svim) {
  # Add CEA variables (ratio measures will be computed later when summing across outbreaks)
  svim %>%
    mutate(
      # Health outcomes
      yld = compute_yld(s_ch_tot, parms=parms),
      yld_averted = compute_yld(s_ch_averted_tot, parms=parms),
      yll = compute_yll(death_tot, life_exp, parms),
      yll_averted = compute_yll(death_averted_tot, life_exp, parms),
      daly_averted = yld_averted + yll_averted,

      # Economic outcomes
      coi_averted = s_ch_averted_tot * coi_per_patient,
      cod_averted = death_averted_tot * gdp * life_exp,
      productivity_lost_averted =
        s_ch_averted_tot * gdp * ((patient_workday_lost/365)*(pct_workforce/100) +
                                    (caregiver_workday_lost/365)),

      # Vaccine costs
      vacc_dose = pop * vacc_cov * dose_regimen,
      vacc_cost = (vacc_cost_per_dose + vacc_delivery_cost) * vacc_dose,
      net_cost = vacc_cost - coi_averted - cod_averted - productivity_lost_averted
    )
}



#==============================================================================
# Random-only outbreak sampler with dose budgeting
#==============================================================================

#' Randomly sample outbreaks under an OCV dose budget and/or a count target
#'
#' At each draw, only outbreaks with population <= (remaining ocv doses / target_coverage)
#' are eligible. Selected outbreaks consume `target_coverage * pop` doses.
#'
#' @param outbreak_data Data frame; must include column `pop` (numeric, outbreak population)
#' @param target_outbreak_count Integer >=1 or NULL; number of outbreaks to select (optional)
#' @param target_ocv_doses Numeric >0 or NULL; total ocv doses available (optional)
#' @param target_coverage Numeric in (0,1]; assumed coverage applied to every selected outbreak
#' @param seed Integer or NULL; random seed for reproducibility
#' @param max_attempts Integer; safety cap on iterations
#'
#' @return List with sampled_data, indices, dose bookkeeping, and stopping_criterion
sample_outbreaks <- function(outbreak_data,
                             target_outbreak_count = NULL,
                             target_ocv_doses = NULL,
                             target_coverage = 1,
                             seed = NULL,
                             max_attempts = 1000) {

  # ---- Checks ----
  if (!"pop" %in% names(outbreak_data))
    stop("`outbreak_data` must contain a numeric `pop` column.")
  if (!is.numeric(outbreak_data$pop) || any(outbreak_data$pop < 0, na.rm = TRUE))
    stop("`pop` must be non-negative numeric.")
  if (is.null(target_outbreak_count) && is.null(target_ocv_doses))
    stop("Specify at least one of `target_outbreak_count` or `target_ocv_doses`.")
  if (!is.null(target_outbreak_count) && (!is.finite(target_outbreak_count) || target_outbreak_count <= 0))
    stop("`target_outbreak_count` must be a positive integer or NULL.")
  if (!is.null(target_ocv_doses) && (!is.finite(target_ocv_doses) || target_ocv_doses <= 0))
    stop("`target_ocv_doses` must be a positive number or NULL.")
  if (!is.numeric(target_coverage) || target_coverage <= 0 || target_coverage > 1)
    stop("`target_coverage` must be in (0, 1].")
  if (!is.null(seed)) set.seed(seed)

  # Defaults
  if (is.null(target_outbreak_count)) target_outbreak_count <- Inf
  dose_mode <- !is.null(target_ocv_doses)
  doses_remaining <- if (dose_mode) target_ocv_doses else Inf
  doses_used <- 0

  # ---- State ----
  n <- nrow(outbreak_data)
  available_indices <- seq_len(n)
  sampled_indices <- integer(0)
  k <- 0L
  attempts <- 0L

  # ---- Helper: eligible indices given remaining doses ----
  eligible_pool <- function() {
    idx <- available_indices
    if (dose_mode) {
      if (doses_remaining <= 0) return(integer(0))
      max_pop <- floor(doses_remaining / target_coverage)
      idx <- idx[outbreak_data$pop[idx] <= max_pop]
    }
    idx
  }

  # ---- Loop ----
  while (k < target_outbreak_count &&
         length(available_indices) > 0 &&
         attempts < max_attempts) {

    candidates <- eligible_pool()
    if (length(candidates) == 0) break

    # Uniform random among eligible
    pick <- if (length(candidates) == 1) candidates else sample(candidates, 1L)

    # Dose accounting (should always fit because of eligibility filter)
    needed <- target_coverage * outbreak_data$pop[pick]
    if (!dose_mode || needed <= doses_remaining) { # it's redundant but keep it for now
      sampled_indices <- c(sampled_indices, pick)
      k <- k + 1L
      if (dose_mode) {
        doses_remaining <- doses_remaining - needed
        doses_used <- doses_used + needed
      }
      # stop if targets met
      if (k >= target_outbreak_count || (dose_mode && doses_remaining <= 0)) {
        available_indices <- setdiff(available_indices, pick)
        break
      }
    }

    # Remove picked index (no replacement)
    available_indices <- setdiff(available_indices, pick)
    attempts <- attempts + 1L
  }

  if (length(sampled_indices) == 0)
    stop("No outbreaks could be sampled within the constraints. Check doses/coverage or data.")

  stopping_criterion <- if (k >= target_outbreak_count) {
    "target_outbreak_count"
  } else if (dose_mode && doses_remaining <= 0) {
    "target_ocv_doses"
  } else if (attempts >= max_attempts) {
    "max_attempts"
  } else {
    "no_more_eligible"
  }

  list(
    sampled_data        = outbreak_data[sampled_indices, , drop = FALSE],
    sampled_indices     = sampled_indices,
    outbreak_count      = k,
    target_outbreak_count = if (is.infinite(target_outbreak_count)) NULL else target_outbreak_count,
    target_ocv_doses    = if (dose_mode) target_ocv_doses else NULL,
    target_coverage     = target_coverage,
    doses_used          = if (dose_mode) doses_used else NULL,
    doses_remaining     = if (dose_mode) doses_remaining else NULL,
    n_samples           = length(sampled_indices),
    success             = k > 0,
    stopping_criterion  = stopping_criterion
  )
}



# Helper function for concise NULL replacement (available in rlang `||`)
`%||%` <- function(a, b) if (is.null(a)) b else a





#==============================================================================
# SECTION 10: FORMATTING AND VISUALIZATION FUNCTIONS
#==============================================================================

#' Format Numbers with Thousands Separators
#'
#' @param x Numeric value to format
#' @param digits Number of decimal places
#' @param big.mark Character to use as thousands separator
#' @return Formatted string
format_num <- function(x, digits=0, big.mark=",") {
  # If big.mark is NULL or "", don't use thousand separator
  if (is.null(big.mark) || big.mark == "") {
    return(format(round(x, digits=digits), trim=TRUE))
  }
  # Otherwise use the specified big.mark
  format(round(x, digits=digits), big.mark=big.mark, trim=TRUE)
}

#' Format Median with 95% Confidence Interval
#'
#' @param x Quantile vector with 2.5%, 50%, and 97.5% values
#' @param digits Number of decimal places
#' @return Formatted string with median and 95% CI
format_median_with_95 <- function(x, digits) {
  median <- format_num(x[["50%"]], digits=digits)
  lb <- format_num(x[["2.5%"]], digits=digits)
  ub <- format_num(x[["97.5%"]], digits=digits)
  paste0(median, " (", lb , " - ", ub, ")")
}

#' Format Value with Confidence Intervals
#'
#' More general version of format_median_with_95 that works with any probability levels
#'
#' @param x Quantile vector
#' @param digits Number of decimal places
#' @param probs Probability levels
#' @param big.mark Character to use as thousands separator
#' @return Formatted string with value and confidence interval
format_with_intervals <- function(x, digits, probs, big.mark=",") {
  # If only one probability is provided, return just that value
  if (length(probs) == 1) {
    prob <- paste0(probs[1] * 100, "%")
    return(format_num(x[[prob]], digits=digits, big.mark=big.mark))
  }

  # For multiple probabilities, get middle, lower and upper bounds
  middle_idx <- ceiling(length(probs)/2)
  median_prob <- paste0(probs[middle_idx] * 100, "%")
  lb_prob <- paste0(probs[1] * 100, "%")
  ub_prob <- paste0(probs[length(probs)] * 100, "%")

  # Format values
  median <- format_num(x[[median_prob]], digits=digits, big.mark=big.mark)
  lb <- format_num(x[[lb_prob]], digits=digits, big.mark=big.mark)
  ub <- format_num(x[[ub_prob]], digits=digits, big.mark=big.mark)

  paste0(median, " (", lb , " - ", ub, ")")
}


#' Group Outbreaks by Size
#'
#' Groups outbreaks into size categories for analysis
#'
#' @param x Data frame with outbreak information including size_grp
#' @return List of data frames grouped by size
get_size_grp <- function(x) {
  lst <- vector("list", 10)
  for(i in 1:length(lst)) {
    lst[[i]] <- x[x$size_grp >= i,]
  }
  return(lst)
}

#' Get Group ID
#'
#' Assigns a group ID based on cutoff values
#'
#' @param x Value to classify
#' @param cutoff Vector of cutoff values
#' @return Group ID
get_group_id <- function(x, cutoff) {
  if (is.na(x)) return(NA)

  for (i in (length(cutoff)-1):1) {
    if (x >= cutoff[i]) {
      return(i)
    }
  }
  return(NA)
}

#' Add Summary Statistics
#'
#' Adds outbreak characteristics and groups outbreaks by size, duration, and attack rate
#'
#' @param d Data frame with outbreak results
#' @param case_trigger Whether vaccination is triggered by case count
#' @return Data frame with added summary statistics
add_summary_stats <- function(d, case_trigger=FALSE) {
  # Join outbreak characteristics
  d <- left_join(
    d,
    ds[, c("location", "spatial_scale", "outbreak_pop",
           "weekly_threshold_per_10e5", "total_s_ch", "total_c_ch",
           "total_deaths", "duration_weeks", "peak_week_incidence",
           "time_to_peak_week", "peak_week_incidence_per_100percent",
           "peak_week_incidence_per_10e3", "mean_r0_1st_week",
           "area_per_1km2", "ar_per_10e3", "cfr_per_100percent",
           "id_outbreak")],
    by="id_outbreak"
  )

  # Calculate percentile cutoffs
  probs <- seq(0, 1, by=0.1)
  ar_grp_cutoff <- quantile(ds$ar_per_10e3, probs=probs)

  # Extract one vaccination coverage for calculations
  vc <- unique(svim$vacc_cov)[1]

  # Create size and duration groups based on scenario
  if (!case_trigger) {
    vw <- unique(svim$vacc_week)[1]
    subset_data <- svim[svim$vacc_week == vw & svim$vacc_cov == vc, ]
  } else {
    ct <- unique(svim$case_trigger)[1]
    subset_data <- svim[svim$case_trigger == ct & svim$vacc_cov == vc, ]
  }

  size_grp_cutoff <- quantile(subset_data$s_ch_tot, probs=probs)
  dur_grp_cutoff <- quantile(subset_data$outbk_dur, probs=probs, na.rm=TRUE)

  # Assign group IDs
  svim$size_grp <- sapply(svim$s_ch_tot, function(x) get_group_id(x, size_grp_cutoff))
  svim$dur_grp <- sapply(svim$outbk_dur, function(x) get_group_id(x, dur_grp_cutoff))
  svim$ar_grp <- sapply(svim$ar_per_10e3, function(x) get_group_id(x, ar_grp_cutoff))

  return(d)
}

#' Summarize Impact Metrics
#'
#' Creates a summary table of key impact metrics with confidence intervals
#'
#' @param d Data frame with impact results
#' @param nrow Number of rows in the output table
#' @param probs Probability levels for confidence intervals
#' @param big.mark Character to use as thousands separator
#' @return Data frame with summarized impact metrics
impact_summary <- function(d, nrow=1,
                           probs=c(0.025, 0.5, 0.975),
                           big.mark=",") {
  # Validate probabilities
  if (!all(probs >= 0 & probs <= 1)) {
    stop("All probabilities must be between 0 and 1")
  }
  if (!all(diff(probs) >= 0)) {
    probs <- sort(probs)
    warning("Probabilities were not in ascending order. They have been sorted.")
  }

  # Create empty output table
  # metrics <- c("PCR", "CA", "PDR", "DA", "CAPD", "DAPD", "CPCA", "CPDA", "ICER")
  metrics <- c("PCR", "CA", "DA", "CAPD", "DAPD", "CPCA", "CPDA", "ICER")
  tab <- matrix(NA, nrow=nrow, ncol=length(metrics)) %>%
    as.data.frame() %>%
    `colnames<-`(metrics)

  # Define columns to summarize and their formatting parameters
  col_specs <- list(
    list(col="pct_reduc_case", out="PCR", digits=1),
    list(col="s_ch_averted_tot", out="CA", digits=0),
    # list(col="pct_reduc_death", out="PDR", digits=1),
    list(col="death_averted_tot", out="DA", digits=0),
    list(col="pct_reduc_death", out="PDR", digits=1),
    list(col="case_averted_per_1000_OCV", out="CAPD", digits=3),
    list(col="death_averted_per_1000_OCV", out="DAPD", digits=3),
    list(col="cost_per_case_averted", out="CPCA", digits=1),
    list(col="cost_per_death_averted", out="CPDA", digits=1),
    list(col="cost_per_daly_averted", out="ICER", digits=1)
  )

  # Fill table with formatted results
  for (i in 1:nrow) {
    for (spec in col_specs) {
      quantiles <- quantile(d[[spec$col]], probs=probs, na.rm=TRUE)
      tab[i, spec$out] <- format_with_intervals(quantiles, digits=spec$digits,
                                                probs=probs, big.mark=big.mark)
    }
  }

  return(tab)
}

#' Summarize Vaccine Impact Across All Outbreaks
#'
#' Creates a comprehensive summary of vaccination impact metrics
#'
#' @param d Data frame with impact results
#' @param case_trigger Whether vaccination is triggered by case count
#' @return Data frame with summary statistics
svim_summary_all_outbreaks <- function(d, case_trigger=FALSE) {
  # Define grouping variables based on case_trigger
  by_cols <- if (case_trigger) {
    c("case_trigger", "vacc_cov")
  } else {
    c("vacc_week", "vacc_cov")
  }

  # Define metrics to summarize
  metrics <- c(
    "s_ch_averted_tot", "death_averted_tot",
    "case_averted_per_1000_OCV", "death_averted_per_1000_OCV",
    "pct_reduc_case", "pct_reduc_death",
    "dur_reduced", "pct_reduc_dur"
  )

  # Function to create summary stats for a metric
  summarize_metric <- function(data, metric) {
    c(
      mean = mean(data[[metric]], na.rm=TRUE),
      q500 = quantile(data[[metric]], probs=0.5, na.rm=TRUE),
      q025 = quantile(data[[metric]], probs=0.025, na.rm=TRUE),
      q975 = quantile(data[[metric]], probs=0.975, na.rm=TRUE)
    )
  }

  # Calculate summary stats for each group and metric
  d %>%
    group_by(across(all_of(by_cols))) %>%
    summarize(
      pop_tot = first(pop_tot),
      s_ch_tot = first(s_ch_tot),
      c_ch_tot = first(c_ch_tot),
      death_tot = first(death_tot),
      outbk_dur = first(outbk_dur),
      mean_vacc_dose = mean(vacc_dose),

      # Generate statistics for each metric
      across(
        all_of(metrics),
        list(
          mean = ~mean(., na.rm=TRUE),
          q500 = ~quantile(., probs=0.5, na.rm=TRUE),
          q025 = ~quantile(., probs=0.025, na.rm=TRUE),
          q975 = ~quantile(., probs=0.975, na.rm=TRUE)
        )
      ),
      .groups = "drop"
    )
}

#' Extract Administrative Names from ID
#'
#' Parses outbreak ID to extract administrative unit names
#'
#' @param x Vector of outbreak IDs
#' @return List of administrative unit names
get_adm <- function(x) {
  lapply(x, function(z) grep("[a-zA-Z+]", strsplit(z, "::|-")[[1]], value=TRUE))
}

#==============================================================================
# SECTION 11: THEMING FUNCTIONS FOR PLOTS
#==============================================================================

#' Theme for Tight Plot Layouts
#'
#' Reduces margins and legend spacing for compact plots
#'
#' @return ggplot2 theme object
theme_tight <- function() {
  theme(
    legend.box.spacing = unit(0, "pt"),
    legend.margin = margin(0,0,0,0),
    legend.position="top",
    plot.margin = margin(2,2,2,6),
    legend.key.size = unit(0.5, "cm"),
    legend.key.spacing = unit(0.1, "cm")
  )
}

#' Theme to Remove X-Axis Elements
#'
#' Removes x-axis text, ticks and title
#'
#' @return ggplot2 theme object
theme_x_blank <- function() {
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.title.x = element_blank()
  )
}

#' Theme to Remove Y-Axis Elements
#'
#' Removes y-axis text, ticks and title
#'
#' @return ggplot2 theme object
theme_y_blank <- function() {
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.title.y = element_blank()
  )
}

#' Theme for Tight Legend Layout
#'
#' Reduces spacing in legend for compact plots
#'
#' @return ggplot2 theme object
theme_tight_legend <- function() {
  theme(
    legend.box.spacing = unit(0, "pt"),
    legend.margin = margin(0, 0, 0, 0),
    legend.key.size = unit(0.5, "cm"),
    legend.key.spacing = unit(0.1, "cm")
  )
}

# # Create helper theme function for tight legend spacing
# theme_tight_legend <- function() {
#   theme(
#     legend.spacing.x = unit(0.1, "cm"),
#     legend.spacing.y = unit(0.1, "cm"),
#     legend.margin = margin(0, 0, 0, 0),
#     legend.key.size = unit(0.5, "lines")
#   )
# }

#' Quantile whiskers + median point-range + mean "X" point
#'
#' Adds three layers:
#'  (1) median with 2.5%–97.5% range (pointrange)
#'  (2) 25%–75% interquartile errorbar (no caps)
#'  (3) mean as an "X" point, optionally grouped/colored
#'
#' @param group Optional grouping variable (bare name).
#' @param color Optional color variable (bare name). Often same as group.
#' @param dodge_width Width for position_dodge().
#' @param lw95 Linewidth for the 95% range.
#' @param lw50 Linewidth for the IQR bar.
#' @param pr_size Size for the median pointrange point.
#' @param mean_size Size for the mean "X" point.
#' @param mean_stroke Stroke for the mean "X" point.
#' @return list of ggplot2 layers
#' @examples
#' ggplot(df, aes(x, y)) + ci_layers(group = grp, color = grp)
ci_layers <- function(group = NULL, color = NULL,
                      dodge_width = 0.5,
                      lw95 = 0.6,
                      lw50 = 0.9,
                      pr_size = 1.5,
                      mean_size = 3,
                      mean_stroke = 0.8) {

  # capture quosures to test presence
  gq <- rlang::enquo(group)
  cq <- rlang::enquo(color)
  has_group <- !rlang::quo_is_null(gq)
  has_color <- !rlang::quo_is_null(cq)

  # build mapping for the mean layer conditionally
  mean_mapping <- if (has_group && has_color) {
    ggplot2::aes(group = as.factor(!!gq), color = as.factor(!!cq))
  } else if (has_group && !has_color) {
    ggplot2::aes(group = as.factor(!!gq))
  } else if (!has_group && has_color) {
    ggplot2::aes(color = as.factor(!!cq))
  } else {
    NULL
  }

  list(
    ggplot2::stat_summary(
      fun = median,
      fun.min = ~stats::quantile(.x, 0.025),
      fun.max = ~stats::quantile(.x, 0.975),
      geom = "pointrange",
      position = ggplot2::position_dodge(width = dodge_width),
      linewidth = lw95,
      size = pr_size
    ),
    ggplot2::stat_summary(
      fun.min = ~stats::quantile(.x, 0.25),
      fun.max = ~stats::quantile(.x, 0.75),
      geom = "errorbar",
      position = ggplot2::position_dodge(width = dodge_width),
      linewidth = lw50,
      width = 0
    ),
    ggplot2::stat_summary(
      mapping = mean_mapping,
      fun = mean,
      geom = "point",
      shape = 4,  # "X"
      size = mean_size,
      stroke = mean_stroke,
      position = ggplot2::position_dodge(width = dodge_width)
    )
  )
}



#' Quantile whiskers + median + mean "X" for PRE-SUMMARIZED data
#' Expects columns: q025,q250,q500,q750,q975,mean
#' Assumes aesthetics are mapped in ggplot(), e.g.,
#'   ggplot(df, aes(x = vacc_week, color = factor(vacc_cov), group = factor(vacc_cov)))
ci_layers_from_summary <- function(dodge_width = 0.5,
                                   lw95 = 0.6,
                                   lw50 = 0.9,
                                   pr_size = 1.5,
                                   mean_size = 3,
                                   mean_stroke = 0.8) {
  list(
    # 95% central interval
    geom_linerange(aes(ymin = q025, ymax = q975),
                   position = position_dodge(width = dodge_width),
                   linewidth = lw95),
    # 50% IQR
    geom_linerange(aes(ymin = q250, ymax = q750),
                   position = position_dodge(width = dodge_width),
                   linewidth = lw50),
    # median (dot)
    geom_point(aes(y = q500),
               position = position_dodge(width = dodge_width),
               size = pr_size),
    # mean as "X"
    geom_point(aes(y = mean),
               position = position_dodge(width = dodge_width),
               shape = 4, size = mean_size, stroke = mean_stroke)
  )
}

#' Add summary braces and markers for mean/median with uncertainty ranges
#'
#' This function creates a list of ggplot2 layers showing mean and median points,
#' 50% and 95% uncertainty intervals with braces, vertical whiskers, and labels.
#' It is designed to be added to a ggplot2 pipeline using \code{+}.
#'
#' @param x0 Numeric scalar. Anchor x-position for the mean/median points and whiskers.
#' @param ymean Numeric scalar. Value of the mean.
#' @param ymedian Numeric scalar. Value of the median.
#' @param y50 Numeric length-2 vector. Lower and upper bounds of the 50% interval.
#' @param y95 Numeric length-2 vector. Lower and upper bounds of the 95% interval.
#' @param dx50 Numeric scalar. Horizontal spacing allocated for the 50% brace (default = 1).
#' @param dx95 Numeric scalar. Horizontal spacing allocated for the 95% brace (default = 1).

#' @param label_size Numeric scalar. Text size for labels (default = 3.2).
#' @param pt_size_median Numeric scalar. Point size for the median marker (default = 2.8).
#' @param pt_size_mean Numeric scalar. Point size for the mean marker (default = 3.2).
#' @param pt_stroke_mean Numeric scalar. Stroke width for the mean marker (default = 0.9).
#' @param lw_50,lw_95 Numeric scalars. Line widths for the 50% and 95% braces.
#' @param whisker_50,whisker_95 Numeric scalars. Line widths for the vertical whiskers.
#' @param text_offset_mean Numeric scalar. Horizontal offset for mean/median labels (default = 0.1).
#' @param brace_offset Numeric scalar. Horizontal offset for positioning the braces (default = 0.24).
#'
#' @return A list of ggplot2 layers that can be added to an existing plot.
#' @examples
#' library(ggplot2)
#' ggplot() +
#'   add_summary_braces(
#'     x0 = 1,
#'     ymean = 10,
#'     ymedian = 9,
#'     y50 = c(8, 11),
#'     y95 = c(6, 13)
#'   )
#'
#' @import ggplot2
#' @importFrom ggbrace stat_brace
#' @export
add_summary_braces <- function(
    x0,                      # anchor x for the points/whiskers (e.g., 11)
    ymean, ymedian,          # scalars
    y50, y95,                # length-2 vectors: c(lower, upper)
    dx50 = 1, dx95 = 1,      # horizontal step sizes for the 50% and 95% braces
    label_size = 3.2,
    pt_size_median = 2.8,
    pt_size_mean = 3.2,
    pt_stroke_mean = 0.9,
    lw_50 = 0.6, lw_95 = 0.6,# brace line widths
    whisker_50 = 0.6, whisker_95 = 0.6,  # vertical segment widths
    text_offset_mean = 0.2, # label offsets from x0
    text_50_x = 0.2, # label offsets from x0
    text_50_y = 0.2, # label offsets from x0
    text_95_x = 0.2, # label offsets from x0
    text_95_y = 0.2,
    brace_50_x = 0.2,
    brace_95_x = 0.2
) {
  # data frames for geoms (keeps inherit.aes = FALSE clean)
  df_median <- data.frame(x = x0, y = ymedian)
  df_mean   <- data.frame(x = x0, y = ymean)

  df_w50 <- data.frame(x = x0, xend = x0, y = y50[1], yend = y50[2])
  df_w95 <- data.frame(x = x0, xend = x0, y = y95[1], yend = y95[2])

  df_b50 <- data.frame(x = c(brace_50_x, brace_50_x + dx50),
                       y = c(y50[1], y50[2]))
  df_b95 <- data.frame(x = c(brace_95_x, brace_95_x + dx95),
                       y = c(y95[1], y95[2]))

  list(
    # points
    geom_point(
      data = df_median,
      mapping = aes(x = x, y = y),
      shape = 16, size = pt_size_median, inherit.aes = FALSE
    ),
    geom_point(
      data = df_mean,
      mapping = aes(x = x, y = y),
      shape = 4, size = pt_size_mean, stroke = pt_stroke_mean, inherit.aes = FALSE
    ),

    # braces (ggbrace::stat_brace)
    ggbrace::stat_brace(
      data = df_b50,
      mapping = aes(x, y),
      outside = FALSE, rotate = 90, linewidth = lw_50, inherit.aes = FALSE
    ),
    ggbrace::stat_brace(
      data = df_b95,
      mapping = aes(x, y),
      outside = FALSE, rotate = 90, linewidth = lw_95, inherit.aes = FALSE
    ),

    # vertical whiskers
    geom_segment(
      data = df_w95,
      mapping = aes(x = x, xend = xend, y = y, yend = yend),
      linewidth = whisker_95, inherit.aes = FALSE
    ),
    geom_segment(
      data = df_w50,
      mapping = aes(x = x, xend = xend, y = y, yend = yend),
      linewidth = whisker_50, inherit.aes = FALSE
    ),

    # labels
    annotate("text", x = x0 - text_offset_mean, y = ymean,   label = "mean",
             hjust = 1, vjust = 0.5, size = label_size),
    annotate("text", x = x0 - text_offset_mean, y = ymedian, label = "median",
             hjust = 1, vjust = 0.5, size = label_size),
    annotate("text", x = text_50_x, y = text_50_y, label = "50%",
             hjust = 0, vjust = 0.5, size = label_size),
    annotate("text", x = text_95_x, y = text_95_y,
             label = "95%",
             hjust = 0, vjust = 0.5, size = label_size)
  )
}


#==============================================================================
# UTILITY FUNCTIONS FOR CHOLERA VACCINATION IMPACT MODELING
# This file contains utility functions for modeling the impact of
# oral cholera vaccines (OCV), processing data, and analyzing results.
#==============================================================================

#==============================================================================
# SECTION 1: EPIDEMIC MODELING FUNCTIONS
#==============================================================================

#' Calculate Final Epidemic Size for SIR Model
#'
#' Computes the final size of an epidemic using the relationship R = 1 - exp(-R0*R)
#' where R is the final epidemic size for the SIR model
#'
#' @param R0 Basic reproduction number (default: 2)
#' @return Final epidemic size as a proportion of the population
final_epidemic_size <- function(R0 = 2) {
  # Define equation to solve: x - 1 + exp(-R0*x) = 0
  y = function(x) x - 1 + exp(-R0*x)

  # Find the root numerically within (0,1)
  final_size <- uniroot(y, interval=c(1e-6, 1-1e-6))$root

  return(final_size)
}

#==============================================================================
# SECTION 2: STATISTICAL AND MATHEMATICAL UTILITY FUNCTIONS
#==============================================================================

#' Print Parameter Values
#'
#' Outputs parameter names and values in a readable format
#'
#' @param params Named list or vector of parameters
print_params <- function(params) {
  n <- names(params)
  cat(paste0(n, "=", params[n], collapse=", "))
}

#' Expit Function (Logistic Function)
#'
#' Inverse of logit function: transforms values from (-∞,∞) to (0,1)
#'
#' @param x Numeric input
#' @return Value between 0 and 1
expit <- function(x) 1/(1+exp(-x))

#' Logit Function
#'
#' Transforms values from (0,1) to (-∞,∞)
#'
#' @param x Numeric input between 0 and 1
#' @return Transformed value on real line
logit <- function(x) log(x/(1-x))

#' Extract Output Days from Date Range
#'
#' @param x Object containing date_range attribute
#' @return Number of days
get_output_days <- function(x) {
  unitdays = gsub("^([0-9]+).*", "\\1", x$date_range)
  l = length(unique(unitdays))
  if (l != 1) {
    stop("Output days have more than one kind")
  }
  return(as.double(unique(unitdays)))
}

#' Define Discrete Color Palette
#'
#' A set of distinct colors for visualizations
my_discrete_colors <-
  c("dodgerblue2", "#E31A1C", "green4", "#6A3D9A",
    "#FF7F00","black", "gold1", "skyblue2", "palegreen2", "#FDBF6F",
    "gray70", "maroon", "orchid1", "darkturquoise", "darkorange4", "brown")

#' Calculate Beta Distribution Parameters
#'
#' Computes alpha and beta parameters for a Beta distribution
#' given mean and variance
#'
#' @param mu Mean of the distribution
#' @param var Variance of the distribution
#' @return List containing alpha and beta parameters
calc_beta_params <- function(mu, var) {
  alpha <- ((1 - mu) / var - 1 / mu) * mu ^ 2
  beta <- alpha * (1 / mu - 1)
  return(params = list(alpha = alpha, beta = beta))
}

#' Calculate Log-Normal Distribution Parameters
#'
#' Computes mu and sigma parameters for a log-normal distribution
#' given mean and standard deviation on the natural scale
#'
#' @param mean Mean of the distribution on natural scale
#' @param sd Standard deviation on natural scale
#' @return List containing mu and sigma parameters
calc_lognorm_params <- function(mean, sd) {
  v <- sd*sd
  m <- mean
  phi <- sqrt(v + m*m)
  mu <- log(m*m/phi)                # Mean of log(Y)
  sigma <- sqrt(log(phi*phi/(m*m))) # Standard deviation of log(Y)

  return(list(mu = mu, sigma = sigma))
}

#' Calculate Standard Deviation for Log-Normal Distribution
#'
#' Computes sigma parameter given mu, a percentile value p, and alpha
#'
#' @param mu Location parameter (mean of log values)
#' @param p Percentile value
#' @param alpha Probability level (default: 0.025)
#' @param max Maximum value to search
#' @return Sigma parameter
calc_sd_lognorm <- function(mu, p, alpha=0.025, max=10) {
  eq <- function(x) exp(mu + x * qnorm(alpha)) - p
  uniroot(eq, interval=c(0, max))$root
}

#' Error Function
#'
#' @param x Numeric input
#' @return Error function value
erf <- function(x) 2 * pnorm(x * sqrt(2)) - 1

#' Quantile Function
#'
#' @param x Numeric input
#' @return Quantile value
quan_func <- function(x) exp(mu + sqrt(2*sigma*sigma) / erf(2*p-1))

#' Journal Figure Size Reference
#'
#' Data frame with standard figure sizes for different journals
figure_size <- data.frame(
  journal = c("Nature", "Elsevier", "Lancet"),
  single = c(89, 90, 75),
  double = c(183, 190, 154),
  unit = c("mm", "mm", "mm")
)

#==============================================================================
# SECTION 3: DATE AND TIME UTILITIES
#==============================================================================

#' Generate Timestamp
#'
#' Creates a formatted timestamp with configurable components
#'
#' @param year Include year (default: TRUE)
#' @param month Include month (default: TRUE)
#' @param day Include day (default: TRUE)
#' @param hour Include hour (default: FALSE)
#' @param minute Include minute (default: FALSE)
#' @param second Include second (default: FALSE)
#' @return Formatted timestamp string
tstamp <- function(year=TRUE, month=TRUE, day=TRUE,
                   hour=FALSE, minute=FALSE, second=FALSE) {
  # Format date part
  date_format <- ""
  if (year) date_format <- paste0(date_format, "%Y")
  if (month) date_format <- paste0(date_format, "%m")
  if (day) date_format <- paste0(date_format, "%d")

  # Format time part
  time_format <- ""
  if (hour) time_format <- paste0(time_format, "%H")
  if (minute) time_format <- paste0(time_format, "%M")
  if (second) time_format <- paste0(time_format, "%S")

  # Create timestamp
  if (date_format == "" && time_format == "") {
    return("You'd better select parameters well.")
  }

  result <- if (date_format != "") format(Sys.time(), date_format) else ""

  if (time_format != "") {
    time_part <- format(Sys.time(), time_format)
    result <- if (result != "") paste0(result, "T", time_part) else time_part
  }

  return(result)
}

#==============================================================================
# SECTION 4: DATA PREPROCESSING FUNCTIONS
#==============================================================================

#' Clean and Standardize Country Names
#'
#' @param country Vector of country names
#' @return Vector of standardized country names
clean_country_names <- function(country) {
  # Common replacements
  replacements <- list(
    "Congo, Democratic Republic of the" = c("DR Congo", "Democratic Republic of the Congo", "DRC", "Congo, Dem. Rep.", "Congo, DR", "Congo, the Democratic Republic of the", "Congo - Kinshasa"),
    "Congo, Republic of the" = c("Congo, Rep.", "Republic of the Congo", "Congo", "Congo - Brazzaville"),
    "Sao Tome e Principe" = c("São Tomé and Príncipe"),
    "Iran, Islamic Republic of" = c("Iran", "Iran, Islamic Rep.", "Iran (Islamic Republic of)"),
    "Korea, Democratic People's Republic of" = c("North Korea", "Korea:North", "Korea, DPR", "DPRK", "Democratic People's Republic of Korea", "Korea DPR"),
    "Korea, the Republic of" = c("South Korea", "Korea:South", "Korea, Rep."),
    "South Sudan" = c("Sudan: South"),
    "Sudan" = c("Sudan: North"),
    "Venezuela, Bolivarian Republic of" = c("Venezuela", "Venezuela, RB", "Venezuela (Bolivarian Republic of)"),
    "Tanzania, United Republic of" = c("Tanzania", "United Republic of Tanzania"),
    "Syrian Arab Republic" = c("Syria"),
    "Moldova, Republic of" = c("Moldova", "Republic of Moldova"),
    "Central African Republic" = c("CAR"),
    "Lao People's Democratic Republic" = c("Lao", "Laos", "Lao PDR"),
    "United States of America" = c("US", "USA"),
    "Cote d'Ivoire" = c("C?te d'Ivoire", "CÃ´te d'Ivoire", "Cì²™te d'Ivoire", "Côte d'Ivoire", "Côte d'Ivoire"),
    "Bolivia, Plurinational State of" = c("Bolivia", "Bolivia (Plurinational State of)"),
    "Cabo Verde" = c("Cape Verde"),
    "Micronesia, Federated States of" = c("Micronesia", "Micronesia (Federated States of)"),
    "Sao Tome and Principe" = c("Sao Tome e Principe"),
    "Viet Nam" = c("Vietnam"),
    "Swaziland" = c("Eswatini")
  )

  for (std in names(replacements)) {
    country[country %in% replacements[[std]]] <- std
  }

  return(country)
}

#' Draw Parameter Samples
#'
#' Generates samples of parameter sets using Sobol's low discrepancy sequence
#'
#' @param nruns Number of runs/samples to generate (default: 200)
#' @param parameter_data Data frame containing parameter information
#' @return Data frame of sampled parameters
draw_parameter_samples <- function(nruns = 200, parameter_data = NULL) {
  # Define parameter names
  param_names <- c("run_id",
                   "dur_campaign",
                   "delay_vacc_effect",
                   "vacc_effect_direct_u5",
                   "vacc_effect_direct_5p",
                   "vacc_effect_indirect_id")

  # Generate Sobol sequence
  sobol_seq <-
    pomp::sobol_design(
      lower = c(camp_dur=0, ve_delay=0, dve_U5=0, dve_5p=0, ive_id=0,
                dw_mild=0, dw_mod=0, dw_sev=0, cfr=0, vo=0),
      upper = c(camp_dur=1, ve_delay=1, dve_U5=1, dve_5p=1, ive_id=1,
                dw_mild=1, dw_mod=1, dw_sev=1, cfr=1, vo=1),
      nseq = nruns)

  # Load parameter data if not provided
  if (is.null(parameter_data)) {
    parameter_data <- data.table::fread("data/paper/parameters.csv")
  }
  dat <- parameter_data

  # Sample vaccination campaign duration
  dur_vcamp_est <- dat[dat$Parameter == "Duration_Campaign",]$Value
  dur_vcamp_min <- dat[dat$Parameter == "Duration_Campaign",]$Min
  dur_vcamp_max <- dat[dat$Parameter == "Duration_Campaign",]$Max
  dur_vcamp_sd <- dat[dat$Parameter == "Duration_Campaign",]$SD
  dur_vcamp_max <- Inf

  dur_vcamp_sample <- truncnorm::qtruncnorm(
    sobol_seq$camp_dur,
    a=dur_vcamp_min,
    b=dur_vcamp_max,
    mean=dur_vcamp_est,
    sd=dur_vcamp_sd
  )

  # Sample vaccine effectiveness delay
  ve_delay_min <- dat[dat$Parameter == "Delay_Vacc_Eff",]$Min
  ve_delay_max <- dat[dat$Parameter == "Delay_Vacc_Eff",]$Max

  ve_delay_sample <- qunif(sobol_seq$ve_delay,
                           min=ve_delay_min, max=ve_delay_max)

  # Sample vaccine efficacy
  # Uses log-transformed incidence rate ratios which are approximately normally distributed
  ss <- cbind(sobol_seq$dve_U5, sobol_seq$dve_5p)

  str <- c("Efficacy_Vaccine_U5", "Efficacy_Vaccine")
  dve_sample <- data.frame(matrix(rep(NA, nrow(ss)*2), ncol=2))

  for (i in 1:2) {
    dve_est <- dat[dat$Parameter == str[i],]$Value
    dve_lb <- dat[dat$Parameter == str[i],]$Lower_95
    dve_ub <- dat[dat$Parameter == str[i],]$Upper_95

    # Incidence rate ratio
    irr_est <- 1 - dve_est
    irr_lb <- 1 - dve_lb
    irr_ub <- 1 - dve_ub

    # log(IRR) is approximately normal
    logirr_se <- (log(irr_lb) - log(irr_ub))/2/qnorm(0.975)
    dve_sample[, i] <- 1 - exp(qnorm(ss[,i], mean = log(irr_est), sd = logirr_se))
  }

  # Get pre-calculated indirect effectiveness data
  # ive_data <- readRDS("outputs/ive_yrep_20241010.rds")
  ive_data <- readRDS("outputs/paper/ive_yhat_yrep_20250919.rds")
  n_ive_sample <- nrow(ive_data$yhat)

  # Combine all parameters
  p_trans <- list(
    run_id = 1:nruns,
    dur_vcamp = dur_vcamp_sample,
    ve_delay = ve_delay_sample,
    dve_U5 = dve_sample[,1],
    dve_5plus = dve_sample[,2],
    ive_id = round(sobol_seq$ive_id * n_ive_sample)
  )

  params_transformed <- as.data.frame(do.call('cbind', p_trans))
  names(params_transformed) <- param_names

  return(params_transformed)
}

#==============================================================================
# SECTION 5: VACCINE IMPACT MODELING FUNCTIONS
#==============================================================================

#' Calculate Vaccine Impact for an Outbreak (Weekly)
#'
#' Computes the impact of vaccination on cases and deaths for a single outbreak
#'
#' @param data Time series data for a single outbreak
#' @param age_dist Age distribution data
#' @param vacc_week Week of vaccination
#' @param case_trigger Case threshold to trigger vaccination
#' @param vacc_cov Vaccination coverage
#' @param dve Direct vaccine effectiveness
#' @param ive_data Indirect vaccine effectiveness data
#' @param ive_rownum Row number for indirect vaccine effectiveness
#' @param week_delay Delay in weeks until vaccine takes effect
#' @return Data frame with vaccine impact results
vacc_impact_outbreak_weekly <- function(data = NULL,
                                        age_dist = NULL,
                                        # vacc_week = NULL,
                                        # case_trigger = NULL,
                                        vacc_cov = NULL,
                                        dve = NULL,
                                        ive_data = NULL,
                                        ive_rownum = NULL,
                                        week_delay = NULL) {

  # Get year as character
  yr_ch <- as.character(data$year[1])

  # Get proportion of population under 5 years old (varies by country and year)
  prop_u5 <-
    age_dist[(age_dist$`ISO3 Alpha-code` == data$country[1] &
                as.character(age_dist$Year) == yr_ch), "prop_u5"]

  # Increment used in posterior predictive values calculation
  increment <- 0.02

  # Get indirect vaccine effectiveness
  ive <- as.numeric(ive_data[ive_rownum, (round(vacc_cov/increment)+1)])

  # Extract relevant columns from input data
  df <- data[, c("country", "week", "id_outbreak", "date", "s_ch", "c_ch",
                 "deaths", "pop", "year")]

  # Add vaccination parameters
  # df$vacc_week <- vacc_week
  df$week_delay_to_vacc_effect <- week_delay
  # df$case_trigger <- case_trigger
  df$vacc_cov <- vacc_cov
  df$ive <- ive
  df$prop_u5 <- prop_u5
  df$ori_occurred <- FALSE

  # If vaccination happens after outbreak ended, no cases are averted
  if (week_delay > nrow(data)) {
    df$s_ch_averted <- 0
    df$c_ch_averted <- 0
    df$deaths_averted <- 0
  }
  else {
    df$ori_occurred <- TRUE

    # Calculate fraction of cases to be averted via vaccination
    # Based on direct and indirect effects for under 5 and 5+ age groups
    fa_u5 <- 1 - (vacc_cov*(1-dve[1])*(1-ive)+(1-vacc_cov)*(1-ive))
    fa_5p <- 1 - (vacc_cov*(1-dve[2])*(1-ive)+(1-vacc_cov)*(1-ive))

    # Apply delay to vaccine effectiveness
    frac_averted_u5 <- rep(fa_u5, nrow(data))
    frac_averted_5p <- rep(fa_5p, nrow(data))

    if (week_delay >= 1) { # If week_delay <= 0, pre-emptive vaccination scenario
      frac_averted_u5[1:week_delay] <- 0
      frac_averted_5p[1:week_delay] <- 0
    }

    # Calculate averted cases and deaths by age group and total
    df$s_ch_averted <- data$s_ch * (prop_u5 * frac_averted_u5 +
                                      (1-prop_u5) * frac_averted_5p)

    df$c_ch_averted <- data$c_ch * (prop_u5 * frac_averted_u5 +
                                      (1-prop_u5) * frac_averted_5p)

    df$deaths_averted <- data$deaths * (prop_u5 * frac_averted_u5 +
                                          (1-prop_u5) * frac_averted_5p)
  }

  # Remove unnecessary columns
  return(subset(df, select = -c(ive, prop_u5)))
}

#' Run Vaccine Impact Model for Multiple Outbreaks (Weekly)
#'
#' Computes vaccine impact across multiple outbreaks with age-specific effects
#'
#' @param outbreak_data Time series data for multiple outbreaks
#' @param ive_data Indirect vaccine effectiveness data
#' @param parameters Parameter values for the model
#' @param age_dist Age distribution data
#' @param vacc_week Week of vaccination
#' @param case_trigger Case threshold to trigger vaccination
#' @param vacc_cov Vaccination coverage
#' @param runid Run ID for parameter set
#' @return Data frame with aggregated vaccine impact results
# run_vacc_impact_outbreak_weekly <- function(outbreak_data = NULL,
#                                             ive_data = NULL,
#                                             parameters = NULL,
#                                             age_dist = NULL,
#                                             vacc_week = NULL,
#                                             case_trigger = NULL,
#                                             vacc_cov = NULL,
#                                             runid = NULL) {
#
#   # Get unique outbreak IDs
#   outbreak_ids <- unique(outbreak_data$id_outbreak)
#
#   # Create lists to store simulation results
#   lst <- vector("list", length(outbreak_ids))
#   lst2 <- vector("list", length(vacc_cov))
#   lst3 <- vector("list", length(vacc_week))
#
#   # Extract parameters for this run
#   p <- parameters[runid, c("dur_campaign", "delay_vacc_effect",
#                            "vacc_effect_direct_u5", "vacc_effect_direct_5p",
#                            "vacc_effect_indirect_id")]
#
#   # Get direct vaccine effectiveness values
#   dve <- c(p$vacc_effect_direct_u5, p$vacc_effect_direct_5p)
#
#   # Time-triggered vaccination scenario
#   if (!is.null(vacc_week) & is.null(case_trigger)) {
#     for (k in 1:length(vacc_week)) {
#       # Calculate time to vaccine effect
#       time_to_vacc <- (vacc_week[k] - 1)*7 + 3.5 # vaccinations start at mid-day
#       week_delay <- round((time_to_vacc + p$dur_campaign/2 + p$delay_vacc_effect)/7)
#
#       for (j in 1:length(vacc_cov)) {
#         for (i in 1:length(outbreak_ids)) {
#           # Calculate impact for this outbreak
#           d <- vacc_impact_outbreak_weekly(
#             data = outbreak_data[outbreak_data$id_outbreak == outbreak_ids[i],],
#             age_dist = age_dist,
#             vacc_week = vacc_week[k],
#             case_trigger = case_trigger,
#             vacc_cov = vacc_cov[j],
#             dve = dve,
#             ive_data = ive_data,
#             ive_rownum = p$vacc_effect_indirect_id,
#             week_delay = week_delay)
#
#           lst[[i]] <- d
#         }
#         lst2[[j]] <- rbindlist(lst)
#       }
#       lst3[[k]] <- rbindlist(lst2)
#     }
#   }
#   # Case-triggered vaccination scenario
#   else if (!is.null(case_trigger) & is.null(vacc_week)) {
#     for (k in 1:length(case_trigger)) {
#       for (j in 1:length(vacc_cov)) {
#         for (i in 1:length(outbreak_ids)) {
#           # Get data for this outbreak
#           data <- outbreak_data[outbreak_data$id == outbreak_ids[i],]
#
#           # Calculate when vaccination would be triggered
#           cum_case <- cumsum(data$s_ch)
#           vacc_week <- min(which(cum_case >= case_trigger[k])) # could be Inf
#
#           # Calculate time to vaccine effect
#           time_to_vacc <- (vacc_week - 1)*7 + 3.5 # vaccinations start at mid-day
#           week_delay <- round((time_to_vacc + p$dur_campaign/2 + p$delay_vacc_effect)/7)
#
#           # Calculate impact for this outbreak
#           d <- vacc_impact_outbreak_weekly(
#             data = data,
#             age_dist = age_dist,
#             vacc_week = vacc_week,
#             case_trigger = case_trigger[k],
#             vacc_cov = vacc_cov[j],
#             dve = dve,
#             ive_data = ive_data,
#             ive_rownum = p$vacc_effect_indirect_id,
#             week_delay = week_delay)
#
#           lst[[i]] <- d
#         }
#         lst2[[j]] <- rbindlist(lst)
#       }
#       lst3[[k]] <- rbindlist(lst2)
#     }
#   }
#
#   # Combine results and add run ID
#   res <- rbindlist(lst3)
#   res$runid <- runid
#
#   return(as.data.frame(res))
# }


#' Run Vaccine Impact Model for Multiple Outbreaks (Weekly)
#'
#' Computes vaccine impact across multiple outbreaks with age-specific effects
#'
#' @param outbreak_data Time series data for multiple outbreaks
#' @param ive_data Indirect vaccine effectiveness data
#' @param parameters Parameter values for the model
#' @param age_dist Age distribution data
#' @param vacc_week Week of vaccination
#' @param case_trigger Case threshold to trigger vaccination
#' @param vacc_cov Vaccination coverage
#' @param runid Run ID for parameter set
#' @return Data frame with aggregated vaccine impact results
run_vacc_impact_outbreak_weekly <- function(outbreak_data = NULL,
                                            ive_data = NULL,
                                            parameters = NULL,
                                            age_dist = NULL,
                                            vacc_week = NULL,
                                            case_trigger = NULL,
                                            vacc_cov = NULL,
                                            runid = NULL) {

  # Get unique outbreak IDs
  outbreak_ids <- unique(outbreak_data$id_outbreak)

  # Create lists to store simulation results
  lst <- vector("list", length(outbreak_ids))
  lst2 <- vector("list", length(vacc_cov))
  lst3 <- vector("list", length(vacc_week))
  lst4 <- vector("list", length(case_trigger))  # New list for hybrid strategy

  # Extract parameters for this run
  p <- parameters[runid, c("dur_campaign", "delay_vacc_effect",
                           "vacc_effect_direct_u5", "vacc_effect_direct_5p",
                           "vacc_effect_indirect_id")]

  # Get direct vaccine effectiveness values
  dve <- c(p$vacc_effect_direct_u5, p$vacc_effect_direct_5p)

  # Time-triggered vaccination scenario (only vacc_week provided)
  if (!is.null(vacc_week) & is.null(case_trigger)) {
    for (k in 1:length(vacc_week)) {
      # Calculate time to vaccine effect
      time_to_vacc <- (vacc_week[k] - 1)*7 + 3.5 # vaccinations start at mid-week and mid-day
      week_delay <- round((time_to_vacc + p$dur_campaign/2 + p$delay_vacc_effect)/7)

      for (j in 1:length(vacc_cov)) {
        for (i in 1:length(outbreak_ids)) {
          # Calculate impact for this outbreak
          d <- vacc_impact_outbreak_weekly(
            data = outbreak_data[outbreak_data$id_outbreak == outbreak_ids[i],],
            age_dist = age_dist,
            # vacc_week = vacc_week[k],
            # case_trigger = case_trigger,
            vacc_cov = vacc_cov[j],
            dve = dve,
            ive_data = ive_data,
            ive_rownum = p$vacc_effect_indirect_id,
            week_delay = week_delay)

          d$vacc_week = vacc_week[k]
          d$case_trigger = case_trigger
          d$strategy <- "onset_triggered"
          d$trigger_week <- 1

          lst[[i]] <- d
        }
        lst2[[j]] <- rbindlist(lst)
      }
      lst3[[k]] <- rbindlist(lst2)
    }
    res <- rbindlist(lst3)
  }

  # Case-triggered vaccination scenario (only case_trigger provided)
  else if (!is.null(case_trigger) & is.null(vacc_week)) {
    for (k in 1:length(case_trigger)) {
      for (j in 1:length(vacc_cov)) {
        for (i in 1:length(outbreak_ids)) {
          # Get data for this outbreak
          data <- outbreak_data[outbreak_data$id_outbreak == outbreak_ids[i],]

          # Calculate when vaccination would be triggered
          cum_case <- cumsum(data$s_ch)
          vacc_week <- min(which(cum_case >= case_trigger[k])) # could be Inf

          # Calculate time to vaccine effect
          time_to_vacc <- (vacc_week - 1)*7 + 3.5 # vaccinations start at mid-day
          week_delay <- round((time_to_vacc + p$dur_campaign/2 + p$delay_vacc_effect)/7)

          # Calculate impact for this outbreak
          d <- vacc_impact_outbreak_weekly(
            data = data,
            age_dist = age_dist,
            # vacc_week = vacc_week,
            # case_trigger = case_trigger[k],
            vacc_cov = vacc_cov[j],
            dve = dve,
            ive_data = ive_data,
            ive_rownum = p$vacc_effect_indirect_id,
            week_delay = week_delay)

           d$vacc_week = vacc_week
           d$case_trigger = case_trigger[k]
           d$strategy <- "fixed_case"
           d$trigger_week <- 1

          lst[[i]] <- d
        }
        lst2[[j]] <- rbindlist(lst2)
      }
      lst3[[k]] <- rbindlist(lst2)
    }
    res <- rbindlist(lst3)
  }
  # NEW: Hybrid strategy - case-triggered with delayed vaccination
  # (both case_trigger and vacc_week provided)
  else if (!is.null(case_trigger) & !is.null(vacc_week)) {
    for (m in 1:length(case_trigger)) {
      for (k in 1:length(vacc_week)) {
        for (j in 1:length(vacc_cov)) {
          for (i in 1:length(outbreak_ids)) {
            # Get data for this outbreak
            data <- outbreak_data[outbreak_data$id_outbreak == outbreak_ids[i],]

            # Calculate when vaccination would be triggered
            cum_case <- cumsum(data$s_ch)
            trigger_week <- min(which(cum_case >= case_trigger[m])) # could be Inf

            # Calculate actual vaccination week (trigger week + additional delay)
            actual_vacc_week <- trigger_week + vacc_week[k] - 1

            # Skip if vaccination would occur too late (after outbreak ends)
            time_to_vacc <- (actual_vacc_week - 1)*7 + 3.5 # vaccinations start at mid-day
            week_delay <- round((time_to_vacc + p$dur_campaign/2 + p$delay_vacc_effect)/7)

              # Calculate impact for this outbreak
            d <- vacc_impact_outbreak_weekly(
              data = data,
              age_dist = age_dist,
              # vacc_week = vacc_week[k],
              # case_trigger = case_trigger[m],
              vacc_cov = vacc_cov[j],
              dve = dve,
              ive_data = ive_data,
              ive_rownum = p$vacc_effect_indirect_id,
              week_delay = week_delay)

            # Add strategy identifier to the results
            d$vacc_week = vacc_week[k]
            d$case_trigger = case_trigger[m]
            d$strategy <- "case_triggered"
            d$trigger_week <- trigger_week

            lst[[i]] <- d
          }
          lst2[[j]] <- rbindlist(lst)
        }
        lst3[[k]] <- rbindlist(lst2)
      }
      lst4[[m]] <- rbindlist(lst3)
    }
    res <- rbindlist(lst4)
  }
  else {
    # No vaccination scenario
    res <- data.frame()
  }

  # Add run ID to results if we have results
  if (nrow(res) > 0) {
    res$runid <- runid
  }

  return(as.data.frame(res))
}



#' Find "Ghost" Outbreaks
#'
#' Identifies outbreaks that may not occur due to immunity from previous vaccination
#'
#' @param x Data frame with outbreak information
#' @param lb Lower bound of immunity duration in days
#' @param ub Upper bound of immunity duration in days
#' @return Vector of outbreak IDs that could be prevented
find_ghost_outbreaks <- function(x, lb=0, ub=1000) {
  mean_dur_campaign <- 6 # days duration of the vaccination campaign
  mean_delay_vacc_eff <- 11 # days, mean delay before the immunity kicks in

  # Calculate date when immunity would be acquired
  x1 <- x %>%
    mutate(date_immunity = start_date + (vacc_week - 1) * 7 + 3.5 +
             mean_dur_campaign + mean_delay_vacc_eff)

  ghost_id <- c()

  # Iteratively identify ghost outbreaks
  while (!is.null(x1) && nrow(x1) > 1) {
    # Find minimum immunity date for each region
    x2 <- x1 %>%
      group_by(admin0, admin1, admin2) %>%
      reframe(min_date_immunity = min(date_immunity))

    # Join and calculate time difference
    x1 <- left_join(
      select(x1, -any_of("min_date_immunity")),
      x2,
      by=c("admin0", "admin1", "admin2")
    ) %>%
      mutate(date_diff = date_immunity - min_date_immunity)

    # Identify new ghost outbreaks
    new_ghosts <- x1$id_outbreak[x1$date_diff > lb & x1$date_diff <= ub]
    ghost_id <- c(ghost_id, new_ghosts)

    # Remove outbreaks that would be protected
    x1 <- x1[x1$date_diff > ub, ]
  }

  return(ghost_id)
}

#==============================================================================
# SECTION 6: DATA CUMULATION AND SUMMATION FUNCTIONS
#==============================================================================

#' Create Cumulative Variables
#'
#' Adds cumulative case counts and remaining cases to outbreak data
#'
#' @param data1 Weekly outbreak data
#' @param data2 Summary outbreak data with total deaths
#' @return Data frame with added cumulative variables
create_cumul_vars <- function(data1, data2) {
  outbreak_ids <- unique(data1$id)
  results <- list()

  for (i in seq_along(outbreak_ids)) {
    d <- data1[data1$id == outbreak_ids[i],]
    total_deaths <- data2[data2$ID_outbreak == outbreak_ids[i],]$total_deaths

    # Add cumulative variables
    d$case_cum <- cumsum(d$sCh)
    d$case_total <- max(d$case_cum)
    d$death_total <- total_deaths

    # Calculate remaining cases at each time point
    d$case_rem <- c(
      d$case_total[1],  # First week, all cases remain
      d$case_total[1] - d$case_cum[1:(nrow(d)-1)]  # Subsequent weeks
    )

    results[[i]] <- d
  }

  return(as.data.frame(rbindlist(results)))
}

#' Apply Vaccination Delay
#'
#' Removes vaccine impact that would occur before the delay period
#'
#' @param data Outbreak data with vaccine impact
#' @param week_delay Delay in weeks until vaccine takes effect
#' @param case_vars Names of variables to modify
#' @return Data frame with adjusted impact values
apply_vacc_delay <- function(data = NULL, week_delay = NULL,
                             case_vars = c("case_wk_averted_U5", "case_wk_averted_5up",
                                           "case_wk_averted_tot")) {

  # Determine number of rows to modify
  nr <- min(week_delay, nrow(data))

  if (nr > 0) {
    data[1:nr, case_vars] <- 0
  }

  return(data)
}

#==============================================================================
# SECTION 7: HEALTH IMPACT CALCULATION FUNCTIONS
#==============================================================================

#' Compute Years of Life with Disability (YLD)
#'
#' Calculates the burden of illness due to disabilities
#'
#' @param cases Number of cases
#' @param parms Parameter data frame
#' @param option Calculation option (1 or 2)
#' @return Total YLD
compute_yld <- function(cases, parms=NULL, option=1) {
  if (is.null(parms)) {
    stop("Parameters to compute YLD must be provided")
  }

  # Extract needed parameters
  pmild <- parms[parms$Parameter == "Prop_Mild", "Value"]
  pm <- parms[parms$Parameter == "Prop_Moderate", "Value"]
  ps <- parms[parms$Parameter == "Prop_Severe", "Value"]
  dur <- parms[parms$Parameter == "Duration_Illness", "Value"]
  wa <- parms[parms$Parameter == "Disability_Weight_Asymptomatic", "Value"]
  wmild <- parms[parms$Parameter == "Disability_Weight_Mild", "Value"]
  wm <- parms[parms$Parameter == "Disability_Weight_Moderate", "Value"]
  ws <- parms[parms$Parameter == "Disability_Weight_Severe", "Value"]

  # Calculate weighted disability weight based on option
  dis_wt_tot <- if (option == 1) {
    # Option 1: Account for moderate and severe cases only
    pm/(pm+ps)*wm + ps/(pm+ps)*ws  # ~0.2164
  } else {
    # Option 2: Account for mild, moderate and severe cases
    pmild/(pmild+pm+ps)*wmild + pm/(pmild+pm+ps)*wm + ps/(pmild+pm+ps)*ws  # ~0.1780
  }

  # Calculate YLD
  yld_per_capita <- (dur/365) * dis_wt_tot
  yld_tot <- cases * yld_per_capita

  return(yld_tot)
}


#==============================================================================
# SECTION 8: EXCLUSION OF DUPLICATED OUTBREAK
#==============================================================================
# The approach focuses on:
#
# 1. Excluding outbreaks reported at the country level (country-level outbreak response unlikely)
# 2. Examining outbreaks at admin1 level and excluding them if there are outbreaks at admin2 or admin3 that:
#   - Fall under the same admin1 region
# - Have start and end dates within the timeframe of the admin1 outbreak
# 3. Examining outbreaks at admin2 level and excluding them if there are outbreaks at admin3 that:
#   - Fall under the same admin2 region
# - Have start and end dates within the timeframe of the admin2 outbreak
#
# The result is a deduplicated dataset that contains only the most granular (specific) outbreak information available.

# For standalone testing, we'll include the functions directly
parse_location <- function(location) {
  parts <- strsplit(location, "::")[[1]]
  result <- list(
    region = if(length(parts) >= 1) parts[1] else "",
    country = if(length(parts) >= 2) parts[2] else "",
    admin1 = if(length(parts) >= 3) parts[3] else "",
    admin2 = if(length(parts) >= 4) parts[4] else "",
    admin3 = if(length(parts) >= 5) parts[5] else "",
    admin_level = 0
  )

  if(length(parts) == 2) result$admin_level <- 0 # Country
  else if(length(parts) == 3) result$admin_level <- 1 # Admin1
  else if(length(parts) == 4) result$admin_level <- 2 # Admin2
  else if(length(parts) >= 5) result$admin_level <- 3 # Admin3

  return(result)
}

# Check if loc2 is contained within loc1
is_location_contained <- function(location1, location2) {
    # Parse both locations
  loc1 <- parse_location(location1)
  loc2 <- parse_location(location2)

  # Cannot be contained if second location has lower or equal admin level
  if(loc2$admin_level <= loc1$admin_level) {
    return(FALSE)
  }

  # Region and country must match
  if(loc1$region != loc2$region || loc1$country != loc2$country) {
    return(FALSE)
  }

  # Check all relevant admin levels
  # We need to check if each level of loc1 contains the corresponding level of loc2

  # Check admin1 level if loc1 has admin1
  if(loc1$admin_level >= 1) {
    # Handle composite admin1 (with pipe characters)
    admin1_parts <- strsplit(loc1$admin1, "\\|")[[1]]
    # If loc2's admin1 doesn't match any part of loc1's admin1, return FALSE
    if(!any(sapply(admin1_parts, function(x) {
      # Either exact match or starts with the part followed by a pipe
      return(loc2$admin1 == x || grepl(paste0("^", x, "\\|"), loc2$admin1) ||
             grepl(paste0("\\|", x, "$"), loc2$admin1) || grepl(paste0("\\|", x, "\\|"), loc2$admin1))
    }))) {
      return(FALSE)
    }
  }

  # Check admin2 level if loc1 has admin2
  if(loc1$admin_level >= 2) {
    # Handle composite admin2 (with pipe characters)
    admin2_parts <- strsplit(loc1$admin2, "\\|")[[1]]
    # If loc2's admin2 doesn't match any part of loc1's admin2, return FALSE
    if(!any(sapply(admin2_parts, function(x) {
      # Either exact match or starts with the part followed by a pipe
      return(loc2$admin2 == x || grepl(paste0("^", x, "\\|"), loc2$admin2) ||
             grepl(paste0("\\|", x, "$"), loc2$admin2) || grepl(paste0("\\|", x, "\\|"), loc2$admin2))
    }))) {
      return(FALSE)
    }
  }
  # If we've passed all checks, loc2 is contained within loc1
  return(TRUE)
}

# Check if start1-end1 encapsulated start2-end2
is_temporally_encapsulated <- function(start1, end1, start2, end2) {
  # Convert to Date objects if they're not already
  if(!inherits(start1, "Date")) start1 <- as.Date(start1)
  if(!inherits(end1, "Date")) end1 <- as.Date(end1)
  if(!inherits(start2, "Date")) start2 <- as.Date(start2)
  if(!inherits(end2, "Date")) end2 <- as.Date(end2)

  # Check if outbreak2 is completely within outbreak1's time period
  return(start1 <= start2 && end2 <= end1)
}

has_temporal_overlap <- function(start1, end1, start2, end2) {
  # Convert to Date objects if they're not already
  if(!inherits(start1, "Date")) start1 <- as.Date(start1)
  if(!inherits(end1, "Date")) end1 <- as.Date(end1)
  if(!inherits(start2, "Date")) start2 <- as.Date(start2)
  if(!inherits(end2, "Date")) end2 <- as.Date(end2)

  # Check if date ranges overlap
  return(start1 <= end2 && start2 <= end1)
}

# First function: identify potential duplicates and assign group IDs
identify_duplicate_outbreaks <- function(data) {
  # Convert dates to Date objects if not already
  data <- data %>%
    mutate(
      start_date = as.Date(start_date),
      end_date = as.Date(end_date),
      duplicate_group = NA_integer_  # Initialize group ID column
    )

  # Define the hierarchy of admin levels to check
  admin_levels <- c("country", "admin1", "admin2", "admin3")
  next_group_id <- 1  # Counter for duplicate groups

  # Function to find all duplicates recursively
  find_duplicates <- function(root_index, current_index, group_id) {
    current_row <- data[current_index, ]
    current_level <- NA

    # Determine the current admin level
    for(i in seq_along(admin_levels)) {
      if(grepl(admin_levels[i], current_row$spatial_scale)) {
        current_level <- i
        break
      }
    }

    if(is.na(current_level)) {
      print(paste("Warning: Unrecognized admin level for row", current_index, ":", current_row$spatial_scale))
      return() # Skip if not a recognized admin level
    }

    # Look for duplicates at lower admin levels
    if(current_level < length(admin_levels)) {
      for(next_level in (current_level+1):length(admin_levels)) {
        for(j in 1:nrow(data)) {
          if(current_index == j) next

          row_j <- data[j, ]

          # Skip if missing spatial scale
          if(is.null(row_j$spatial_scale) || is.na(row_j$spatial_scale)) {
            print(paste("Warning: Missing spatial_scale for row", j))
            next
          }

          # Check if this row is at the next admin level we're looking for
          if(!grepl(admin_levels[next_level], row_j$spatial_scale)) next

          # Check if this is a potential duplicate
          if(is_location_contained(current_row$location, row_j$location) &&
             is_temporally_encapsulated(current_row$start_date, current_row$end_date,
                                        row_j$start_date, row_j$end_date)) {

            print(paste("Found duplicate: Row", current_index, "(", current_row$spatial_scale,
                        ")", "contains row", j, "(", row_j$spatial_scale, ")"))

            # Mark as part of this duplicate group
            if(is.na(data$duplicate_group[current_index]))
              data$duplicate_group[current_index] <<- group_id

            data$duplicate_group[j] <<- group_id

            # Recursively find duplicates of this lower admin level
            find_duplicates(root_index, j, group_id)
          }
        }
      }
    }
  }

  # Start finding duplicate chains from all admin levels
  # Sort by admin level so we start with country, then admin1, etc.
  indices <- order(match(data$spatial_scale, admin_levels, nomatch = Inf))

  for(i in indices) {
    # Skip if this row is already part of a duplicate group
    if(!is.na(data$duplicate_group[i])) next

    # Start a new duplicate group
    current_group_id <- next_group_id

    # Find all duplicates recursively, starting with this row
    find_duplicates(i, i, current_group_id)

    # Check if we actually found any duplicates
    if(!is.na(data$duplicate_group[i])) {
      # If we did, increment the group ID for the next set
      next_group_id <- next_group_id + 1
    }
  }

  # Create a summary of duplicate groups
  duplicate_summary <- data %>%
    filter(!is.na(duplicate_group)) %>%
    group_by(duplicate_group) %>%
    summarize(
      count = n(),
      levels = paste(sort(unique(spatial_scale)), collapse = ", "),
      locations = paste(unique(location), collapse = ", "),
      time_range = paste(min(start_date), "to", max(end_date))
    )

  # Return the data with duplicate groups and summary
  return(list(
    data_with_groups = data,
    duplicate_summary = duplicate_summary
  ))
}

# Second function: select outbreaks from duplicate groups based on specified criteria
select_from_duplicates <- function(data_with_groups,
                                   selection_method = "size_duration_specificity") {
  # Clone the data to avoid modifying the original
  data <- data_with_groups %>%
    mutate(keep = TRUE)  # Initialize keep column

  # Define the hierarchy of admin levels for specificity comparison
  admin_levels <- c("country", "admin1", "admin2", "admin3")
  admin_level_rank <- setNames(seq_along(admin_levels), admin_levels)

  # Find all duplicate groups
  duplicate_groups <- unique(data$duplicate_group[!is.na(data$duplicate_group)])

  for(group_id in duplicate_groups) {
    # Get all records in this duplicate group
    group_rows <- which(data$duplicate_group == group_id)

    if(length(group_rows) <= 1) next  # Skip if only one record in group

    # Different selection methods:
    if(selection_method == "most_specific") {
      # Find the most specific admin level in this group
      admin_ranks <- sapply(data$spatial_scale[group_rows], function(scale) {
        for(level in names(admin_level_rank)) {
          if(grepl(level, scale)) return(admin_level_rank[level])
        }
        return(0)  # Unknown admin level
      })

      most_specific_level <- max(admin_ranks)

      # Keep only the most specific admin levels
      for(row in group_rows) {
        current_rank <- 0
        for(level in names(admin_level_rank)) {
          if(grepl(level, data$spatial_scale[row])) {
            current_rank <- admin_level_rank[level]
            break
          }
        }

        if(current_rank < most_specific_level) {
          data$keep[row] <- FALSE
        }
      }
    } else if(selection_method == "keep_all") {
      # Keep all records in the group
      # Nothing to do here, as all keep values are already TRUE
    } else if(selection_method == "earliest") {
      # Keep only the earliest report
      earliest_date <- min(data$start_date[group_rows])
      for(row in group_rows) {
        if(data$start_date[row] > earliest_date) {
          data$keep[row] <- FALSE
        }
      }
    } else if(selection_method == "longest_duration") {
      # Keep the report with the longest duration
      durations <- as.numeric(data$end_date[group_rows] - data$start_date[group_rows])
      max_duration <- max(durations)
      max_duration_indices <- which(durations == max_duration)

      for(i in seq_along(group_rows)) {
        if(!i %in% max_duration_indices) {
          data$keep[group_rows[i]] <- FALSE
        }
      }
    } else if(selection_method == "size_duration_specificity") {
      # First priority: largest outbreak (highest total_cases)
      largest_size <- max(data$total_s_ch[group_rows], na.rm = TRUE)
      largest_rows <- group_rows[data$total_s_ch[group_rows] == largest_size]

      if(length(largest_rows) > 1) {
        # Second priority: longest duration
        durations <- as.numeric(data$end_date[largest_rows] - data$start_date[largest_rows])
        max_duration <- max(durations, na.rm = TRUE)
        longest_rows <- largest_rows[durations == max_duration]

        if(length(longest_rows) > 1) {
          # Third priority: most specific admin level
          admin_ranks <- sapply(data$spatial_scale[longest_rows], function(scale) {
            for(level in names(admin_level_rank)) {
              if(grepl(level, scale)) return(admin_level_rank[level])
            }
            return(0)  # Unknown admin level
          })

          most_specific_level <- max(admin_ranks, na.rm = TRUE)
          most_specific_rows <- longest_rows[admin_ranks == most_specific_level]

          # Keep only the most specific rows with largest size and longest duration
          for(row in group_rows) {
            if(!row %in% most_specific_rows) {
              data$keep[row] <- FALSE
            }
          }
        } else {
          # Keep only the rows with largest size and longest duration
          for(row in group_rows) {
            if(!row %in% longest_rows) {
              data$keep[row] <- FALSE
            }
          }
        }
      }
      else {
        # Keep only the rows with largest size
        for(row in group_rows) {
          if(!row %in% largest_rows) {
            data$keep[row] <- FALSE
          }
        }
      }
    }
  }
  # Apply the filtering
  selected_data <- data %>%
    filter(keep) %>%
    select(-keep)  # Remove the temporary column

  return(selected_data)
}

# Combined function that calls both functions (for convenience)
remove_nested_outbreaks <- function(data,
                                    selection_method = "size_duration_specificity") {
  # Step 1: Identify duplicate groups
  result <- identify_duplicate_outbreaks(data)
  # Step 2: Select records based on criteria
  selected_data <- select_from_duplicates(result$data_with_groups, selection_method)

  # Return all results
  return(list(
    deduplicated = selected_data,
    all_with_groups = result$data_with_groups,
    duplicate_summary = result$duplicate_summary
  ))
}


#' Optimally Match OCV Deployments to Outbreak Populations
#'
#' Uses the Hungarian algorithm to find the globally optimal matching between OCV
#' deployments and outbreak populations, minimizing the total difference while
#' respecting error tolerance constraints.
#'
#' @param ocv_deployments Numeric vector with number of OCVs deployed during campaigns
#' @param outbreak_populations Data frame with outbreak information. Must contain 'id_outbreak' and 'population_size' columns.
#' @param error_tolerance Maximum acceptable difference between OCVs and population size (as proportion)
#' @param maximize_coverage Logical; if TRUE, optimizes for maximum vaccine coverage
#'
#' @return A list containing matches, unmatched deployments, and unmatched outbreaks
#'
#' @examples
#' ocv_deployments <- c(50000, 75000, 100000, 125000)
#' outbreak_populations <- data.frame(
#'   id_outbreak = c("OUT1", "OUT2", "OUT3", "OUT4"),
#'   population_size = c(52000, 80000, 95000, 130000),
#'   location = c("Region A", "Region B", "Region C", "Region D"),
#'   start_date = as.Date(c("2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05"))
#' )
#' result <- match_ocv_to_outbreaks(ocv_deployments, outbreak_populations)
#'
#' @import lpSolve
match_ocv_to_outbreaks <- function(ocv_deployments, outbreak_populations,
                                           error_tolerance = 0.2,
                                           maximize_coverage = FALSE) {
  # Check if lpSolve package is installed
  if (!requireNamespace("lpSolve", quietly = TRUE)) {
    stop("The lpSolve package is required for this function.
         Please install it with install.packages('lpSolve')")
  }

  # Validate input data frame
  if (!is.data.frame(outbreak_populations)) {
    stop("outbreak_populations must be a data frame")
  }

  if (!all(c("id_outbreak", "population_size") %in% names(outbreak_populations))) {
    stop("outbreak_populations must contain 'id_outbreak' and 'population_size'
         columns")
  }

  # Number of deployments and outbreaks
  n_deployments <- length(ocv_deployments)
  n_outbreaks <- nrow(outbreak_populations)

  # Create indices for deployments
  deployment_indices <- 1:n_deployments

  # Create a cost matrix representing the absolute difference between each deployment and outbreak
  cost_matrix <- matrix(0, nrow = n_deployments, ncol = n_outbreaks)
  valid_matrix <- matrix(TRUE, nrow = n_deployments, ncol = n_outbreaks)

  for (i in 1:n_deployments) {
    for (j in 1:n_outbreaks) {
      # Calculate the absolute difference as the cost
      difference <- abs(ocv_deployments[i] - outbreak_populations$population_size[j])

      # Check if the match is within error tolerance
      if (difference <= ocv_deployments[i] * error_tolerance) {
        if (maximize_coverage) {
          # When maximizing coverage, prefer matches with higher coverage ratios
          # (smaller populations relative to OCVs)
          coverage_ratio <- ocv_deployments[i] / outbreak_populations$population_size[j]
          # Convert to a cost (lower is better)
          cost_matrix[i, j] <- -coverage_ratio
        } else {
          # Standard case: minimize absolute differences
          cost_matrix[i, j] <- difference
        }
      } else {
        # Mark invalid matches (beyond error tolerance) with a very high cost
        cost_matrix[i, j] <- max(ocv_deployments) * 1000
        valid_matrix[i, j] <- FALSE
      }
    }
  }

  # If different numbers of deployments and outbreaks, pad the cost matrix
  # This creates a square matrix needed for the Hungarian algorithm
  if (n_deployments != n_outbreaks) {
    max_dim <- max(n_deployments, n_outbreaks)
    padded_cost <- matrix(max(cost_matrix) * 1000, nrow = max_dim, ncol = max_dim)
    padded_cost[1:n_deployments, 1:n_outbreaks] <- cost_matrix
    cost_matrix <- padded_cost
  }

  # Use the Hungarian algorithm to find the optimal assignment
  # lpSolve uses minimization by default, which works for our cost matrix
  result <- lpSolve::lp.assign(cost_matrix)

  # Extract assignment matrix (1 indicates a match)
  assignment <- matrix(result$solution, nrow = nrow(cost_matrix))

  # Initialize output data structures
  matches <- data.frame(
    deployment_idx = integer(),
    id_outbreak = character(),
    ocv_count = numeric(),
    population_size = numeric(),
    difference = numeric(),
    percent_difference = numeric(),
    coverage_ratio = numeric(),
    within_tolerance = logical()
  )

  # Reserve space for additional outbreak characteristics
  outbreak_columns <- setdiff(names(outbreak_populations),
                              c("id_outbreak", "population_size"))

  # Extract the matches from the assignment
  for (i in 1:n_deployments) {
    # Find which outbreak this deployment is assigned to
    j <- which(assignment[i, 1:n_outbreaks] == 1)

    # Only process real matches (not to padding)
    if (length(j) > 0 && j <= n_outbreaks) {
      # Calculate metrics
      difference <- abs(ocv_deployments[i] - outbreak_populations$population_size[j])
      percent_difference <- (difference / ocv_deployments[i]) * 100
      coverage_ratio <- ocv_deployments[i] / outbreak_populations$population_size[j]
      within_tolerance <- valid_matrix[i, j]

      # Create base match data
      match_data <- data.frame(
        deployment_idx = i,
        id_outbreak = outbreak_populations$id_outbreak[j],
        ocv_count = ocv_deployments[i],
        population_size = outbreak_populations$population_size[j],
        difference = difference,
        percent_difference = percent_difference,
        coverage_ratio = coverage_ratio,
        within_tolerance = within_tolerance
      )

      # Add all additional outbreak characteristics
      for (col in outbreak_columns) {
        match_data[[col]] <- outbreak_populations[[col]][j]
      }

      # Record the match
      matches <- rbind(matches, match_data)
    }
  }

  # Sort matches by deployment index for readability
  matches <- matches[order(matches$deployment_idx), ]

  # Identify unmatched deployments
  matched_deployments <- matches$deployment_idx
  unmatched_deployments <- setdiff(deployment_indices, matched_deployments)

  # Identify unmatched outbreaks
  matched_outbreak_ids <- matches$id_outbreak
  unmatched_outbreaks <- outbreak_populations[!outbreak_populations$id_outbreak %in% matched_outbreak_ids, ]

  # Filter out matches that are outside tolerance if needed
  invalid_matches <- matches[!matches$within_tolerance, ]
  valid_matches <- matches[matches$within_tolerance, ]

  # Summary statistics
  total_ocvs <- sum(matches$ocv_count)
  total_population <- sum(matches$population_size)
  overall_coverage <- total_ocvs / total_population

  return(list(
    matches = matches,
    valid_matches = valid_matches,
    invalid_matches = invalid_matches,
    unmatched_deployments = unmatched_deployments,
    unmatched_outbreaks = unmatched_outbreaks,
    summary = data.frame(
      total_matches = nrow(matches),
      valid_matches = nrow(valid_matches),
      invalid_matches = nrow(invalid_matches),
      total_ocvs = total_ocvs,
      total_population = total_population,
      overall_coverage = overall_coverage
    )
  ))
}
