## Kim JH, Duong M, Lee E,… Impact of cholera outbreak response immunization: Insights from modeling extensive outbreaks from sub-Saharan Africa, 2010-2023.

This study models the outcomes of outbreak response immunization (ORI)
campaigns across 1,406 cholera outbreaks in sub-Saharan Africa from 2010
to 2023. We incorporated vaccine effectiveness (both direct and
indirect), age-specific efficacy, deployment delays, and immune response
timelines into our analysis. Beyond evaluating all outbreaks, we
analyzed prioritized subsets based on attack rates and outbreak scale to
mimic scenarios with constrained vaccine availability. Key outcomes
assessed include cases prevented, deaths averted, disability-adjusted
life years (DALYs) gained, and cost-effectiveness.

### R code and data for reproducing analysis

**utils.R**: Houses various in-house utility functions including those
to calculate the impact of vaccines.

**ive_beta_reg.qmd**: Uses beta regression to estimate indirect vaccine
effectiveness.

**parameters.qmd**: Generates 200 parameter samples via Sobol’s
low-discrepancy sequence.

**vacc_impct.qmd**: Contains code to: - Load utility functions and
vaccine impact calculations (i.e., source `utils.R`). - Calculate
vaccine impact via time-triggered and case-triggered strategies.
Simulation can be done in parallel using ‘doParallel’ package and then
merged later

-   Summarize vaccine impact

**plots_tables.qmd**: Provides code for creating plots and tables
featured in the main manuscript and supplementary materials.

**julia folder**: Houses files for the dynamic model:

-   *params.jl*: Sets parameters for the dynamic model.

-   *utils.jl*: Loads utility functions supporting the dynamic model.

-   *seiarw_2ag_erlang_vacc.jl*: Specifies ODEs for the dynamic model.

-   *fit_model.jl*: Manages fitting of the dynamic model.

-   *vacc_simulation.jl*: Simulates vaccine impact and compares it to
    pre-emptive vaccination from a static model.

### Datasets used

-   Global Taskforce for Cholera Control’s (GTFCC) Global Cholera
    Database<sup>1</sup>

-   [United Nations World Population Prospects 2024 -
    Mortality](https://population.un.org/wpp/Download/Standard/Mortality/)

-   [United Nations World Population Prospects 2024 -
    Population](https://population.un.org/wpp/Download/Standard/Population/)

-   [World Bank Data, GDP per capita (current
    US$)](https://data.worldbank.org/indicator/NY.GDP.PCAP.CD)

-   [World Bank Data, Labor force participation rate, total (% of total
    population ages 15-64) (modeled ILO
    estimate)](https://data.worldbank.org/indicator/SL.TLF.ACTI.ZS)

-   [International Monetary Fund. Inflation rate, average consumer
    prices](https://www.imf.org/external/datamapper/PCPIPCH@WEO/OEMDC/ADVEC/WEOWORLD)

<sup>1</sup>Zheng Q, Luquero FJ, Ciglenecki I, Wamala JF, Abubakar A,
Welo P, et al. Cholera outbreaks in sub-Saharan Africa during 2010-2019:
a descriptive analysis. Int J Infect Dis 2022 Sep;122:215–21.
