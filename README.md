## Kim JH, Duong M, Lee E,… Impact of cholera outbreak response immunization: Insights from modeling extensive outbreaks from sub-Saharan Africa, 2010-2023.

We modeled the outcomes of outbreak response immunization (ORI)
campaigns for 1,406 cholera outbreaks in sub-Saharan Africa (2010–2023).
Our analysis accounted for vaccine effectiveness (direct and indirect),
age-specific vaccine efficacy, deployment delays, and immune response
times. In addition to all outbreaks, we evaluated subsets prioritized
based on attack rates and outbreak size to simulate scenarios with
limited vaccine supplies. Key metrics included cases, deaths,
disability-adjusted life years (DALYs) averted, and cost-effectiveness.

### R code and data for reproducing analysis

*ive_beta_reg.qmd* implements beta regression to model the indirect
vaccine effectiveness

*vacc_impct.qmd* file includes codes for the following:

-   set parameters for vaccine impact and cost-effectiveness analysis
    with related datasets and equations

-   loads utility and vaccine impact functions

-   loads the R packages and sources utils.R

-   downloads, manipulates, and pre-processes the data

*plots_tables.qmd* file includes codes for plots and tables for the main
manuscript and the supplementary material:

*julia* folder includes files for the dynamic model

-   *params.jl* setting parameters for dynamic model

-   *utils.jl* loads utility functions for dynamic model

-   *seiarw_2ag_erlang_vacc.jl* defines ODEs for dynamic model

-   *fit_model.jl* handles dynamic model fitting

-   *vacc_simulation.jl* simulates vaccine impact and compares to
    pre-emptive vaccination from static model

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
