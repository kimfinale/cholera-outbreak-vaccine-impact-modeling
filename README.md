---
title: "README"
author: "Monica Duong"
date: "2024-06-05"
output:
  pdf_document: default
  html_document: default
---

## Kim JH, Duong M, Lee E,... Impact of cholera outbreak response immunization: Insights from modeling extensive outbreaks from sub-Saharan Africa, 2010-2023.

We modeled the outcomes of outbreak response immunization (ORI) campaigns for 1,406 cholera outbreaks in sub-Saharan Africa (2010--2023). Our analysis accounted for vaccine effectiveness (direct and indirect), age-specific vaccine efficacy, deployment delays, and immune response times. In addition to all outbreaks, we evaluated subsets prioritized based on attack rates and outbreak size to simulate scenarios with limited vaccine supplies. Key metrics included cases, deaths, disability-adjusted life years (DALYs) averted, and cost-effectiveness.

### R code and data for reproducing analysis

The R scripts in this repository can be used to replicate the analysis and results illustrated in the article. 

### Static model 

-   *00.params.R* setting parameters for vaccine impact and cost-effectiveness analysis with related datasets and equations

-   *01.utils.R* loads utility and vaccine impact functions

-   *02.package.R* loads the R packages and sources utils.R

-   *03.prepdata.R* downloads, manipulates, and pre-processes the data

-   *04.ive_beta_reg.qmd* implements beta regression to model the indirect vaccine effectiveness

-   *05.simulations.R* performs model simulations

-   *06.vaccineimpact.R* computes impact for different vaccination scenarios


### Dynamic model

-   *07.params.jl* setting parameters for dynamic model

-   *08.utils.jl* loads utility functions for dynamic model

-   *09.seiarw_2ag_erlang_vacc.jl* defines ODEs for dynamic model

-   *10.fit_model.jl* handles dynamic model fitting 

-   *11.vacc_simulation.jl* simulates vaccine impact and compares to pre-emptive vaccination from static model

### Manuscript outputs

-   *06.figures.R* produces the figures of the article

-   *07.tables.R* produces the tables included in the article

-   *08.supp.R* produces other outputs of the analysis


### Datasets used

-   Global Taskforce for Cholera Control's (GTFCC) Global Cholera Database^1^
-   [United Nations World Population Prospects 2024 - Mortality](https://population.un.org/wpp/Download/Standard/Mortality/)
-   [United Nations World Population Prospects 2024 - Population](https://population.un.org/wpp/Download/Standard/Population/)
-   [World Bank Data, GDP per capita (current US\$)](https://data.worldbank.org/indicator/NY.GDP.PCAP.CD)
-   [World Bank Data, Labor force participation rate, total (% of total population ages 15-64) (modeled ILO estimate)](https://data.worldbank.org/indicator/SL.TLF.ACTI.ZS)
-   [International Monetary Fund. Inflation rate, average consumer prices](https://www.imf.org/external/datamapper/PCPIPCH@WEO/OEMDC/ADVEC/WEOWORLD)

^1^Zheng Q, Luquero FJ, Ciglenecki I, Wamala JF, Abubakar A, Welo P, et al. Cholera outbreaks in sub-Saharan Africa during 2010-2019: a descriptive analysis. Int J Infect Dis IJID Off Publ Int Soc Infect Dis. 2022 Sep;122:215--21.
