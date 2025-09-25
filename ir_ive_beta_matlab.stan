data {
  int<lower=1> N;
  int<lower=1> K;
  vector<lower=0, upper=1>[N] y; // indirect vaccine effectiveness at some vacc cov
  matrix[N,K] X;
}

parameters {
  real<lower=0> phi;
  vector[K] beta;
}

transformed parameters{
  vector<lower=0,upper=1>[N] mu;  // transformed linear predictor for mean of beta distribution

  for (i in 1:N) {
    mu[i]  = inv_logit(X[i,] * beta);
  }
}

model {
  // priors
  beta ~ normal(0, 5);
  phi ~ cauchy(0, 5);

  // likelihood
  y ~ beta_proportion(mu, phi);

}

generated quantities{

  matrix[50,2] new_X;
  real yrep[50];
  real new_mu[50];

  for (i in 1:50) {
    new_X[i,1] = 1;
    new_X[i,2] = (1.0/50.0) * (i-1) + 1e-12;
    new_X[i,2] = logit(new_X[i,2]); # logit transform
    new_mu[i]  = inv_logit(new_X[i,] * beta);
    yrep[i] = beta_proportion_rng(new_mu[i], phi);
  }
}
