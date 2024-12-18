data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  vector<lower=0,upper=1>[N] y;
  matrix[N,K] X;
  matrix[N,J] Z;
}

parameters {
  vector[K] beta;
  vector[J] gamma;
}

transformed parameters{
  vector<lower=0,upper=1>[N] mu;    // transformed linear predictor for mean of beta distribution
  vector<lower=0>[N] phi;           // transformed linear predictor for precision of beta distribution
  vector<lower=0>[N] A;             // parameter for beta distn
  vector<lower=0>[N] B;             // parameter for beta distn

  for (i in 1:N) {
    mu[i]  = inv_logit(X[i,] * beta);
    phi[i] = exp(Z[i,] * gamma);
  }

  A = mu .* phi;
  B = (1.0 - mu) .* phi;
}

model {
  // priors
  beta ~ normal(2, 1);
  gamma ~ normal(0, 1);
  // likelihood
  y ~ beta(A, B);
}

generated quantities{

  matrix[50,2] new_X;
  matrix[50,2] new_Z;

  for (i in 1:50) {
    new_X[i,1] = 1;
    new_X[i,2] = (1.0/50.0)*(i-1) + 1e-6;
    new_X[i,2] = logit(new_X[i,2]); # logit transform

    new_Z[i,1] = 1;
    new_Z[i,2] = (1.0/50.0)*(i-1) + 1e-6;
    // new_Z[i,2] = log(new_Z[i,2]);
  }
  real yrep[50];
  real new_mu[50];
  real new_phi[50];
  real new_A[50];
  real new_B[50];

  for (i in 1:50) {
    new_mu[i]  = inv_logit(new_X[i,] * beta);
    new_phi[i] = exp(new_Z[i,] * gamma);
    new_A[i] = new_mu[i] * new_phi[i];
    new_B[i] = (1.0 - new_mu[i]) * new_phi[i];

    yrep[i] = beta_rng(new_A[i], new_B[i]);
  }
}
