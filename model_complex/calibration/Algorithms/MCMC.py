from xxlimited import new

import numpy as np
import pymc as pm

from ...models import Model
from ...utils import ModelParams
import pytensor.tensor as pt


def strict_epidemic_distance(epsilon, obs_data, sim_data):
    """
    Correct PyMC Simulator signature: (epsilon, obs_data, sim_data)
    Must return a negative pseudo-log-likelihood!
    """
    sim_data_clipped = np.clip(sim_data, a_min=0, a_max=None)
    max_obs = np.max(obs_data)
    
    obs_scaled = obs_data / max_obs
    sim_scaled = sim_data_clipped / max_obs
    
    # 1. RMSE on peak region (> 35% of max)
    important_indices = obs_scaled > 0.35
    rmse = np.sqrt(np.mean((obs_scaled[important_indices] - sim_scaled[important_indices]) ** 2))
    
    # 2. Peak height penalty
    peak_penalty = np.abs(np.max(obs_scaled) - np.max(sim_scaled))
    
    # 3. Smooth peak timing penalty (Center of Mass of peak region)
    t = np.arange(len(obs_data))
    w_obs = np.maximum(0, obs_scaled - 0.35)
    w_sim = np.maximum(0, sim_scaled - 0.35)
    
    t_center_obs = np.sum(t * w_obs) / (np.sum(w_obs) + 1e-7)
    t_center_sim = np.sum(t * w_sim) / (np.sum(w_sim) + 1e-7)
    
    # Smooth timing penalty (e.g., 0.05 per day of discrepancy)
    timing_penalty = np.abs(t_center_obs - t_center_sim) * 0.05
    
    total_distance = rmse + peak_penalty + timing_penalty
    
    # negative pseudo-log-likelihood for PyMC to maximize!
    return -0.5 * ((total_distance / epsilon) ** 2)


class MCMC:
    
    
    
    @classmethod
    def calibrate(
        self,
        coef_array_data:np.array,
        model: Model,
        data: np.array,
        time_step: str,
        model_params: ModelParams,
        sample=100,
        epsilon=10000,
        tune=2500,
        draws=500,
        chains=4,
    ):
        """
        Parameters:
            - with_rho -- tune population size
            - with_initi -- tune initial infected
            - tune -- number of mcmc warmup samples
            - draws -- number of mcmc draws
            - chains -- number of chains
        """
        new = True

        def simulation_func(rng, alpha, beta, size=None):
                simulate_params.alpha = alpha
                simulate_params.beta = beta
        
                model.simulate(params=simulate_params, modeling_duration=duration)
                res = get_newly_infected_base_on_time_step()
                return res
        alpha_dim, beta_dim = model.params()
        duration = len(data) // alpha_dim
        get_newly_infected_base_on_time_step = model.get_daily_newly_infected

        if time_step == "week":
            duration *= 7
            get_newly_infected_base_on_time_step = model.get_weekly_newly_infected

        simulate_params = ModelParams(
            alpha=[0],
            beta=[0],
            population_size=model_params.population_size,
            initial_infectious=model_params.initial_infectious,
        )
        if new:
            with pm.Model() as pm_model:
                log_ab_combined = pm.Normal("log_ab_combined", mu=-0.888, sigma=0.04, shape=(alpha_dim,))
                log_ratio = pm.Normal("log_ratio", mu=-0.847, sigma=0.04) 
                
                log_alpha = (log_ab_combined + log_ratio) / 2.0
                log_beta = (log_ab_combined - log_ratio) / 2.0
                
                alpha = pm.Deterministic("alpha", pm.math.exp(log_alpha))
                beta = pm.Deterministic("beta", pm.math.exp(log_beta))
                
                out_of_bounds = (alpha > 1.0) | (beta > 1.0)
                pm.Potential("bounds_check", pm.math.switch(out_of_bounds, -np.inf, 0.0))
                
                sim = pm.Simulator(
                    "sim",
                    simulation_func,
                    params=[alpha, beta],
                    epsilon=epsilon,
                    distance=strict_epidemic_distance,
                    observed=data,
                )
                
                idata = pm.sample_smc(draws=draws, chains=chains, progressbar=True)

        else:
        
            with pm.Model() as pm_model:
                alpha = pm.Uniform(name="alpha", lower=0, upper=1, shape=(alpha_dim,))
                beta = pm.Uniform(name="beta", lower=0, upper=1, shape=(beta_dim,))

                sim = pm.Simulator(
                    "sim",
                    simulation_func,
                    list(alpha) + [0] * (beta_dim - alpha_dim),
                    beta,
                    epsilon=epsilon,
                    observed=data,
                )

                # Differential evolution (DE) Metropolis sampler
                step = pm.DEMetropolisZ()

                idata = pm.sample(
                    tune=tune,
                    draws=draws,
                    chains=chains,
                    step=step,
                    progressbar=False,
                )
                idata.extend(pm.sample_posterior_predictive(idata, progressbar=False))

            posterior = idata.posterior.stack(samples=("draw", "chain"))

        import arviz as az

        print(az.summary(idata)) # r_hat
        az.plot_trace(idata)

        if new:
            alphas_sampled = idata.posterior["alpha"].values.reshape(-1)
            betas_sampled = idata.posterior["beta"].values.reshape(-1)
            step_size = len(alphas_sampled) // 300  # Нарисует "...// N" кривых
            ci_params = []
            for i in range(0, len(alphas_sampled), step_size):
                a_val = alphas_sampled[i]
                b_val = betas_sampled[i]
                ci_par = ModelParams(
                    alpha=[a_val], #alpha[:, i],
                    beta=[b_val], #beta[:, i],
                    population_size=model_params.population_size,
                    initial_infectious=model_params.initial_infectious,
                )

                ci_params.append(ci_par)
            model.set_ci_params(ci_params)
            
            simulate_params.alpha = [np.median(alphas_sampled, axis=0)]# [a.mean() for a in alpha]
            simulate_params.beta = [np.median(betas_sampled, axis=0)]#[b.mean() for b in beta]
            print(simulate_params)

        else:
            aS = idata.posterior["alpha"].values          # (chain, draw, alpha_dim)
            bS = idata.posterior["beta"].values           # (chain, draw, beta_dim)
            A  = aS.reshape(-1, aS.shape[-1])             # (S, alpha_dim) — row i = one sample
            B  = bS.reshape(-1, bS.shape[-1])             # (S, beta_dim)
            
            step = len(A) // sample
            print(len(A), step, sample)
            At, Bt = A[::step], B[::step]

            ci_params = []

            for a,b in zip(At, Bt):
                #print(a,b)
                ci_par = ModelParams(
                    alpha=a.ravel(), #alpha[:, i],
                    beta=b.ravel(), #beta[:, i],
                    population_size=model_params.population_size,
                    initial_infectious=model_params.initial_infectious,
                )

                ci_params.append(ci_par)

            model.set_ci_params(ci_params)

            simulate_params.alpha = [np.median(A, axis=0)]# [a.mean() for a in alpha]
            simulate_params.beta = [np.median(B, axis=0)]#[b.mean() for b in beta]
            print(simulate_params)


        model.set_best_params(simulate_params)