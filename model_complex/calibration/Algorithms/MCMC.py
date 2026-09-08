import numpy as np
import pymc as pm

from ...models import Model
from ...utils import ModelParams
import pytensor.tensor as pt

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

        def simulation_func(rng, alpha, beta, size=None):
            simulate_params.alpha = alpha
            simulate_params.beta = beta

            model.simulate(params=simulate_params, modeling_duration=duration)
            return get_newly_infected_base_on_time_step()#*coef_array_data
        
        with pm.Model() as pm_model:
            
            alpha = pm.Uniform(name="alpha", lower=0., upper=1., shape=(alpha_dim,))
            beta = pm.Uniform(name="beta", lower=0., upper=1., shape=(beta_dim,))
            '''

            s = pm.Gamma("minus_log_product", alpha=2.0, beta=1.0, shape=(alpha_dim,))   # s = -log(alpha*beta): identified
            d = pm.Uniform("log_ratio", lower=-s, upper=s, shape=(alpha_dim,))           # d =  log(alpha/beta): nuisance

            alpha = pm.Deterministic("alpha", pt.exp(-(s + d) / 2.0))
            beta  = pm.Deterministic("beta",  pt.exp(-(s - d) / 2.0))
            '''
            

            step = pm.DEMetropolisZ()   # tuning lambda beat tuning scaling in my tests
            
            sim = pm.Simulator(
                "sim",
                simulation_func,
                list(alpha) + [0] * (beta_dim - alpha_dim),
                beta,
                epsilon=epsilon,
                observed=data,
            )
            
            # Differential evolution (DE) Metropolis sampler
            # step=pm.DEMetropolisZ(proposal_dist=pm.LaplaceProposal)
            #step = pm.DEMetropolisZ()
            #idata = pm.sample_smc(draws=draws, chains=chains,progressbar=False)

            idata = pm.sample(
                tune=tune,
                draws=draws,
                chains=chains,
                step=step,
                progressbar=False,
            )
            idata.extend(pm.sample_posterior_predictive(idata, progressbar=False))

        posterior = idata.posterior.stack(samples=("draw", "chain"))
        print(posterior)

        import arviz as az
        def rhat1(sv):
            return float(az.rhat(az.dict_to_dataset({"s": sv}))["s"].values.mean())
        print(az.summary(idata, var_names=["alpha", "beta"])) # r_hat
        az.plot_trace(idata)

        '''
        aS = idata.posterior["alpha"].values          # (chain, draw, alpha_dim)
        bS = idata.posterior["beta"].values           # (chain, draw, beta_dim)
        A  = aS.reshape(-1, aS.shape[-1])             # (S, alpha_dim) — row i = one sample
        B  = bS.reshape(-1, bS.shape[-1])             # (S, beta_dim)
        # or the idiomatic one-liner: az.extract(idata, var_names=["alpha", "beta"])
        # thin if the simulator is expensive — 200-500 curves is plenty for bands
        step = len(A) // sample
        At, Bt = A[::step], B[::step]

        # deterministic runs: NO rng noise, NO epsilon — that's the calibration overlay
        #curves = np.array([simulator(a.ravel(), b.ravel()) for a, b in zip(At, Bt)])  # (S, T)
        '''

        '''
        
        '''
        '''
        aS = idata.posterior["alpha"].values          # (chain, draw, alpha_dim)
        bS = idata.posterior["beta"].values           # (chain, draw, beta_dim)
        A  = aS.reshape(-1, aS.shape[-1])             # (S, alpha_dim) — row i = one sample
        B  = bS.reshape(-1, bS.shape[-1])             # (S, beta_dim)
        step = len(A) // sample
        At, Bt = A[::step], B[::step]

        ci_params = []

        for a,b in zip(At, Bt):

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

        model.set_best_params(simulate_params)
        '''



        alpha = np.array(
            [
                np.random.choice(posterior["alpha"][i], size=sample)
                for i in range(alpha_dim)
            ]
        )
        beta = np.array(
            [np.random.choice(posterior["beta"][i], size=sample) for i in range(beta_dim)]
        )

        ci_params = []

        for i in range(sample):

            ci_par = ModelParams(
                alpha=alpha[:, i],
                beta=beta[:, i],
                population_size=model_params.population_size,
                initial_infectious=model_params.initial_infectious,
            )

            ci_params.append(ci_par)

        model.set_ci_params(ci_params)

        simulate_params.alpha = [a.mean() for a in alpha]
        simulate_params.beta = [b.mean() for b in beta]
        model.set_best_params(simulate_params)