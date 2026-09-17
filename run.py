from model_complex import InfluenzaData
from prediction_module.predictor import PredictionGenerator



if __name__ == '__main__':
    city = 'russia'
    save_path_folder='report_rscf_40/'
    # 35 36
    epid_data = InfluenzaData(city, begin_year=2025, begin_week=35, end_year=2026, end_week=36)

    predictor = PredictionGenerator(epid_data=epid_data,
                                    city=city,
                                    save_path=save_path_folder)


    method = 'mcmc'
    type = 'total'
    eps = 2


    forecast_duration = 4
    predictor.generate_forecasts(method=method,
                                type=type,
                                forecast_duration=forecast_duration,
                                epsilon_start=eps,
                                epsilon_end=eps,
                                epsilon_step=1,
                                n_trials=5,
                                sample=600, # choosing N lines for the final plot?
                                tune=400,#1000
                                draws=200,#1000
                                chains=4)# 6 from Dinara's ipynb