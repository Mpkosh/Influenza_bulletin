import os
from datetime import datetime
from model_complex import InfluenzaData
from plot_module.calibration.calibration_and_forecast import calibration_forecast_plot

class PredictionGenerator:
    def __init__(self, epid_data: InfluenzaData, city: str, save_path: str):
        '''TODO взять параметры базовые
        1. путь для сохранения
        2. название алгоритма
        3. возможно, гиперпараметры. Хотя может и перебор их здесь реализовать?'''
        self.epid_data = epid_data
        self.city = city
        # self.method =  method
        # self.type =  type
        self.save_path = save_path
        # self.forecast_duration = forecast_duration
        # self.epsilon_start = epsilon_start
        # self.epsilon_end =  epsilon_end

    
    def generate_forecasts(self, method: str, type: str, forecast_duration: int, epsilon_start: int = 10, 
                           epsilon_end: int = 100, epsilon_step: int = 10, n_trials: int = 2, sample: int = 100, tune: int = 2500, draws: int = 600, chains: int = 4):
        """Генерирует прогноз на несколько шагов вперед
        n_trials -- кол-во запусков с одинаковыми параметрами"""

        save_path_folder=_get_or_create_today_folder(self.save_path, method = method)
        timestamp = datetime.now().strftime("%Y%m%d")
        epsilon = epsilon_start
        #for epsilon in range(epsilon_start, epsilon_end + 1, epsilon_step):
        epsilon=epsilon_start
        for i in range(n_trials):
            title = f"method_{method}_eps_{epsilon}_iter_{i}_time_{timestamp}"
            save_path = save_path_folder + title
            calibration_forecast_plot(epid_data=self.epid_data, 
                                city=self.city, 
                                method=method,
                                type=type,
                                forecast_duration=forecast_duration,
                                title=title,
                                save_path=save_path,
                                epsilon=epsilon,
                                is_prevalence_plot=True,
                                is_recovered_plot=True,
                                sample=sample,
                                tune=tune,
                                draws=draws,
                                chains=chains)
                
def _get_or_create_today_folder(base_path: str, method: str) -> str:
    """
    Проверяет существование папки с сегодняшней датой (year_month_day).
    Если папка существует - возвращает путь к ней.
    Если не существует - создает и возвращает путь.
    
    Args:
        base_path: Базовый путь, где должна быть создана папка
        
    Returns:
        str: Полный путь к папке с сегодняшней датой
    """
    # Получаем сегодняшнюю дату в формате year_month_day
    today_date = datetime.now().strftime("%Y_%m_%d")
    
    # Формируем полный путь к папке
    folder_path = os.path.join(base_path, today_date)
    method_folder = os.path.join(folder_path, method)
    
    # Проверяем существование папки
    if not os.path.exists(method_folder):
        # Создаем папку (включая все родительские директории при необходимости)
        os.makedirs(method_folder, exist_ok=True)
        print(f"Создана папка: {method_folder}")
    else:
        print(f"Папка уже существует: {method_folder}")
    
    return method_folder+"/"