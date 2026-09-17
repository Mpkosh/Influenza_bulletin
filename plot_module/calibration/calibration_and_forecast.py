import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import io
from typing import Literal, Optional, Union, Dict, Any
from model_complex import Calibration, FactoryModel, ModelParams
from PIL import Image


# ========== УТИЛИТЫ ДЛЯ РАБОТЫ С ДАННЫМИ ==========

def week_to_label(week_num, year):
    """Конвертирует номер недели в метку с учетом нового года"""
    while week_num > 52:
        week_num -= 52
        year += 1
    return f"{week_num}"


def prepare_calibration_data(epid_data, type):
    """Подготавливает данные для калибровки"""
    epid_data.get_wave_data(type=type)
    data = epid_data.get_data()
    dur = epid_data.get_duration()
    plot_data = epid_data.prepare_for_plot()
    return data, dur, plot_data


def get_model_and_function(epid_data, type):
    """Создает модель и возвращает функцию для получения данных"""
    model_params = ModelParams(
        alpha=[0],
        beta=[0],
        population_size=epid_data.get_rho() // 10,
        initial_infectious=[100],
    )
    
    model = FactoryModel.get_model(type)
    
    if type == "age":
        model_params.initial_infectious = [100, 100]
    
    if epid_data.get_data().attrs["time_step"] == "week":
        func_to_get_newly_data = model.get_weekly_newly_infected_by_group
    else:
        func_to_get_newly_data = model.get_daily_newly_infected_by_group
    
    return model, model_params, func_to_get_newly_data


def run_calibration(model, data, model_params, coef_array_data, method, epsilon,sample,tune,draws,chains):
    """Запускает калибровку модели"""
    calibration = Calibration(model, data, model_params)
    if method.lower() == "annealing":
        calibration.annealing_calibration()
    elif method.lower() == "abc":
        calibration.abc_calibration(sample=200, epsilon=epsilon)
    elif method.lower() == "mcmc":
        calibration.mcmc_calibration(coef_array_data=coef_array_data, sample=sample, epsilon=epsilon,tune=tune,draws=draws,chains=chains)
    else:
        calibration.optuna_calibration()
    
    return calibration


def calculate_coefficients(epid_data, forecast_duration):
    """Рассчитывает коэффициенты нормализации"""
    coef_array_data = np.array(10000 / epid_data.returned_df["total_population"])
    coef_array_forecast = np.full(forecast_duration + 1, 10000 / epid_data.returned_df["total_population"].iloc[-1])
    coef_array_all = np.concatenate([coef_array_data, coef_array_forecast[1:]])
    return coef_array_data, coef_array_forecast, coef_array_all


def prepare_x_axis(epid_data, dur):
    """Подготавливает ось X и метки"""
    array = list(range(epid_data.begin_week, epid_data.begin_week + round(dur / 7)))
    x_labels = [week_to_label(week, epid_data.begin_year) for week in array]
    return array, x_labels


# ========== ФУНКЦИИ ДЛЯ ПОСТРОЕНИЯ ГРАФИКА ==========

def setup_figure():
    """Настраивает фигуру matplotlib"""
    desired_width_px, desired_height_px =1000, 330
    dpi = 600
    width_inches = desired_width_px / dpi
    height_inches = desired_height_px / dpi
    plt.figure(figsize=(width_inches * 8, height_inches * 8))
    return dpi


def plot_confidence_intervals(model, func_to_get_newly_data, array, coef_array_all, ci_params, dur):
    """Рисует доверительные интервалы"""
    print(len(ci_params),'len ciparams')
    n=5
    for ci_par in ci_params:
        model.simulate(params=ci_par, modeling_duration=dur)
        res = func_to_get_newly_data()
        for i in range(len(res)):
            plt.plot(array[n:], res[i][n:] * coef_array_all[n:], lw=0.3, alpha=0.05, color='lightblue')


def plot_best_model_and_data(model, func_to_get_newly_data, array, plot_data, 
                              coef_array_data, coef_array_forecast, dur):
    """Рисует лучшую модель и данные"""
    print(model.get_best_params())

    model.simulate(params=model.get_best_params(), modeling_duration=dur)
    res = func_to_get_newly_data()
    n = 5
    for i in range(len(res)):
        # r2 = round(r2_score(plot_data[:, i], res[i][:len(plot_data[:, i])]), 2)
        print(len(plot_data[:, i]), len(res[i][:len(plot_data[:, i])]))
        r2 = round(r2_score(plot_data[n:, i], res[i][n:n+len(plot_data[n:, i])]), 2)
        plt.plot(
            array[n: n+len(plot_data[n:, i])],
            res[i][n: n+len(plot_data[n:, i])] * coef_array_data[n: n+len(plot_data[n:, i])],
            label=f"Лучшая модель, $R^2$: {r2}",
            lw=1.0,
            color='royalblue',
        )
        
        last_known_idx = len(plot_data[n:, i]) - 1
        print(last_known_idx, array[last_known_idx:])
        plt.plot(
            array[n+last_known_idx:],
            res[i][n+last_known_idx:] * coef_array_forecast,
            '--', color='lightcoral', alpha=1.0,
            label='Прогноз'
        )
        
        plt.scatter(
            array[n:n+len(plot_data[n:, i])],
            plot_data[n:, i] * coef_array_data[n:],
            marker="o", color='blue', zorder=5, label="Данные"
        )
    
       

def configure_plot_style(array, x_labels, max_real_data):
    """Настраивает стиль графика"""
    n=5
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    y_min, y_max = plt.ylim()#-1, max(1,max_real_data))
    plt.ylim(-1, y_max)
    #plt.ylim(0, max_real_data)
    if y_max < 5:
        step = 1
    else:
        step = 5
    y_ticks = np.arange(0, np.ceil(y_max) + 0.5, step)
    plt.yticks(y_ticks)
    plt.xticks(array[n:], x_labels[n:], rotation=0, fontsize=10)

    '''
    x_min, x_max = plt.xlim()
    print(x_min, x_max)
    if x_min < 40:
        plt.xlim(40, x_max)
    '''
    plt.tight_layout(rect=[0.03, 0.05, 0.97, 1])


def add_russian_labels():
    """Добавляет русские подписи"""
    plt.legend(fontsize=10)
    plt.xlabel('Недели', fontsize=12)
    plt.ylabel('Заболеваемость гриппом на 10 тыс. населения', fontsize=10)


def add_english_labels():
    """Добавляет английские подписи"""
    plt.xlabel('Weeks', fontsize=12)
    plt.ylabel('Influenza morbidity per 10000 population', fontsize=10)


def translate_legend():
    """Переводит легенду на английский"""
    handles, labels = plt.gca().get_legend_handles_labels()
    label_mapping = {
        "Лучшая модель": "Best model",
        "Прогноз": "Forecast",
        "Данные": "Data"
    }
    
    new_labels = []
    for label in labels:
        if "Лучшая модель" in label:
            new_label = label.replace("Лучшая модель", "Best model")
            new_labels.append(new_label)
        elif label in label_mapping:
            new_labels.append(label_mapping[label])
        else:
            new_labels.append(label)
    
    plt.legend(handles, new_labels, fontsize=10)


# ========== ФУНКЦИИ ДЛЯ РАБОТЫ С ИЗОБРАЖЕНИЯМИ ==========

def resize_image_bytes(img_bytes: bytes, future_size: tuple = (900, 330)) -> bytes:
    """Изменяет размер изображения из байтов и возвращает байты"""
    img = Image.open(io.BytesIO(img_bytes))
    img_resized = img.resize(future_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img_resized.save(buf, format="png")
    return buf.getvalue()


def resize_image_file(file_path: str, future_size: tuple = (900, 330), prefix_name: str = "small", format: str = "png") -> str:
    """Изменяет размер изображения из файла и сохраняет с префиксом"""
    img = Image.open(file_path)
    img_resized = img.resize(future_size, Image.Resampling.LANCZOS)
    new_file_path = file_path.replace(format, "_" + prefix_name + "." + format)
    img_resized.save(new_file_path)
    return new_file_path


def save_plot_to_bytes(dpi: int) -> Dict[str, bytes]:
    """Сохраняет текущий график в BytesIO и возвращает словарь с байтами"""
    # Сохраняем в PNG в память
    buf_png = io.BytesIO()
    plt.savefig(buf_png, format='png', dpi=dpi, bbox_inches="tight")
    buf_png.seek(0)
    
    # Сохраняем в PDF в память
    buf_pdf = io.BytesIO()
    plt.savefig(buf_pdf, format='pdf', bbox_inches="tight")
    buf_pdf.seek(0)
    
    return {
        "png": buf_png.getvalue(),
        "pdf": buf_pdf.getvalue()
    }


def save_plot_to_files(save_path: str, suffix: str, dpi: int) -> Dict[str, str]:
    """Сохраняет график в файлы и возвращает пути"""
    img_path = save_path + suffix
    plt.savefig(img_path + ".pdf", bbox_inches="tight")
    plt.savefig(img_path + ".png", dpi=dpi, bbox_inches="tight")
    small_path = resize_image_file(img_path + ".png", prefix_name="small")
    
    return {
        "png": img_path + ".png",
        "pdf": img_path + ".pdf",
        "small": small_path
    }


# ========== ОСНОВНАЯ ФУНКЦИЯ ==========

def calibration_forecast_plot(
    epid_data,
    city: str,
    method: str,
    type: str,
    forecast_duration: int,
    title: str,
    save_path: Optional[str] = None,
    epsilon: int = 3000,
    output_mode: Literal["local", "bytes", "both"] = "local",
    is_prevalence_plot=True,
    is_recovered_plot=True,
    sample:int=100,
    tune:int=2500,
    draws:int=600,
    chains:int=4,
) -> Dict[str, Any]:
    """
    Создает калибровочный график прогноза
    
    Args:
        epid_data: данные эпидемии
        city: название города
        method: метод калибровки
        type: тип прогноза ("total" или "age")
        forecast_duration: длительность прогноза в неделях
        title: заголовок графика
        save_path: путь для сохранения (нужен при output_mode="local" или "both")
        epsilon: параметр epsilon для калибровки
        output_mode: режим вывода
            - "local": сохранить локально (требует save_path)
            - "bytes": вернуть байты (без сохранения на диск)
            - "both": и сохранить локально, и вернуть байты
    
    Returns:
        Dict с результатами в зависимости от режима:
            - local: {"ru": {...}, "en": {...}} с путями к файлам
            - bytes: {"ru": {...}, "en": {...}} с байтами
            - both: {"ru": {...}, "en": {...}, "local_paths": {...}, "bytes": {...}}
    """
    
    # Проверка save_path для локального сохранения
    if output_mode in ["local", "both"] and not save_path:
        raise ValueError(f"save_path is required when output_mode='{output_mode}'")
    
    # 1. Подготовка данных
    data, dur, plot_data = prepare_calibration_data(epid_data, type)
    model, model_params, func_to_get_newly_data = get_model_and_function(epid_data, type)

    coef_array_data = np.array(10000 / epid_data.returned_df["total_population"])
    # 2. Калибровка
    calibration = run_calibration(model, data, model_params, coef_array_data, method, epsilon, sample, tune, draws, chains)
    
    # 3. Подготовка к построению
    dur += forecast_duration * 7
    array, x_labels = prepare_x_axis(epid_data, dur)
    ci_params = list(model.get_ci_params())

    coef_array_data, coef_array_forecast, coef_array_all = calculate_coefficients(epid_data, forecast_duration)
    dpi = setup_figure()
    
    # 4. Построение графика
    plot_confidence_intervals(model, func_to_get_newly_data, array, coef_array_all, ci_params, dur)
    plot_best_model_and_data(model, func_to_get_newly_data, array, plot_data, 
                              coef_array_data, coef_array_forecast, dur)
    real_data = plot_data[:, 0] * coef_array_data
    configure_plot_style(array, x_labels, real_data.max()*1.2)
    
    # Базовое имя файла
    base_filename = f"{save_path}_{city}_{method}_{type}_{epid_data.begin_year}-{epid_data.end_year}" if save_path else ""
    
    results = {}
    
    # 5. Обработка русской версии
    add_russian_labels()
    
    if output_mode in ["local", "both"]:
        ru_paths = save_plot_to_files(base_filename, "", dpi)
        results["ru"] = ru_paths
    
    if output_mode in ["bytes", "both"]:
        ru_bytes = save_plot_to_bytes(dpi)
        # Создаем маленькую версию
        ru_bytes["small"] = resize_image_bytes(ru_bytes["png"])
        results["ru_bytes"] = ru_bytes
    
    # 6. Обработка английской версии
    # Очищаем текущую фигуру для английской версии
    plt.clf()
    setup_figure()
    plot_confidence_intervals(model, func_to_get_newly_data, array, coef_array_all, ci_params, dur)
    plot_best_model_and_data(model, func_to_get_newly_data, array, plot_data, 
                              coef_array_data, coef_array_forecast, dur)
    configure_plot_style(array, x_labels, real_data.max()*1.2)
    
    add_english_labels()
    translate_legend()
    
    if output_mode in ["local", "both"]:
        en_paths = save_plot_to_files(base_filename, "_en", dpi)
        results["en"] = en_paths
    
    if output_mode in ["bytes", "both"]:
        en_bytes = save_plot_to_bytes(dpi)
        en_bytes["small"] = resize_image_bytes(en_bytes["png"])
        results["en_bytes"] = en_bytes
    
    # 7. Очистка
    plt.clf()
    
    # 8. Возврат результата
    if output_mode == "local":
        return {"ru": results["ru"], "en": results["en"]}
    elif output_mode == "bytes":
        return {"ru": results["ru_bytes"], "en": results["en_bytes"]}
    else:  # both
        return {
            "local_paths": {"ru": results["ru"], "en": results["en"]},
            "bytes": {"ru": results["ru_bytes"], "en": results["en_bytes"]}
        }