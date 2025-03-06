# Стандартные пакеты Python
import datetime
from collections import namedtuple
from typing import Union, List, Optional

# Сторонние пакеты
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest
from sklearn.model_selection import train_test_split
from statsmodels.stats.proportion import proportion_confint
from tqdm.auto import tqdm

# Пользовательские пакеты
from ml_prediction import *
from stats_criteria import *

# Определение пользовательских типов данных
ExperimentComparisonResults = namedtuple('ExperimentComparisonResults', 
                                        ['pvalue', 'effect', 'ci_length', 'left_bound', 'right_bound'])


def choose_date(data: pd.DataFrame, 
                metric: str, 
                start_date: Union[str, datetime.datetime], 
                duration: int, 
                share: float = 0.5, 
                random_state: int = 42, 
                unit_name: str = 'user_id') -> pd.DataFrame:
    """
    Выбирает данные за определенный период времени и случайным образом отбирает заданную долю уникальных значений unit_name.

    Args:
        data (pd.DataFrame): Входной DataFrame, содержащий данные.
        metric (str): Название столбца с метрикой.
        start_date (Union[str, datetime.datetime]): Дата начала периода в формате 'YYYY-MM-DD' или объект datetime.
        duration (int): Длительность периода в днях.
        share (float, optional): Доля уникальных значений unit_name для случайного отбора. По умолчанию 0.5.
        random_state (int, optional): Зерно для генератора случайных чисел. По умолчанию 42.
        unit_name (str, optional): Название столбца, содержащего идентификаторы объектов. По умолчанию 'user_id'.

    Returns:
        pd.DataFrame: DataFrame с отобранными данными.
    """
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = start_date + datetime.timedelta(days=duration)
    
    tmp = data[(data.event_date >= start_date) & (data.event_date <= end_date)]   
    tmp = tmp.groupby([unit_name])[metric].sum().reset_index()

    # Отбираем заданную долю уникальных значений unit_name (случайным образом)
    unique_units = pd.Series(tmp[unit_name].unique())
    selected_units = unique_units.sample(frac=share, random_state=random_state)
    result = tmp[tmp[unit_name].isin(selected_units)].reset_index(drop=True)
    
    return result


def MDE_calc(data: pd.DataFrame, 
             metric: str, 
             start_date: Union[str, datetime.datetime],
             duration_range: List[int] = [7, 14, 21, 28, 35],
             share: float = 0.5,
             alpha: float = 0.05, 
             power: float = 0.8) -> None:
    """
    Вычисляет минимальный обнаруживаемый эффект (MDE) для различных длительностей эксперимента.

    Args:
        data (pd.DataFrame): Входной DataFrame, содержащий данные.
        metric (str): Название столбца с метрикой.
        start_date (Union[str, datetime.datetime]): Дата начала периода в формате 'YYYY-MM-DD' или объект datetime.
        duration_range (List[int], optional): Список длительностей эксперимента в днях. По умолчанию [7, 14, 21, 28, 35].
        share (float, optional): Доля уникальных значений unit_name для случайного отбора. По умолчанию 0.5.
        alpha (float, optional): Уровень значимости. По умолчанию 0.05.
        power (float, optional): Мощность теста. По умолчанию 0.8.

    Returns:
        None
    """
    for duration in duration_range:
        x1 = choose_date(data, metric, start_date, duration, share)
        mean = x1[metric].mean()
        std = x1[metric].std()
        
        effect_size = tt_ind_solve_power(
            effect_size=None, 
            alpha=alpha, 
            power=power, 
            nobs1=x1.shape[0],
            ratio=1, 
            alternative='two-sided'
        )

        result_effect = effect_size / (mean/std)    
        print(f'Число юнитов {x1.shape[0]}, длительность {duration} дней => MDE = {round(result_effect*100, 2)}%')
    print()

    
def MDE_relative_ttest_calc(data: pd.DataFrame, 
                            metric_range: List[str], 
                            start_date: str,
                            duration_range: List[int] = [5, 10, 15, 20, 25, 30],
                            share: float = 0.5,
                            alpha: float = 0.05,
                            power: float = 0.8,
                            random_state: int = 42,
                            n_iters: int = 100,
                            to_plot: bool = True,
                            unit_name: str = 'user_id',
                            date_name: str = 'event_date') -> pd.DataFrame:
    """
    Вычисляет минимальный обнаруживаемый эффект (MDE) для относительного t-теста 
    при различных длительностях эксперимента и метриках.

    Args:
        data (pd.DataFrame): Входной DataFrame, содержащий данные.
        metric_range (List[str]): Список названий столбцов с метриками.
        start_date (str): Дата начала периода в формате 'YYYY-MM-DD'.
        duration_range (List[int], optional): Список длительностей эксперимента в днях. По умолчанию [7, 14, 21, 28, 35].
        share (float, optional): Доля данных для тестовой группы. По умолчанию 0.5.
        alpha (float, optional): Уровень значимости. По умолчанию 0.05.
        power (float, optional): Мощность теста. По умолчанию 0.8.
        random_state (int, optional): Зерно для генератора случайных чисел. По умолчанию 42.
        n_iters (int, optional): Количество итераций для усреднения MDE. По умолчанию 100.
        to_plot (bool, optional): Флаг для построения графиков. По умолчанию True.
        unit_name (str, optional): Название столбца, содержащего идентификаторы объектов. По умолчанию 'user_id'.
        date_name (str, optional): Название столбца, содержащего даты. По умолчанию 'event_date'.

    Returns:
        pd.DataFrame: DataFrame с результатами вычисления MDE.
    """

    mde_relative_ttest = []
    np.random.seed(random_state)

    for metric in tqdm(metric_range, desc='Processing metrics'):
        for duration in tqdm(duration_range, desc=f'Processing durations for metric {metric}', leave=False):

            # отберем данные за промежуток времени от start_date до start_date + duration
            start_date_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end_date_dt = (start_date_dt + datetime.timedelta(days=duration))
            tmp = (data[(data[date_name] >= start_date_dt) & (data[date_name] <= end_date_dt)])

            # сагрегируем по юнитам за весь период эксперимента
            tmp = tmp.groupby([unit_name])[metric].sum().reset_index()

            # далее мы много раз делаем случайное сплитование и находим среднее MDE и ДИ для MDE
            mde_li = []
            for _ in range(n_iters):
                control, test = train_test_split(tmp, test_size=share) 
                mde = relative_ttest(control[metric], test[metric]).ci_length / 2 * 100
                mde_li.append(mde)
            mde_avg = round(np.mean(mde_li), 2)   
            left_mde, right_mde = np.quantile(mde_li, [0.025, 0.975])

            # сохраняем результаты
            dataframe = pd.DataFrame({'mde': [mde_avg]}, index=[0])
            dataframe['left_mde'] = left_mde
            dataframe['right_mde'] = right_mde
            dataframe['metric'] = metric
            dataframe['total_manager_cnt'] = tmp.shape[0]
            dataframe['test_group'] = int(tmp.shape[0] * share)
            dataframe['duration'] = duration
            dataframe['pre_duration'] = None
            mde_relative_ttest.append(dataframe)
        
    mde_relative_ttest = pd.concat(mde_relative_ttest, ignore_index = True)
    
    if to_plot:
        mde_plot(mde_relative_ttest) 
    
    return mde_relative_ttest


def mde_plot(data: pd.DataFrame) -> None:
    """
    Строит графики зависимости MDE от длительности эксперимента для каждой метрики.

    Args:
        data (pd.DataFrame): DataFrame с результатами вычисления MDE.

    Returns:
        None
    """
    for metric in data['metric'].unique():
        plt.figure(figsize=(8, 4))
        tmp = data[data['metric'] == metric]

        # Построение графика MDE от длительности
        plt.plot(tmp['duration'], tmp['mde'])
        plt.fill_between(tmp['duration'], tmp['left_mde'], tmp['right_mde'], alpha=0.05)  

        # Добавляем подписи к графику
        plt.xlabel('Длительность эксперимента, в днях')
        plt.ylabel('MDE, %')
        plt.title(f'Зависимость MDE от длительность эксперимента для метрики "{metric}"')
        plt.grid(True)
            
        plt.show()
        

def calc_metric_for_exp_periods(data: pd.DataFrame, 
                                start_date: str, 
                                duration: int, 
                                metric: str, 
                                premetric_days: int = 10, 
                                unit_name: str = 'user_id', 
                                date_name: str = 'event_date') -> pd.DataFrame:
    """
    Вычисляет значения метрики для экспериментального и предэкспериментального периодов.

    Args:
        data (pd.DataFrame): Входной DataFrame, содержащий данные.
        start_date (str): Дата начала экспериментального периода в формате 'YYYY-MM-DD'.
        duration (int): Длительность экспериментального периода в днях.
        metric (str): Название столбца с метрикой.
        premetric_days (int, optional): Количество дней в предэкспериментальном периоде. По умолчанию 10.
        unit_name (str, optional): Название столбца, содержащего идентификаторы объектов. По умолчанию 'user_id'.
        date_name (str, optional): Название столбца, содержащего даты. По умолчанию 'event_date'.

    Returns:
        pd.DataFrame: DataFrame с значениями метрики для экспериментального и предэкспериментального периодов.
    """
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = start_date_dt + pd.Timedelta(days=duration)
    premetric_end_dt = start_date_dt - pd.Timedelta(days=1)
    premetric_start_dt = premetric_end_dt - pd.Timedelta(days=premetric_days)

    metric_data = (
        data.loc[(data[date_name] >= start_date_dt) & (data[date_name] <= end_date_dt)]
        .groupby(unit_name)[metric]
        .sum()
        .reset_index()
        .rename(columns={metric: 'metric'})
    )

    premetric_data = (
        data.loc[(data[date_name] >= premetric_start_dt) & (data[date_name] <= premetric_end_dt)]
        .groupby(unit_name)[metric]
        .sum()
        .reset_index()
        .rename(columns={metric: 'premetric'})
    )

    all_metric_data = pd.merge(premetric_data, metric_data, on=unit_name, how='outer').fillna(0)

    return all_metric_data

def MDE_relative_cuped_calc(data: pd.DataFrame, 
                            metric_range: list, 
                            start_date: str,
                            duration_range: List[int] = [5, 10, 15, 20, 25, 30],
                            share: float = 0.5,
                            alpha: float = 0.05,
                            power: float = 0.8,
                            random_state: int = 42,
                            n_iters: int = 100,
                            premetric_days: int = 10,
                            to_plot: bool = True,
                            unit_name: str = 'user_id',
                            date_name: str = 'event_date',
                            ML_flg: bool = False,
                            test_size_ML: float = 0.2,
                            model_params: dict = None) -> pd.DataFrame:
    """
    Вычисляет минимальный обнаруживаемый эффект (MDE) для относительного CUPED теста 
    при различных длительностях эксперимента и метриках.

    Args:
        data (pd.DataFrame): Входной DataFrame, содержащий данные.
        metric_range (list): Список названий столбцов с метриками.
        start_date (str): Дата начала экспериментального периода в формате 'YYYY-MM-DD'.
        duration_range (list, optional): Список длительностей эксперимента в днях. По умолчанию [5, 10, 15, 20].
        share (float, optional): Доля данных для тестовой группы. По умолчанию 0.5.
        alpha (float, optional): Уровень значимости. По умолчанию 0.05.
        power (float, optional): Мощность теста. По умолчанию 0.8.
        random_state (int, optional): Зерно для генератора случайных чисел. По умолчанию 42.
        n_iters (int, optional): Количество итераций для усреднения MDE. По умолчанию 100.
        premetric_days (int, optional): Количество дней в предэкспериментальном периоде. По умолчанию 10.
        to_plot (bool, optional): Флаг для построения графиков. По умолчанию True.
        unit_name (str, optional): Название столбца, содержащего идентификаторы объектов. По умолчанию 'user_id'.
        date_name (str, optional): Название столбца, содержащего даты. По умолчанию 'event_date'.
        ML_flg (bool, optional): Флаг для использования ML критерия предсказания ковариаты. По умолчанию False.
        test_size_ML (float, optional): Размер тестовой выборки для ML модели. По умолчанию 0.2.
        model_params (dict): Словарь с параметрами ML-модели. По умолчанию None.

    Returns:
        pd.DataFrame: DataFrame с результатами вычисления MDE.
    """
    mde_relative_cuped_strat = []
    np.random.seed(random_state)

    for metric in tqdm(metric_range, desc='Processing metrics'):
        for duration in tqdm(duration_range, desc=f'Processing durations for metric {metric}', leave=False):
            if ML_flg:
                # в случае True используем ML критерий для предсказания ковариаты (ML Cuped)
                df_cuped = ML_modification(data=data, 
                                           start_date=start_date,
                                           duration=duration,
                                           metric=metric,
                                           unit_name=unit_name,
                                           date_name=date_name,
                                           random_state=random_state,
                                           **model_params)
            else:   
                # считаем метрики в периоде и в предпериоде обычным способом (эвристика, как в CUPED)
                df_cuped = calc_metric_for_exp_periods(data=data, 
                                                       start_date=start_date, 
                                                       duration=duration, 
                                                       metric=metric,
                                                       premetric_days=premetric_days,
                                                       unit_name=unit_name,
                                                       date_name=date_name)
            
            # много раз считаем MDE и усредняем (заодно проверим корректность)
            
            mde_results = []
            for _ in range(n_iters):
                control, test = train_test_split(df_cuped, test_size=share) 
                mde_result = relative_cuped(control['metric'], test['metric'], control['premetric'], test['premetric'], alpha=alpha)
                mde_results.append(mde_result)
            
            mde_values = [result.ci_length / 2 * 100 for result in mde_results]
            p_values = [result.pvalue for result in mde_results]
            bad_cnt = sum(pvalue < alpha for pvalue in p_values)
            
            mde_avg = np.mean(mde_values)
            left_mde, right_mde = np.quantile(mde_values, [alpha/2, 1-alpha/2])
            bad_cnt_pct = bad_cnt / n_iters * 100
            left_real_level, right_real_level = proportion_confint(count=bad_cnt, 
                                                                   nobs=n_iters, 
                                                                   alpha=alpha, 
                                                                   method='wilson')
            # сохраняем результаты
            dataframe = pd.DataFrame({
                'mde': [mde_avg],
                'left_mde': left_mde,
                'right_mde': right_mde,
                'metric': metric,
                'total_unit_cnt': df_cuped.shape[0],
                'test_group': int(df_cuped.shape[0] * share),
                'duration': duration,
                'pre_duration': premetric_days,
                'bad_cnt_pct': bad_cnt_pct,
                'FPR_test': (left_real_level <= alpha <= right_real_level),
                'uniform_pvalues': (kstest(p_values, 'uniform').pvalue >= alpha)
            })    
            
            mde_relative_cuped_strat.append(dataframe)
    mde_relative_cuped_strat = pd.concat(mde_relative_cuped_strat, ignore_index = True)
    if to_plot:
        mde_plot(mde_relative_cuped_strat)    
    return mde_relative_cuped_strat        