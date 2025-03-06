from collections import namedtuple
from typing import Union

import numpy as np
import pandas as pd
import scipy.stats as sps

ExperimentComparisonResults = namedtuple('ExperimentComparisonResults', 
                                        ['pvalue', 'effect', 'ci_length', 'left_bound', 'right_bound', 'var'])


# Абсолютный ttest (для расчета абсолютной оценки эффекта: ET-EC)
def absolute_ttest(control, test):
    mean_control = np.mean(control)
    mean_test = np.mean(test)
    var_mean_control  = np.var(control) / len(control)
    var_mean_test  = np.var(test) / len(test)
    
    difference_mean = mean_test - mean_control
    difference_mean_var = var_mean_control + var_mean_test
    difference_distribution = sps.norm(loc=difference_mean, scale=np.sqrt(difference_mean_var))

    left_bound, right_bound = difference_distribution.ppf([0.025, 0.975])
    ci_length = (right_bound - left_bound)
    pvalue = 2 * min(difference_distribution.cdf(0), difference_distribution.sf(0))
    effect = difference_mean
    return ExperimentComparisonResults(pvalue, effect, ci_length, left_bound, right_bound, difference_mean_var)


# Относительный ttest (для расчета относительной оценки эффекта: (ET-EC) / EC)
def relative_ttest(control: Union[np.ndarray, pd.Series], 
                   test: Union[np.ndarray, pd.Series], 
                   alpha: float = 0.05) -> ExperimentComparisonResults:
    """
    Проводит относительный t-тест для сравнения двух выборок.

    Args:
        control (Union[np.ndarray, pd.Series]): Контрольная выборка.
        test (Union[np.ndarray, pd.Series]): Тестовая выборка.
        alpha (float, optional): Уровень значимости. По умолчанию 0.05.

    Returns:
        ExperimentComparisonResults: Результаты сравнения эксперимента, включающие p-значение,
            эффект, длину доверительного интервала и его границы.
    """
    if isinstance(control, pd.Series):
        control = control.values
    if isinstance(test, pd.Series):
        test = test.values
    
    mean_control = np.mean(control)
    var_mean_control  = np.var(control) / len(control)

    difference_mean = np.mean(test) - mean_control
    difference_mean_var  = np.var(test) / len(test) + var_mean_control
    
    covariance = -var_mean_control

    relative_mu = difference_mean / mean_control
    relative_var = difference_mean_var / (mean_control ** 2) \
                    + var_mean_control * ((difference_mean ** 2) / (mean_control ** 4))\
                    - 2 * (difference_mean / (mean_control ** 3)) * covariance
    relative_distribution = sps.norm(loc=relative_mu, scale=np.sqrt(relative_var))
    left_bound, right_bound = relative_distribution.ppf([alpha/2, 1 - alpha/2])
    
    ci_length = (right_bound - left_bound)
    pvalue = 2 * min(relative_distribution.cdf(0), relative_distribution.sf(0))
    effect = relative_mu
    return ExperimentComparisonResults(pvalue, effect, ci_length, left_bound, right_bound, difference_mean_var)


# Далее идет реализация классического CUPED критерия

def cuped_ttest(control, test, control_before, test_before):
    theta = (np.cov(control, control_before)[0, 1] + np.cov(test, test_before)[0, 1]) /\
                (np.var(control_before) + np.var(test_before))
    control_cup = control - theta * control_before
    test_cup = test - theta * test_before
    return absolute_ttest(control_cup, test_cup)

def relative_cuped(control: Union[np.ndarray, pd.Series], 
                   test: Union[np.ndarray, pd.Series],
                   control_before: Union[np.ndarray, pd.Series], 
                   test_before: Union[np.ndarray, pd.Series],
                   alpha: float = 0.05) -> ExperimentComparisonResults:
    """
    Проводит относительный CUPED (Controlled-experiment Using Pre-Experiment Data) анализ для сравнения двух выборок
    с учетом данных до эксперимента.

    Args:
        control (Union[np.ndarray, pd.Series]): Контрольная выборка после эксперимента.
        test (Union[np.ndarray, pd.Series]): Тестовая выборка после эксперимента.
        control_before (Union[np.ndarray, pd.Series]): Контрольная выборка до эксперимента.
        test_before (Union[np.ndarray, pd.Series]): Тестовая выборка до эксперимента.
        alpha (float, optional): Уровень значимости. По умолчанию 0.05.

    Returns:
        ExperimentComparisonResults: Результаты сравнения эксперимента, включающие p-значение,
            эффект, длину доверительного интервала и его границы.
    """
    if isinstance(control, pd.Series):
        control = control.values
    if isinstance(test, pd.Series):
        test = test.values
    if isinstance(control_before, pd.Series):
        control_before = control_before.values
    if isinstance(test_before, pd.Series):
        test_before = test_before.values
        
    theta = (np.cov(control, control_before)[0, 1] + np.cov(test, test_before)[0, 1]) /\
                (np.var(control_before) + np.var(test_before))

    control_cup = control - theta * control_before
    test_cup = test - theta * test_before

    mean_den = np.mean(control)
    mean_num = np.mean(test_cup) - np.mean(control_cup)
    var_mean_den  = np.var(control) / len(control)
    var_mean_num  = np.var(test_cup) / len(test_cup) + np.var(control_cup) / len(control_cup)

    cov = -np.cov(control_cup, control)[0, 1] / len(control)

    relative_mu = mean_num / mean_den
    relative_var = var_mean_num / (mean_den ** 2)  + var_mean_den * ((mean_num ** 2) / (mean_den ** 4))\
                - 2 * (mean_num / (mean_den ** 3)) * cov
    
    relative_distribution = sps.norm(loc=relative_mu, scale=np.sqrt(relative_var))
    left_bound, right_bound = relative_distribution.ppf([alpha/2, 1 - alpha/2])
    
    ci_length = (right_bound - left_bound)
    pvalue = 2 * min(relative_distribution.cdf(0), relative_distribution.sf(0))
    effect = relative_mu
    return ExperimentComparisonResults(pvalue, effect, ci_length, left_bound, right_bound, relative_var)
