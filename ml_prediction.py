import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split


def ML_modification(data: pd.DataFrame, 
                    metric: str,
                    start_date: str, 
                    duration: int,
                    model: object = Ridge(),
                    param_grid: dict = None,                    
                    categorical_columns: list = [],
                    numerical_columns: list = [],
                    unit_name: str = 'user_id',
                    date_name: str = 'event_date',
                    random_state: int = 42,
                    test_size: float = 0.2,
                    verbose: bool = False) -> pd.DataFrame:
    """
    Модифицирует данные с помощью ML методов для подготовки ковариаты.

    Args:
        data (pd.DataFrame): Входной DataFrame, содержащий данные.
        metric (str): Название столбца с метрикой.
        start_date (str): Дата начала экспериментального периода в формате 'YYYY-MM-DD'.
        duration (int): Длительность экспериментального периода в днях.
        model (object, optional): Модель машинного обучения. По умолчанию Ridge().
        param_grid (dict, optional): Словарь с гиперпараметрами для поиска лучшей модели.
            Если None, то используется модель по умолчанию без поиска гиперпараметров.
            Пример: {'alpha': [0.1, 1, 10]} для Ridge().
            По умолчанию None.        
        categorical_columns (list, optional): Список категориальных столбцов. По умолчанию [].
        numerical_columns (list, optional): Список численных столбцов. По умолчанию [].
        unit_name (str, optional): Название столбца, содержащего идентификаторы объектов. По умолчанию 'user_id'.
        date_name (str, optional): Название столбца, содержащего даты. По умолчанию 'event_date'.
        random_state (int, optional): Зерно для генератора случайных чисел. По умолчанию 42.
        test_size (float, optional): Размер тестовой выборки. По умолчанию 0.2.
        verbose (bool): Флаг для вывода дополнительной информации (качество предсказания на тесте). По умолчанию False.

    Returns:
        pd.DataFrame: DataFrame с модифицированными данными.
    """
    # Определение порядкового номера периода
    data['period_num'] = (data['event_date'] - pd.to_datetime(start_date)).dt.days // duration
    
    # Сделаем группировку трат по каждому юзеру на каждый месяц-год
    agg_funcs = {col: 'sum' for col in numerical_columns + [metric]}
    agg_funcs.update({col: lambda x: x.mode().iloc[0] for col in categorical_columns})

    data_grouped = data.groupby([unit_name, 'period_num']).agg(agg_funcs).reset_index()
    
    # Сортировка данных по unit_name и period_num
    data_grouped = data_grouped.sort_values([unit_name, 'period_num'])

    # Создание новой колонки pre_spend
    data_grouped[f'{metric}_1_period_before'] = data_grouped.groupby(unit_name)[metric].shift(1).fillna(0)
    data_grouped[f'{metric}_2_period_before'] = data_grouped.groupby(unit_name)[metric].shift(2).fillna(0)
    
    # Два значения метрики предпериодов включаем в численные колонки и получаем общий список фич
    numerical_columns_with_premetrics = numerical_columns + [f'{metric}_1_period_before', f'{metric}_2_period_before']
    features = numerical_columns_with_premetrics + categorical_columns
    
    # Все категориальные колонки переводим в тип "Строка"
    data_grouped[categorical_columns] = data_grouped[categorical_columns].astype(str)
    
    # выделим обучающий и экспериментальный датасет
    training_dataset = data_grouped[data_grouped['period_num'] < 0].reset_index(drop=True)
    experimental_dataset = data_grouped[data_grouped['period_num'] == 0].reset_index(drop=True)
    
    # Разделение данных на признаки и целевую переменную    
    X_train = pd.get_dummies(training_dataset[features], columns=categorical_columns)
    y_train = training_dataset[metric]    

    # Обучение модели
    if param_grid is None:
        best_model = model
        best_model.fit(X_train, y_train)
    else:
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

    # После обучения модели сделаем предсказания для экспериментальной выборки
    X_test = pd.get_dummies(experimental_dataset[features], columns=categorical_columns)
    y_test = experimental_dataset[metric]
    
    y_pred_test = best_model.predict(X_test)
    
    if verbose:
        r2_test = r2_score(y_test, y_pred_test)
        print(f'Метрика {metric}, длительность {duration} дней, R-squared (на тесте): {r2_test:.2f}') 
    
    experimental_dataset[f'{metric}_predicted'] = y_pred_test
    experimental_dataset = experimental_dataset.rename({metric: 'metric', f'{metric}_predicted': 'premetric'}, axis=1)
    return experimental_dataset