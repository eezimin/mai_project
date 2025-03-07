# AB-framework for Dispersion Reduction and MDE Improvement 

## Overview
Этот проект посвящен разработке AB-фреймворка, направленного на снижение дисперсии, снижение MDE (минимально детектируемого эффекта) и, соответственно, ускорение проведение экспериментов с использованием ML. Мы сосредоточим свое внимание на методах CUPED (Controlled-experiment Using Pre-Experiment Data) и ML-CUPED, который позволяет контролировать эксперимент с помощью прогнозов как ковариатов. 

Методы CUPED и ML-CUPED являются мощным инструментом для улучшения эффективности экспериментов, так как они использует прогнозы, которые могут продемонстрировать различные эффекты от тестируемого продукта или услуги. Это позволяет оптимизировать процесс тестирования, делая его более целенаправленным и эффективным.

## Важность для бизнеса
Улучшение MDE и снижение дисперсии имеет важное значение для бизнеса, так как это позволяет принимать более обоснованные решения при проведении A/B-тестирования. С помощью AB-фреймворка, реализующего методы CUPED и ML-CUPED, компании могут ожидать следующих преимуществ:

1. **Ускоренное принятие решений**: Более точные прогнозы позволяют быстрее проводить эксперименты, что сокращает время, затрачиваемое на определение эффективных стратегий.

2. **Повышенная эффективность**: CUPED и ML-CUPED помогают выявлять подгруппы участников, для которых тестируемый продукт или услуга могут иметь различный эффект. Это позволяет оптимизировать предложение, делая его более персонализированным и эффективным, что в конечном итоге повышает конверсию.

3. **Более точное прогнозирование**: Снижение дисперсии ведет к более точному прогнозированию, что помогает избежать ошибочных выводов и неудачных решений.

4. **Экономия ресурсов**: Использование прогнозов в качестве ковариатов позволяет сократить объем данных, необходимых для достижения значимых результатов, что экономит ресурсы и сокращает издержки на проведение испытаний.

## Цели проекта
Целями данного проекта являются:

1. Реализация AB-фреймворка с фокусом на методы CUPED, включающие в себя модели машинного обучения для прогнозирования.

2. Исследование и сравнение различных методов машинного обучения для улучшения точности прогнозов и, как следствие, эффективности фреймворка.

3. Предоставление практического руководства по использованию фреймворка, включая инструкции по установке, конфигурации и интерпретации результатов.

4. Демонстрация эффективности предлагаемого подхода с помощью реальных кейсов.

--- 

Надеемся, что этот проект поможет улучшить процесс A/B-тестирования и повысит эффективность бизнес-решений.

## Основные возможности

- Расчет MDE для различных метрик и длительностей эксперимента.
- Поддержка классического t-теста и методов CUPED.
- Возможность использования моделей машинного обучения для модификации CUPED.
- Визуализация результатов с помощью графиков.

## Установка

1. Клонируйте репозиторий:
`git clone https://github.com/eezimin/ab-test-duration-calculator.git`

2. Установите необходимые зависимости:
`pip install -r requirements.txt`

## Использование

Примеры использования приведены в AB_desing.ipynb, AB_analysis.ipynb

## Вклад
Если вы хотите внести свой вклад в проект, пожалуйста, следуйте этим шагам:

- Создайте форк репозитория.
- Создайте новую ветку для своих изменений.
- Внесите изменения и создайте коммит.
- Отправьте свои изменения в форк.
- Создайте пулл-реквест.


## Контакты
Если у вас есть какие-либо вопросы или предложения, пожалуйста, свяжитесь с нами по адресу eezimin@yandex.ru.




