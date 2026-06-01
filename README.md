# Лабораторная 3: Lakehouse на Polars + Delta Lake

Пайплайн `bronze -> silver -> gold` для датасета задержек авиарейсов US Flight Delays или похожего CSV-датасета.

## Структура

- `src/` - код пайплайна.
- `notebooks/` - место для исследовательских ноутбуков.
- `logs/` - служебные выгрузки, включая пример time travel.
- `data/raw/` - входные CSV-файлы.
- `data/bronze`, `data/silver`, `data/gold` - локальные Delta-таблицы.

## Запуск

1. Положите CSV-файлы в `data/raw/`. Удобный вариант - один файл на год или месяц.
2. Запустите приложение и MLflow:

```bash
docker-compose up --build
```

MLflow UI будет доступен на `http://localhost:5000`.

Для локального запуска без Docker:

```bash
pip install -r requirements.txt
python src/main.py
```

## Bronze

`src/bronze_layer.py` читает все CSV из `data/raw/` в отсортированном порядке и записывает каждый файл отдельным батчем в Delta-таблицу `data/bronze`.

Первый батч создает таблицу, следующие пишутся в режиме `append`. Это дает историю версий Delta и имитирует инкрементальный приход данных. В Bronze добавляются технические поля:

- `source_file`
- `ingestion_timestamp`

Также включена эволюция схемы через `schema_mode="merge"`.

## Silver

`src/silver_layer.py` читает Bronze через `pl.scan_delta` и строит lazy-цепочку:

```python
df = pl.scan_delta(str(BRONZE_PATH))

df_clean = (
    df
    .filter(...)
    .with_columns(...)
    .select(...)
)
```

В Silver выполняется:

- удаление отмененных рейсов: `Cancelled == 0`;
- удаление строк без `ArrDelay`;
- простая фильтрация выбросов: `ArrDelay` от `-60` до `360` минут;
- нормализация категорий `airline`, `origin`, `dest`;
- признаки `hour`, `day_of_week`, `season`, `route`;
- выбор нужного подмножества колонок;
- запись с `partition_by=["year", "month"]`;
- повторный запуск через Delta `MERGE`, чтобы не плодить дубли.

Ключ MERGE:

```text
flight_date, airline, flight_number, origin, dest
```

## Gold

Создаются аналитические витрины и отдельная feature table для ML:

- `data/gold/analytics/airline_delay` - задержки и delay rate по авиакомпаниям;
- `data/gold/analytics/airport_delay` - задержки по аэропортам отправления и прибытия;
- `data/gold/analytics/route_delay` - задержки по маршрутам;
- `data/gold/analytics/time_delay` - задержки по дню недели, часу и сезону;
- `data/gold/ml_features` - feature table для ML с таргетами `arr_delay` и `is_delayed`.

`is_delayed = arr_delay > 15`.

В ML feature table также добавляются признаки нагрузки: количество рейсов за день по origin, destination, airline, route, а также почасовые признаки для origin и route.

## ML

`src/ml_pipeline.py` обучает и сравнивает модели:

- регрессия: `DummyMeanRegressor`, `LinearRegression`, `RidgeRegression`, `RandomForestRegressor`;
- классификация: `DummyMostFrequentClassifier`, `LogisticRegression`, `RandomForestClassifier`.

В MLflow логируются:

- параметры модели;
- список признаков;
- размер обучающей выборки;
- regression metrics: `mae`, `mse`, `rmse`, `r2`;
- classification metrics: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`;
- confusion matrix для классификации: `confusion_tn`, `confusion_fp`, `confusion_fn`, `confusion_tp`;
- сама sklearn-модель;
- feature importance для RandomForest;
- версия Gold feature table: `gold_table_version`.

ML-модели используют признаки, известные до рейса: расписание, маршрут, аэропорты, авиакомпанию, расстояние и агрегированные признаки нагрузки. `dep_delay` сохраняется в Gold для анализа, но не используется как ML-признак, чтобы не создавать утечку целевой информации.

## EDA и notebook

Рабочий notebook находится в:

```text
notebooks/eda_ml.ipynb
```

Он читает готовые Silver/Gold Delta-таблицы, выводит обзор датасета, распределение задержек, топы по Gold-витринам, пример Polars Lazy `.explain()` и компактный benchmark моделей с baseline.

По умолчанию ML обучается на сэмпле `50000` строк, чтобы локальный запуск не требовал сотни гигабайт памяти. Размер можно изменить:

```bash
$env:ML_SAMPLE_ROWS="200000"
python src/main.py
```

Tracking URI берется из переменной окружения `MLFLOW_TRACKING_URI`, поэтому один и тот же код работает локально и в Docker.

## Delta Lake возможности

Помимо обязательного `MERGE`, проект использует:

1. `OPTIMIZE` compaction: `DeltaTable(...).optimize.compact()`.
2. `Z-ORDER`: `DeltaTable(...).optimize.z_order(["origin", "dest", "hour"])`.
3. `VACUUM`: удаление устаревших файлов с retention 168 часов.
4. Time travel: чтение предыдущей версии Silver через `DeltaTable(path, version=previous_version)` и сохранение примера в `logs/silver_time_travel_previous_version.csv`.
5. Schema evolution в Bronze через `schema_mode="merge"`.

## Почему партиции `year`, `month`

Для авиарейсов типичны запросы по периоду: год, месяц, сезон, праздничные интервалы. Партиционирование Silver по `year` и `month` позволяет Delta/Polars отбрасывать лишние директории при чтении временных диапазонов. Это достаточно крупные партиции для локального проекта и при этом не создает слишком много мелких директорий, как могло бы быть при партиционировании по дню.

## Пример Polars `.explain()`

Запрос:

```python
query = (
    pl.scan_delta("data/silver")
    .filter((pl.col("year") == 2024) & (pl.col("month") == 1))
    .select(["airline", "origin", "hour", "arr_delay"])
    .group_by(["airline", "origin", "hour"])
    .agg(pl.col("arr_delay").mean().alias("avg_arr_delay"))
)

print(query.explain())
```

Фактический вывод на локальной Silver-таблице:

```text
AGGREGATE[maintain_order: false]
  [col("arr_delay").mean().alias("avg_arr_delay")] BY [col("airline"), col("origin"), col("hour")]
  FROM
  simple π 4/4 ["airline", "origin", "hour", ... 1 other column]
    Parquet SCAN [data/silver/year=2015/month=1/part-00000-...zstd.parquet]
    PROJECT 6/16 COLUMNS
    SELECTION: [([(col("month")) == (1)]) & ([(col("year")) == (2015)])]
    ESTIMATED ROWS: ...
```

В плане видны:

- `PROJECT ... COLUMNS` - читаются только нужные колонки;
- `SELECTION` - фильтр проталкивается к скану;
- путь `year=2024/month=1` показывает partition pruning.
