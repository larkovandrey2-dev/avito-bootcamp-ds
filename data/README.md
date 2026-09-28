# Исходные данные

Положите в эту папку три файла из задания:

- `train.parquet` — история выбранных объявлений;
- `benchmark_queries.parquet` — запросы, для которых нужно построить ответ;
- `benchmark_items.parquet` — объявления benchmark-корпуса.

Сами parquet-файлы исключены из Git: вместе они занимают около 654 МБ и распространяются отдельно организаторами.

```text
data/
├── train.parquet
├── benchmark_queries.parquet
└── benchmark_items.parquet
```

Скрипт перед запуском проверяет наличие файлов и обязательных колонок.
