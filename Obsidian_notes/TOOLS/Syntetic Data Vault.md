Topics:[[@DATA AUGMENTATION]] 
Sources : [The Synthetic Data Vault. Put synthetic data to work!](https://sdv.dev/)
Projets : [[Grand Défi 6 - Une IA digne de confiance pour les systèmes critiques]] [[Grand Défi 5 - Apprentissage automatique faiblement supervisé, vers une IA plus générale]]
Tags : #library #github
Date : 2022-10-11
***
[GitHub - sdv-dev/SDV: Synthetic Data Generation for tabular, relational and time series data.](https://github.com/sdv-dev/SDV)


## Overview
The **Synthetic Data Vault (SDV)** is a **Synthetic Data Generation** ecosystem of libraries that allows users to easily learn [single-table](https://sdv.dev/SDV/user_guides/single_table/index.html), [multi-table](https://sdv.dev/SDV/user_guides/relational/index.html) and [timeseries](https://sdv.dev/SDV/user_guides/timeseries/index.html) datasets to later on generate new **Synthetic Data** that has the **same format and statistical properties** as the original dataset.

Synthetic data can then be used to supplement, augment and in some cases replace real data when training Machine Learning models. Additionally, it enables the testing of Machine Learning or other data dependent software systems without the risk of exposure that comes with data disclosure.

## [Current functionality and features](https://sdv.dev/SDV/#current-functionality-and-features "Permalink to this headline")

-   Synthetic data generators for [single table datasets](https://sdv.dev/SDV/user_guides/single_table/index.html#single-table) with the following features:
    -   Using [Copulas](https://sdv.dev/SDV/user_guides/single_table/gaussian_copula.html#gaussian-copula) and [Deep Learning](https://sdv.dev/SDV/user_guides/single_table/ctgan.html#ctgan) based models.
    -   Handling of multiple data types and missing data with minimum user input.
    -   Support for [pre-defined and custom constraints](https://sdv.dev/SDV/user_guides/single_table/constraints.html#single-table-constraints) and data validation.
-   Synthetic data generators for [complex, multi-table, relational datasets](https://sdv.dev/SDV/user_guides/relational/index.html#relational) with the following features:
    -   Definition of entire [multi-table datasets metadata](https://sdv.dev/SDV/user_guides/relational/relational_metadata.html#relational-metadata) with a custom and flexible [JSON schema](https://sdv.dev/SDV/developer_guides/sdv/metadata.html#metadata-schema).
    -   Using Copulas and recursive modeling techniques.
-   Synthetic data generators for [multi-type, multi-variate timeseries datasets](https://sdv.dev/SDV/user_guides/timeseries/index.html#timeseries) with the following features:
    -   Using statistical, Autoregressive and Deep Learning models.
    -   Conditional sampling based on contextual attributes.
-   Metrics for [Synthetic Data Evaluation](https://sdv.dev/SDV/user_guides/evaluation/index.html#evaluation), including:
    -   An easy to use [Evaluation Framework](https://sdv.dev/SDV/user_guides/evaluation/evaluation_framework.html#evaluation-framework) to evaluate the quality of your synthetic data with a single line of code.
    -   Metrics for multiple data modalities, including [Single Table Metrics](https://sdv.dev/SDV/user_guides/evaluation/single_table_metrics.html#single-table-metrics) and [Multi Table Metrics](https://sdv.dev/SDV/user_guides/evaluation/multi_table_metrics.html#multi-table-metrics).
-   A [Benchmarking Framework](https://sdv.dev/SDV/user_guides/benchmarking/index.html#benchmarking-framework) to easily compare multiple synthetic data generators, including:
    -   Dozens of datasets of multiple data modalities already prepared to be run on.
    -   Tools to easily add new synthetic data generators and datasets.
    -   Distributed computing to reduce computing times.
    -   Comprehensive results presented in multiple leaderboard formats.


## Benchmark framework : SDGym
[GitHub - sdv-dev/SDGym: Benchmarking synthetic data generation methods.](https://github.com/sdv-dev/SDGym)

## Notes
TVAE >> CTGAN for data augmentation ([GAN-VAE-to-generate-Synthetic-Tabular-Data/gan.ipynb at main · saha0073/GAN-VAE-to-generate-Synthetic-Tabular-Data · GitHub](https://github.com/saha0073/GAN-VAE-to-generate-Synthetic-Tabular-Data/blob/main/gan.ipynb))

- [How to evaluate synthetic data for your project](https://datacebo.com/blog/how-to-evaluate-synthetic-data)