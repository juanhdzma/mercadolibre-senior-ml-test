# Mercado Libre ML Assessment

Exploratory analysis for classifying marketplace listings as new or used. The final approach prioritizes interpretable, production-ready rules over a black-box classifier, using structured attributes available when a listing is created.

![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue) ![Python](https://img.shields.io/badge/python-data%20science-3776AB) ![Modeling](https://img.shields.io/badge/modeling-XGBoost-006600)

## Contents

- [Approach](#approach)
- [Feature decisions](#feature-decisions)
- [Results](#results)
- [Repository layout](#repository-layout)
- [Reviewing the analysis](#reviewing-the-analysis)
- [License](#license)

## Approach

The task is to classify a listing as new or used at publication time. The analysis treats exploratory work as the primary deliverable: it identifies stable patterns in structured listing attributes and translates them into explicit decision rules.

An XGBoost tree is used to surface useful splits and evaluate the approach, not as an opaque production model. The resulting logic remains auditable and can be reproduced in application code, SQL, or an ETL pipeline without model-serving infrastructure.

## Feature decisions

Only attributes available when a listing is created are eligible. Post-publication fields such as sold quantity and stop time are excluded, as are identifiers and unstructured fields that would require disproportionate processing for this scope.

The analysis normalizes boolean and categorical signals from listing, shipping, warranty, and seller attributes. It evaluates title-based NLP, but rejects it because its operational cost and inconsistent results do not justify it for this use case.

## Results

The decision-tree analysis reaches:

| Metric | Result |
| --- | ---: |
| Accuracy | 0.8678 |
| Precision | 0.9051 |

Precision is the primary metric because classifying a used listing as new has the highest expected user-facing cost. The validation process evaluates rule stability across 100 shuffled dataset partitions.

![Confusion matrix](docs/Matrix.png)

![Validation results](docs/Validation.png)

The complete extracted tree is available in [`docs/Tree_rules.pdf`](docs/Tree_rules.pdf).

## Repository layout

```text
EDA.ipynb                 Exploratory analysis and rule derivation
utils/data_converter.py   Attribute cleaning and normalization
utils/file_converter.py   Input conversion helpers
utils/new_or_used.py      Rule-based classification logic
utils/plot_helper.py      Visualization helpers
docs/                     Requirements and evaluation artifacts
```

## Reviewing the analysis

Open [`EDA.ipynb`](EDA.ipynb) in a Jupyter-compatible Python environment and run its cells in order. The notebook documents the feature-selection rationale, the XGBoost tree analysis, the metrics, and the stability validation.

## License

[AGPL-3.0](LICENSE)
