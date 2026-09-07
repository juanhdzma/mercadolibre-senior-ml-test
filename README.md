# Mercado Libre ML Assessment

Exploratory analysis for classifying marketplace listings as new or used. The assessment derives interpretable decision rules from structured listing attributes available when a listing is created, rather than treating a black-box predictive model as the final deliverable.

![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue) ![Python](https://img.shields.io/badge/python-data%20science-3776AB) ![Modeling](https://img.shields.io/badge/modeling-XGBoost-006600)

## Contents

- [Problem and decision](#problem-and-decision)
- [Feature-selection assumptions](#feature-selection-assumptions)
- [Exploratory analysis](#exploratory-analysis)
- [Text evaluation](#text-evaluation)
- [Rule extraction](#rule-extraction)
- [Validation and metrics](#validation-and-metrics)
- [Risks and evolution](#risks-and-evolution)
- [Repository layout](#repository-layout)
- [Reviewing the analysis](#reviewing-the-analysis)
- [License](#license)

## Problem and decision

The task is to classify a listing as new or used at publication time. The key constraint is that every input must be available when the listing is created; information that appears after publication cannot be used to make the decision.

The initial material included a train-test path, but the assessment deliberately starts with exploratory analysis. The goal is to understand which stable, explainable attributes distinguish the classes and convert that understanding into logic that business and engineering teams can inspect.

The resulting rules can be implemented as application conditionals, SQL filters, or ETL transformations without a model-serving stack. An XGBoost tree is used as an analysis tool to discover and validate useful splits, not as an opaque production artifact.

## Feature-selection assumptions

The feature set follows three criteria:

- The attribute is available in the source JSON at listing creation time.
- Its meaning can be explained without relying on future behavior.
- Its operational value justifies its processing and maintenance cost.

The following transformations summarize the design:

| Attributes | Treatment | Reasoning |
| --- | --- | --- |
| `warranty`, `video_id`, `official_store_id` | Converted to presence booleans. | Presence is a more stable signal than the highly variable raw value. |
| `sub_status`, `tags` | Kept as categorical candidates for encoding. | Their values retain business meaning such as suspension or expiration status. |
| `local_pickup`, `free_shipping`, `has_dimensions`, `accepts_mercadopago` | Converted to booleans. | These listing and shipping attributes are available at publication and have direct operational meaning. |
| `seller_address`, `permalink`, `id` | Excluded. | They are identifiers, unstructured values, or do not provide direct predictive value in this scope. |
| `start_time`, `stop_time`, `created_time`, `sold_quantity`, `available_quantity` | Excluded. | They are unavailable, incomplete, or behaviorally contaminated for a decision made at creation time. |

This prevents leakage from post-publication data and keeps the final logic suitable for synchronous use at the point of listing creation.

## Exploratory analysis

The exploratory work compares the relative distribution of attributes between new and used listings. It prioritizes structural differences between the classes over raw frequency: a feature that consistently changes the proportion of new versus used listings is more useful than one that merely appears often.

The workflow cleans and normalizes nested shipping, seller, warranty, and listing attributes before comparing their class behavior. The full transformations are available in [`utils/data_converter.py`](utils/data_converter.py), while visualization helpers live in [`utils/plot_helper.py`](utils/plot_helper.py).

## Text evaluation

The `title` field was evaluated with NLP approaches, including zero-shot classification with `facebook/bart-large-mnli`. It was not included in the final solution for three reasons:

- Inference time is high for a lightweight, creation-time decision.
- Results on real titles were inconsistent after basic cleaning.
- The operational complexity does not justify the incremental value compared with structured attributes.

This is a scoped trade-off, not a claim that text is universally unhelpful. A future version with reliable labeled data and a latency budget could reevaluate title features.

## Rule extraction

XGBoost is configured to produce interpretable trees with limited depth. The objective is not to maximize a single offline score through a complex ensemble; it is to identify combinations of conditions that are stable enough to translate into business rules.

The tree helps answer three questions:

1. Which variables have useful discriminatory power?
2. Which combinations of conditions are explainable to non-technical reviewers?
3. Which rules achieve high precision without relying on unavailable information?

The extracted tree is published at [`docs/Tree_rules.pdf`](docs/Tree_rules.pdf). Each branch can be reviewed as a sequence of conditions and translated directly into the rule-based logic under [`utils/new_or_used.py`](utils/new_or_used.py).

## Validation and metrics

The analysis reports the following results:

| Metric | Result | Why it matters |
| --- | ---: | --- |
| Accuracy | 0.8678 | Measures overall correct classification. |
| Precision | 0.9051 | Controls false positives when a used product is labeled new. |

Precision is the primary metric because labeling a used listing as new creates the most direct user-facing cost: potential complaints, loss of trust, and post-sale friction. Labeling a new listing as used is still undesirable, but generally less harmful in this context.

![Confusion matrix](docs/Matrix.png)

The rules are also evaluated for stability across 100 shuffled dataset partitions. The metrics remain tightly grouped across those partitions, reducing the likelihood that the selected rules only fit one ordering of the sample.

![Validation results](docs/Validation.png)

This validation does not replace monitoring after deployment. It shows that the current rules are stable within the supplied data, not that they will remain stable as catalog composition or marketplace behavior changes.

## Risks and evolution

The main risks are feature drift, changing source schemas, and over-reliance on structured signals that may disappear or change semantics. A production implementation should log the applied rule, the relevant inputs, and the eventual observed condition when available.

Useful next steps include:

- Periodically validate the rules on new labeled samples and alert on false-positive changes.
- Version the rule set and the data transformations together.
- Reintroduce text features only when their measured value exceeds their latency and maintenance cost.
- Use a hybrid approach for ambiguous cases: explicit rules for confident decisions and a supervised model or manual review for the remainder.
- Add model monitoring, drift checks, and retraining only if the problem justifies the operational complexity.

## Repository layout

- [`EDA.ipynb`](EDA.ipynb): exploratory analysis, tree construction, metrics, and validation.
- [`utils/data_converter.py`](utils/data_converter.py): attribute cleaning and normalization.
- [`utils/file_converter.py`](utils/file_converter.py): input conversion helpers.
- [`utils/new_or_used.py`](utils/new_or_used.py): rule-based classification logic.
- [`utils/plot_helper.py`](utils/plot_helper.py): visualization helpers.
- [`docs/`](docs/): requirements, evaluation figures, and the extracted tree.

## Reviewing the analysis

Open [`EDA.ipynb`](EDA.ipynb) in a Jupyter-compatible Python environment and run its cells in order. The notebook documents the feature-selection rationale, exploratory comparisons, XGBoost tree analysis, metrics, and stability validation.

The project is intentionally an assessment artifact, not a deployed inference service. It does not include a packaged API, automated training pipeline, or runtime monitoring configuration.

## License

[AGPL-3.0](LICENSE)
