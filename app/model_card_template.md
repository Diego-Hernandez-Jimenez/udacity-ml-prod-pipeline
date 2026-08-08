# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- **Algorithm:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)
- **Hyperparameters:** `n_estimators=100`, `max_depth=None`, `random_state=13`, `n_jobs=-1`
- **Categorical encoding:** One-Hot Encoding (`handle_unknown='ignore'`)
- **Label encoding:** `LabelBinarizer` (`<=50K` → 0, `>50K` → 1)
- **Training framework:** scikit-learn 1.7
- **Pipeline orchestration:** DVC (4 stages: preprocess → split → encode → train)
- **Serving:** FastAPI REST API, deployed on Render

## Intended Use

The model predicts whether an individual's annual income exceeds $50,000 based on demographic and employment characteristics from a U.S. Census record.

**Primary use case:** educational demonstration of an end-to-end ML production pipeline (data versioning, CI/CD, model serving).

**Out-of-scope uses:** real-world hiring or credit decisions, any application where the prediction influences legally protected outcomes, extrapolation to populations outside the United States or to years significantly beyond 1994.

## Training Data

- **Source:** UCI Machine Learning Repository — [Adult Census Income dataset](https://archive.ics.uci.edu/dataset/2/adult)
- **Original size:** ~48,842 rows, 14 features + 1 label
- **Preprocessing:** rows with missing values (encoded as `?` in the original) were dropped; whitespace was stripped from string fields. After cleaning, approximately 45,222 rows remain.
- **Train split:** 80% of the cleaned data (~36,177 rows), stratified on the label.
- **Features used:**
  - *Numerical (6):* `age`, `fnlgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`
  - *Categorical (8):* `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`

## Evaluation Data

- **Test split:** the remaining 20% of the cleaned dataset (~9,045 rows), stratified on the label, held out before any fitting.
- The encoder and label binarizer are fitted exclusively on the training split and applied without re-fitting to the test split.

## Metrics

Performance is reported on the held-out test split using the metrics below. For a breakdown by each categorical feature slice, run `python src/slice_metrics.py` from the `app/` directory; results are also saved to `slice_output.txt`.

| Metric    | Description                                                   |
|-----------|---------------------------------------------------------------|
| Precision | Fraction of positive predictions that are actually positive   |
| Recall    | Fraction of actual positives that are correctly predicted     |
| F1        | Harmonic mean of precision and recall (β = 1)                 |

> **Note:** Exact values depend on the trained artifact. Run the evaluation pipeline and consult `slice_output.txt` for up-to-date figures.

Key observations from slice analysis:
- Performance varies notably across `native-country` slices due to class imbalance in less-represented countries.
- The `sex` and `race` slices reflect historical biases present in the 1994 census data; see Ethical Considerations.

## Ethical Considerations

- **Historical bias:** the dataset reflects 1994 U.S. Census data and encodes socioeconomic disparities of that era. Patterns learned by the model (e.g., income differences by `sex`, `race`, or `native-country`) mirror historical inequalities, not causal relationships.
- **Protected attributes:** `race`, `sex`, and `native-country` are included as features. Any deployment that uses predictions to make decisions affecting individuals in these groups is subject to anti-discrimination law (e.g., Title VII, ECOA, FCRA) and should not proceed without a thorough fairness audit.
- **Slice disparity:** model performance is uneven across demographic slices. Groups with fewer training examples tend to have lower recall, meaning the model is less reliable for underrepresented populations.

## Caveats and Recommendations

- The income threshold of $50K corresponds to 1994 dollars (~$106K in 2024). The label has not been inflation-adjusted and may be misleading if interpreted literally today.
- The model is trained on a static snapshot; it does not account for economic, demographic, or labor-market changes since 1994.
- Before any production use beyond this demo, a full bias and fairness audit (e.g., using Aequitas or Fairlearn) should be conducted, and underrepresented slices should be examined closely.
- The `fnlgt` (final sampling weight) feature is included but its meaning is census-specific; it may introduce spurious correlations and should be reconsidered for any retraining effort.
