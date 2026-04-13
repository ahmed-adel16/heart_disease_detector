# Accuracy Comparison Report

## Expert System

The expert system uses simple dictionary rules. It is easy to understand, but it depends on fixed human-written conditions.  
Its main strength is explainability. Its main weakness is that it may miss cases that do not match the stored rules.

Test results on the same dataset:

- Accuracy: `0.395122`
- Precision: `0.424000`
- Recall: `0.504762`
- F1-score: `0.460870`

## Decision Tree Model

The decision tree model learns from the dataset and gives the following test results:

- Accuracy: `0.921951`
- Precision: `0.908257`
- Recall: `0.942857`
- F1-score: `0.925234`
- Best max depth: `6`
- Best min samples split: `2`

It is still explainable compared to many other machine learning models, but it is less direct than the handwritten rules.

## Final Comparison

| Metric | Expert System | Decision Tree |
|--------|--------------|---------------|
| Accuracy | 0.395 | 0.922 |
| Precision | 0.424 | 0.908 |
| Recall | 0.505 | 0.943 |
| F1-score | 0.461 | 0.925 |

- The expert system is simpler and easier to explain.
- The decision tree is better for learning patterns from data.
- The expert system has lower accuracy because it uses fixed rules.
- The decision tree is more suitable for prediction accuracy.
- Using both systems in one project gives a good comparison between symbolic AI and machine learning.
