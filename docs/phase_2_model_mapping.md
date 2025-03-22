# Model / classifier mapping

| ID | Requested model | Features | Classifier | Relation to paper |
|---|---|---|---|---|
| 01 | Random | none | random | Paper family or explicitly requested duplicate |
| 02 | Majority | none | majority | Paper family or explicitly requested duplicate |
| 03 | Harbingers | harbingers | logistic_regression | Paper family or explicitly requested duplicate |
| 04 | Logistic Regression | numerical | logistic_regression | New engineering baseline |
| 05 | Harbingers Lr | harbingers | logistic_regression | Paper family or explicitly requested duplicate |
| 06 | Harbingers Power | harbingers+power | logistic_regression | Paper family or explicitly requested duplicate |
| 07 | Harbingers Lr Power | harbingers+power | logistic_regression | Paper family or explicitly requested duplicate |
| 08 | Bow Lr | bow | logistic_regression | Paper family or explicitly requested duplicate |
| 09 | Bow Power | bow+power | logistic_regression | Paper family or explicitly requested duplicate |
| 10 | Bow Lr Power | bow+power | logistic_regression | Paper family or explicitly requested duplicate |

Paper §4.2 and Figure 5 describe **logistic regression** with Harbingers, BoW, and optional power. Models 3/5, 6/7, and 9/10 are therefore deliberately overlapping configurations, tracked by distinct IDs but not distinct classification algorithms. Model 4 is a newly specified length-statistics LR baseline and is not in the paper.

The printed Table 8 **truncates** subjectivity entries; we only transcribed the visible part. This prohibits claiming an exact Harbingers reproduction. Phrase matching/counting, power standardization, BoW tokenization, and searched hyperparameters are documented implementation decisions, not verified original hyperparameters.
