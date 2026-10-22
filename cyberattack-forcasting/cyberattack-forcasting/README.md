# Cyberattack Frequency Forecasting Using Attention-LSTM

## 1. Project Overview

This project investigates monthly cyberattack frequency forecasting using machine learning and deep learning techniques. It compares a Naive Baseline, Random Forest, and Attention-LSTM model to predict future cyberattack counts from historical observations.

The project is inspired by the research paper **"Optimizing Cyberattack Frequency Forecasting by Advanced Machine Learning Adaptation with Tumbling Windows."** It implements selected forecasting concepts and experiments with attention mechanisms and window-based prediction.

## 2. Objectives

* Analyze historical cyberattack frequency data.
* Identify trends in monthly cyberattack counts.
* Prepare time-series data for forecasting.
* Implement and compare machine learning and deep learning models.
* Evaluate forecasting performance using MAE, RMSE, and SMAPE.
* Investigate weighted input filters and tumbling-window forecasting.

## 3. Dataset

**Dataset:** `T-HMGDN-F-0711-0322.csv`

The original dataset contains 129 monthly records covering **July 2011 to March 2022**, with 1,114 columns.

The preprocessing pipeline aggregates country-specific attack columns into attack categories and calculates the total monthly attack count.

The resulting dataset is saved as:

`data/monthly_attack_summary.csv`

The main forecasting target is `Total_Attacks`.

## 4. Models Implemented

### Naive Baseline

Predicts the next month's attack count using the previous month's actual count.

### Random Forest Regressor

Uses the previous eight months of attack counts as input features to predict the next month's count.

### Attention-LSTM

Uses an LSTM network with an attention mechanism to assign different weights to historical time steps and generate a forecast.

### Tumbling-Window and Filter Experiments

Additional experiments investigate block-based forecasting and three weighted input filters: STCF, SMCF, and LCF.

These experiments are adaptations of selected research concepts and are not an exact reproduction of the reference paper.

## 5. Methodology

1. Load and inspect the original dataset.
2. Aggregate attack counts by category and month.
3. Visualize monthly attack trends.
4. Prepare eight-month historical input windows.
5. Split forecasting examples chronologically into training, validation, and test sets.
6. Train the forecasting models.
7. Evaluate predictions using MAE, RMSE, and SMAPE.
8. Save comparison tables and prediction outputs.

The final one-step-ahead comparison uses 20 test observations from **August 2020 to March 2022**. For each test month, the model uses the preceding eight actual monthly counts.

## 6. Final One-Step-Ahead Results

| Model          |     MAE |    RMSE |  SMAPE |
| -------------- | ------: | ------: | -----: |
| Attention-LSTM | 1146.97 | 1389.14 | 15.79% |
| Naive Baseline | 1252.60 | 1581.23 | 17.81% |
| Random Forest  | 2718.29 | 2962.86 | 45.36% |

Lower values indicate better forecasting performance for all three metrics.

In this experiment, Attention-LSTM achieved the lowest errors among the three models. The results are preliminary because the test set contains only 20 observations.

## 7. Evaluation Metrics

* **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted attack counts.
* **RMSE (Root Mean Squared Error):** Measures prediction error while penalizing larger errors more heavily.
* **SMAPE (Symmetric Mean Absolute Percentage Error):** Measures prediction error relative to the magnitudes of actual and predicted values, expressed as a percentage.

## 8. Project Structure

```text
cyberattack-forcasting/
├── data/
│   ├── T-HMGDN-F-0711-0322.csv
│   ├── monthly_attack_summary.csv
│   ├── forecasting_dataset.csv
│   ├── monthly_attack_trend.png
│   ├── attack_category_trends.png
│   ├── model_comparison.csv
│   ├── test_predictions.csv
│   ├── attention_lstm_predictions.csv
│   ├── one_step_model_comparison.csv
│   ├── one_step_predictions.csv
│   ├── final_forecast_comparison.png
│   ├── research_tumbling_results.csv
│   └── research_tumbling_predictions.csv
├── inspect_dataset.py
├── visualize_attack.py
├── prepare_data.py
├── create_windows.py
├── baseline_model.py
├── random_forest_model.py
├── compare_models.py
├── attention_lstm_model.py
├── research_tumbling_lstm.py
├── inspect_tumbling_results.py
├── one_step_attention_lstm.py
├── plot_one_step_results.py
└── README.md
```

## 9. Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* PyTorch

## 10. Installation and Setup

### Prerequisites

* Python 3.10 or later recommended
* pip
* Visual Studio Code or another Python editor

### Create a virtual environment

```powershell
python -m venv cc
.\cc\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install pandas numpy matplotlib scikit-learn torch
```

Run commands from the project root directory.

## 11. Execution

### Inspect the dataset

```powershell
python .\inspect_dataset.py
```

### Generate visualizations

```powershell
python .\visualize_attack.py
```

### Prepare forecasting data

```powershell
python .\prepare_data.py
python .\create_windows.py
```

### Run baseline and Random Forest models

```powershell
python .\baseline_model.py
python .\random_forest_model.py
python .\compare_models.py
```

### Train Attention-LSTM

```powershell
python .\attention_lstm_model.py
```

### Run tumbling-window experiments

```powershell
python .\research_tumbling_lstm.py
python .\inspect_tumbling_results.py
```

### Run the final one-step-ahead comparison

```powershell
python .\one_step_attention_lstm.py
python .\plot_one_step_results.py
```

The comparison and visualization scripts use the previously generated data files. Run the preprocessing steps first if those files do not exist.

## 12. Limitations

* The dataset ends in March 2022 and may not reflect current cyberattack patterns.
* The final test set contains only 20 monthly observations.
* Forecasting performance has not been validated across multiple random seeds or independent datasets.
* One-step-ahead forecasting and multi-month tumbling-window forecasting are different tasks and their error scores should not be compared as if they were identical.
* The implementation adapts selected ideas from the reference paper and does not reproduce every architectural and experimental detail.

## 13. Future Scope

* Evaluate the models on more recent cyberattack data.
* Validate performance across multiple time periods and random seeds.
* Investigate forecasting individual attack categories.
* Improve multi-month forecasting and assess the effects of accumulated prediction errors.
* Compare additional time-series models where appropriate.

## 14. Conclusion

This project demonstrates an end-to-end workflow for cyberattack frequency forecasting, from dataset preprocessing and time-series preparation to model training, evaluation, and visualization.

In the final one-step-ahead experiment, Attention-LSTM outperformed the Naive Baseline and Random Forest on the selected test period. Further evaluation on larger and more diverse datasets is necessary to establish the reliability and generalizability of these findings.

## 15. Research Reference

Kim and Zhan, "Optimizing Cyberattack Frequency Forecasting by Advanced Machine Learning Adaptation with Tumbling Windows," *Scientific Reports*, 2026.

https://doi.org/10.1038/s41598-026-59048-y
