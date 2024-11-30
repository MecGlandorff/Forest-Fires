# Forest Fire Prediction with SVC
A project predicting forest fires using Support Vector Classifier (SVC) and meteorological data.  

## Results
## Model Performance Summary

| Metric           | Precision | Recall | F1-Score | Support | False negative Rate   |
|-------------------|-----------|--------|----------|---------|----------------------|
| **Class 0**       | 0.98      | 1.00   | 0.99     | 51      | 0.00                 |
| **Class 1**       | 1.00      | 0.98   | 0.99     | 53      | 0.02                 |
|                   |           |        |          |         |                      |
| **Accuracy**      |           |        | 0.99     | 104     |                      |
| **Macro Avg**     | 0.99      | 0.99   | 0.99     | 104     |                      |
| **Weighted Avg**  | 0.99      | 0.99   | 0.99     | 104     |                      |

- **Mean Squared Error (MSE):** 0.9903846153846154
- **False Negative Rate of class 1 (Fire predicted: no, fire actual: Yes):** 0.02, 1 case in all the data. So just to make model perfect we will say this is arson ;).



## Intro
Forest fires suck. This project uses machine learning to classify whether a fire will occur based on environmental factors like temperature, humidity, wind, and rainfall.

## Dataset
- **Source:** UCI Machine Learning Repository  
- **Details:** Covers forest fire data from Montesinho, Portugal, including weather indices and burn areas.  
- **Key Features:** Temperature, humidity, wind, rainfall, and fire weather indices (e.g., FFMC, DMC).  

## Installation
1. Clone this repository:  
   ```bash
   git clone https://github.com/yourusername/forest-fire-prediction.git

## Install dependencies:

**pip install -r requirements.txt**

Add the dataset:
Place forestfires.csv in the data directory.

## Usage
Run the Jupyter Notebook as if it is main.py, no need to do anything with the modules.

## License
This project is mine coding wise, however, the data is from UCI Forest Fires Dataset, so a thanks to: Creative Commons Attribution 4.0 International (CC BY 4.0) license.

## Acknowledgments
Thanks to the UCI Machine Learning Repository and contributors for the dataset and libraries like Pandas, NumPy, and Scikit-learn & Creative Commons Attribution 4.0 International (CC BY 4.0) license.
