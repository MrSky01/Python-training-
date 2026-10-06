# Crop Recommendation System

A machine learning project that suggests the most suitable crop from soil and weather values. Made for the BSc CSIT 6th semester Python with ML training.

## Dataset
Crop Recommendation Dataset from Kaggle (`crop_recommendation.csv`): 2200 rows, 22 crops, 100 rows per crop.

Inputs: N, P, K, temperature, humidity, pH, rainfall. Output: crop name.

## Models and results
80% of the data is used for training and 20% for testing. CV is 5-fold cross-validation on the training data.

| Model | Test accuracy | CV accuracy |
|---|---|---|
| Random Forest | 99.55% | 99.32% |
| Decision Tree | 97.95% | 98.52% |
| KNN | 97.73% | 97.73% |
| Logistic Regression | 97.27% | 96.76% |

Random Forest is the best model and is used in the app.

## Files
- `train.py` trains the models, saves the best one as `model.pkl` and saves two charts in `charts/`
- `app.py` is the Streamlit web app
- `crop_analysis.ipynb` is the data analysis notebook
- `comments.txt` explains the code line by line

## How to run
```
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Limitations
The dataset is very clean, so real farm data would give lower accuracy. Only the 22 crops in the dataset can be predicted.
