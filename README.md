# Crop_Prediction_model
Crop Prediction Model using Random forrest tree

# Crop Recommendation System

This project builds a machine learning model to recommend the best crop to plant based on soil and weather conditions such as Nitrogen, Phosphorus, Potassium content, temperature, humidity, pH, and rainfall.

---

## Features (Inputs)

- Nitrogen content in soil (mg/kg)
- Phosphorus content in soil (mg/kg)
- Potassium content in soil (mg/kg)
- Average temperature (°C)
- Average relative humidity (%)
- Soil pH value
- Rainfall (mm)

## Output

- Recommended crop to plant

---

## Model Details

- Algorithm: Random Forest Classifier
- High accuracy (~99.3%) with detailed classification report
- Uses label encoding to map crop names to numerical labels

---

## Requirements

Install the required Python packages using:

```bash

pip install numpy pandas scikit-learn matplotlib seaborn joblib
Required libraries:

numpy

pandas

scikit-learn

matplotlib

seaborn

joblib

How to Use
Clone this repository.

Load the pre-trained model and label encoder .pkl files using joblib.

Prepare input features as a NumPy array.

Use the model to predict the crop and decode it using the label encoder.

Example:

python
Copy
Edit
import joblib
import numpy as np

model = joblib.load('crop_recommendation_model.pkl')
le = joblib.load('label_encoder.pkl')

sample = np.array([[90, 40, 40, 25, 80, 6.5, 200]])
prediction = model.predict(sample)
crop = le.inverse_transform(prediction)

print("Recommended Crop:", crop[0])
```
Author
Aakash Pal
Email: palaakaah148@gmail.com


