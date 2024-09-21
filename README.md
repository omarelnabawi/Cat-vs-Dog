# Cat vs Dog Classifier

This repository contains a machine learning model and a web interface for classifying images of cats and dogs.

## Files

### 1. `cat-vs-dog.ipynb`
This Jupyter Notebook contains the implementation for training the model to classify images of cats and dogs. It includes data preprocessing, model building, training, and evaluation. 

### 2. `app.py`
This is a Streamlit application that provides a user-friendly interface for uploading images and receiving predictions from the trained model. Users can upload an image of a cat or a dog, and the app will display the predicted category.
### 3.`final.h5`
This file is the final model to predict the image you can use it with `app.py` only 
## Getting Started

### Prerequisites
- Python 3.x
- Libraries:
  - TensorFlow/Keras
  - OpenCV
  - Streamlit
  - NumPy
  - Matplotlib

You can install the required libraries using:

```bash
pip install tensorflow opencv-python streamlit numpy matplotlib
```
## Running the Application
1-Train the Model:
- Open cat-vs-dog.ipynb in Jupyter Notebook or any compatible environment.
- Run all cells to train the model and save it.
2-Launch the Streamlit App:
- Run the following command in your terminal:
```bash
streamlit run app.py
```
3-Upload an Image:
- Navigate to the local URL provided by Streamlit (usually http://localhost:8501).
- Upload an image of a cat or a dog to see the prediction.
## Acknowledgements
- TensorFlow
- Streamlit
