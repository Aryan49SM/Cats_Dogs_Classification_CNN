## Overview
A Convolutional Neural Network (CNN) implementation for binary classification of cat and dog images. This project uses deep learning techniques to classify images between cats and dogs.

## Features
* Binary image classification (Cats vs Dogs)
* Convolutional Neural Network architecture
* Image preprocessing
* Model training and evaluation
* Prediction Functionality

## Installation
Follow these steps to set up the project on your local machine:

1. **Clone the repository:**
   
   ```bash
   git clone https://github.com/Aryan49SM/Cats_Dogs_Classification_CNN.git
   cd Cats_Dogs_Classification_CNN
   
3. **Create a virtual environment:**
   ```bash
   python -m venv env
   
5. **Activate the virtual environment:**
   
   1. On Windows:
      
      ```bash
      .\env\Scripts\activate
      
    3. On macOS/Linux:
       ```bash
       source env/bin/activate
       
7. **Install the required packages:**
   
   ```bash
   pip install numpy pandas tensorflow keras matplotlib opencv-python Pillow scikit-learn

## Usage

1. Run the Jupyter Notebook
   
    ```bash
    jupyter notebook Cats_Dogs_Classification_CNN.ipynb
    ```

2. Run streamlit webapp (using .h5 model)
   
   ```bash
   python -m streamlit run app.py
   ```

## Model Architecture
* **Input Layer:** Accepts image data.
* **Convolutional Layers:** Extract features using multiple filters.
* **BatchNormalization:** Normalize the output and accelerate training.
* **MaxPooling Layers:** Reduce spatial dimensions and control overfitting.
* **Dense Layers:** Uses a ReLU & sigmoid activation function for binary classification.
* **Dropout Layers:** Prevent overfitting by randomly disabling neurons during training.
    
