# Fruit & Vegetable Classification using CNN 🍏🥦

## 📌 Project Overview
This project is a deep learning-based image classification model that identifies 36 different types of fruits and vegetables. It is built using Convolutional Neural Networks (CNN) and deployed using Streamlit for an interactive user interface.

## 🛠️ Technologies Used
- **Python** 🐍
- **TensorFlow/Keras** 🤖
- **Streamlit** 🌐
- **NumPy & Pandas** 📊
- **PIL (Pillow) for Image Processing** 🖼️
- **Google Colab** (for training)

## 📂 Dataset
The dataset consists of images of 36 different fruits and vegetables, divided into **train, validation, and test** sets. Images are resized to 180x180 pixels before training.
Here's the dataset link https://drive.google.com/file/d/1CGiAWso43GCsNo_faRq4jdDIlmwy7YI4/view

## 📑 Model Architecture
- **Convolutional Layers (CNN)**: Extracts spatial features
- **MaxPooling Layers**: Reduces dimensions while retaining important features
- **Flatten & Dense Layers**: Transforms extracted features into class probabilities
- **Softmax Activation**: Outputs probability distribution across 36 classes
- **Early Stopping**: Prevents overfitting by monitoring validation loss

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Swami-Siva-Rama-Raju/Fruits-and-Vegetables-Images-Classification.git
   cd Fruits-and-Vegetables-Images-Classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Upload an image of a fruit or vegetable and get predictions! 📸🍎🥕

## 🎯 Model Performance
The model was trained for **25 epochs** with **early stopping** and achieved high accuracy. The CNN model significantly outperforms a basic ANN model previously used for MNIST digit classification.

## 📸 Demo Video
Check out the working demo of the project here:  🎥🔥

## 📬 Connect with Me
If you liked this project, feel free to connect with me on www.linkedin.com/in/SaiKrishnaMulukutla
or check out my other projects on https://github.com/SaiKrishnaMulukutla ! 🚀



