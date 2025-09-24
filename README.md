# **PneumoniaDetectionCNN: CNN-Based Pneumonia Detector**

## **Project Overview**

**PneumoniaDetectionCNN** is a deep learning project designed to classify chest X-Ray images into two categories: **Pneumonia** and **Normal**.  
The workflow includes dataset cleaning, re-splitting for proper training, duplication detection and removal, exploratory visualization, image augmentation, CNN model training, evaluation, and deployment as an interactive Streamlit application.

## **Objective**

Build and deploy a robust **Convolutional Neural Network (CNN)** to assist in the early detection of Pneumonia from chest X-Ray images, ensuring high accuracy and clinical relevance.

## **Dataset**

- **Source**: [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)  
- **Classes**: Pneumonia, Normal  
- **Image Size**: Resized to (224 × 224), Grayscale  
- Dataset was re-split to maintain balanced training, validation, and test sets.  
- Removed duplicated images and conducted visual exploration of the dataset.  

## **Data Exploration & Preprocessing**

- Performed **dataset re-splitting** for balanced evaluation.    
- Conducted **exploratory visualization** of class distributions and example images.  
- Applied **ImageDataGenerator** for augmentation:  
  - Rescaling (1.0 / 255)  
  - Random rotation, width/height shift  
  - Zoom augmentation  
  - Horizontal flipping  

## **Modeling & Training**

- **Architecture**:  
  - Multiple **Conv2D + ReLU** layers with increasing filters (64 → 128 → 256 → 512).  
  - **MaxPooling2D** layers after convolutional blocks.  
  - Final **Flatten** followed by Dense(256 → 64 → 1) with Sigmoid output.  
- **Training Setup**:  
  - Optimizer: **Adamax (lr=0.001)**  
  - Loss: **Binary Crossentropy**  
  - Callbacks: **EarlyStopping** (patience=10, restore best weights) and **ModelCheckpoint** (save best model as `my_model.h5`).  
  - Epochs: 15

## **Performance**

- **Accuracy**: `0.961`  
- **Precision**: `0.966`  
- **Recall**: `0.981`  
- **F1 Score**: `0.905`  

The model achieved **high precision and recall**, indicating reliable Pneumonia detection while minimizing false positives.

## **Deployment**

A **Streamlit web app**  deployment was developed for real-time predictions:  

- **Streamlit App**: [PneumoniaDetectionCNN](https://medicalimgapp-s2loinj6uazbmi7dcs29qy.streamlit.app/)

The app allows users to **upload chest X-Ray images** and receive **real-time predictions** on whether the image shows Pneumonia or appears Normal.

Watch a short demo of the app here: [View on LinkedIn](https://www.linkedin.com/in/mohadeseh-mokhtari1997?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BvSCRICacSaqJF5KbfY0dpA%3D%3D)
