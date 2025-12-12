# CECS456 Deep Learning Project
Dataset Chosen: Chest X-Ray Images (Pneumonia)  
Dataset Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia  
Author: Tiffany Lin

---
### Replication Steps  
1. Download formatting.py, modelA.py, modelB.py, and the chest x-ray dataset from Kaggle
2. Install python libraries: tensorflow, scikit-learn, seaborn, matplotlib
3. Run [formatting.py](formatting.py) first
4. Run [modelA.py](modelA.py) and save results
5. Run [modelB.py](modelB.py) and save results
6. Compare results of modelA that is a basic CNN model of 3D, and a transfer learning model ResNet50 that is trained with medical history.

### Purpose
Pneumonia is a lung infection that causes inflammation of the alveoli, leading to difficulty breathing, coughing, and feverish symptoms. Its severity ranges from mild to life-threatening. However, prompt and effective diagnosis of pneumonia dramatically increases patient survival. Chest x-rays are the primary tool in diagnosing this disease, but manual interpretation is prone to human error, especially in high-volume clinical environments. Deep learning has strong potential for diagnostic accuracy and reducing costs.
The objective of the project is to design, train, and evaluate deep learning models given a dataset. Training and evaluating by the Chest X-Rays (pneumonia) dataset, I implemented two different models: a custom CNN and a ResNet50 transfer learning model to compare the validity of my model’s performance. Performance was assessed using confusion and classification matrices, and accuracy and loss graphs.

---

### Report
Contains dataset overview, methodology, model architecture, results, plots, and analysis. 
[Link to Report](https://github.com/Luwuinnz/CECS456_DLProject/blob/main/456%20Report.pdf)


### Project Summary
Goal: Build and compare two deep learning models to detect pneumonia using the Kaggle Chest X-Rays Dataset.

---

### Models Implemented:  
#### Custom CNN (Binary classification)
- 32 → 64 → 128 convolution filters
- Dropout 0.5
- Binary sigmoid classifier

#### ResNet50 (Transfer Learning)
- Pretrained ImageNet weights
- Frozen convolution layers
- Custom classification head + dropout

---
### Analyzed Data Output
- Training Curves
    - [Custom CNN Training Plots](CNNmodelA_training_plots.png)
    - [ResNet50 Training Plots](Resnet50modelB_training_plots.png)
- Model Evaluation
    - [CNN Confusion Matrix](cnn_confusion_matrix.png)
    - [ResNet50 Confusion Matrix](resnet50_confusion_matrix.png)
- Classification Reports
    - [CNN Model Classification Report](cnn_classification_report.txt)
    - [ResNet50 Model Classification Report](resnet50_classification_report.txt)
---

### Code Files
Implementation of the custom CNN architecture, training loop, threshold adjustments, and evaluation logic.  
[modelA.py](modelA.py)  

Implementation of the ResNet50 transfer-learning pipeline including preprocessing, callbacks, training, and evaluation.  
[modelB.py](modelB.py)  

Handles dataset augmentation and preprocessing to improve generalization and maintain consistency across training experiments.  
[formatting.py](formatting.py)  

---

### Key Insights (from Project Report) For detailed numerical results, see 456 Report.pdf

- Dataset imbalance leads to strong recall for Pneumonia but lower recall for Normal.  
- ResNet50 generalizes better and produces smoother learning curves.  
- Threshold tuning significantly affects precision–recall balance.  
- Both models show high pneumonia sensitivity, essential in clinical settings.  
- No major signs of overfitting in either model.  


