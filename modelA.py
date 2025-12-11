#Tiffany Lin SID#030451783
#CECS 456 - Sec 01
#Deep Learning Project 
#Due Date: Dec. 17 2025
#Program Description: modelA.py is intended to be the baseline CNN model trained on the Kaggle dataset: "Chest Xray (Pneumonia)"
#Its purpose is to bring light to how DL models can help in clinical diagnostic efforts.

#Import libraries
import os
import tensorflow 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import matplotlib
matplotlib.use("TkAgg") #plot stuff
import matplotlib.pyplot as plt



#Data paths from running formatting.py first
path = r"archive\chest_xray\chest_xray"
train_dir = os.path.join(path, "train")
test_dir  = os.path.join(path, "test")

#Similar to formatting.py; gets the data from the directory stated in formatting.py

#Test and val is untouched by augmentation
test_datagen = ImageDataGenerator(rescale = 1/255)

test = test_datagen.flow_from_directory(
    test_dir,
    target_size = (224, 224),
    batch_size = 32,
    class_mode = "binary",
    shuffle = False)

train_datagen = ImageDataGenerator(
    rescale = 1/255,
    rotation_range = 10,
    zoom_range = 0.1,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    validation_split = 0.2)

train = train_datagen.flow_from_directory(
    train_dir,
    target_size = (224, 224),
    batch_size = 32,
    class_mode = "binary",
    subset = "training",
    shuffle = True)

val_test_datagen = ImageDataGenerator(
    rescale = 1/255,
    validation_split = 0.2
)

val = val_test_datagen.flow_from_directory(
    train_dir,
    target_size = (224, 224),
    batch_size = 32,
    class_mode = "binary",
    subset = "validation",
    shuffle = False)


#Baseline CNN Model

#3 dimensions
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    #Modify to 0.4 or 0.5
    Dropout(0.4),
    Dense(1, activation="sigmoid") ])

#Adam optimizaer with 1e-4
model.compile(
    optimizer = Adam( learning_rate = 1e-4),
    loss = "binary_crossentropy",
    metrics = ["accuracy"] )

model.summary()

#Training; experimented with epochs. Epoch 10 is best
history = model.fit( train, validation_data=val, epochs = 10 )
print(history.history.keys())

#Test Set Evaluation
test.reset()
test_loss, test_acc = model.evaluate(test)
print("Test Accuracy:", test_acc)

#Result Reports-----------------------

#Confusion Matrix

#Normal =>0; Pneumonia => 1
#Positive output is Normal; 0.5<= binary
#Pneumonia; 0.5> binary

#True labels; similar to HW 1 and 2
y_true = test.classes

#Predicted labels
test.reset()
y_pred = (model.predict(test) > 0.5).astype("int32")
y_pred = y_pred.reshape(-1)   # flattening shape (N,1) → (N,) 2D volumized vecotr to 1D for scikitlearn stuff

# Confusion Matrix
confusionm = confusion_matrix(y_true, y_pred)

plt.figure(figsize = (6,4))
sns.heatmap(confusionm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = ['Normal', 'Pneumonia'], yticklabels = ['Normal', 'Pneumonia'])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
print("SHowing A confusion matrix...") #checker
plt.savefig("cnn_confusion_matrix.png")
plt.show()

# Classification Matrix
print("\n\nClassification Matrix:")
report = classification_report(y_true, y_pred, target_names = ["Normal", "Pneumonia"])

with open("cnn_classification_report.txt", "w") as file:
    file.write(report)
    file.close()

print("Saved report to cnn_classification_report.txt")
print(report)
print("Final Train Accuracy:", history.history['accuracy'][-1])
print("Final Validation Accuracy:", history.history['val_accuracy'][-1])


#math plots for accuracy/val accuracy; and loss/valueloss
plt.figure(figsize = (10,4))
plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'], label = "Train Accuracy")
plt.plot(history.history['val_accuracy'], label = "Val Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)

plt.plot(history.history['loss'], label = "Train Loss")
plt.plot(history.history['val_loss'], label = "Val Loss")
plt.legend()
plt.title("Loss")

plt.savefig("CNNmodelA_training_plots.png")
plt.show()

