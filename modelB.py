#Tiffany Lin SID#030451783
#CECS 456 - Sec 01
#Deep Learning Project 
#Due Date: Dec. 17 2025
#Program Description: modelB.py is intended to be the ResNet50 model trained on the Kaggle dataset: "Chest Xray (Pneumonia)"
#Its purpose is to bring light to how DL models can help in clinical diagnostic efforts and compare 
#preformance to base CNN models to well-established ones.

#Import libraries
import os
import tensorflow 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib
matplotlib.use("TkAgg") #plot stuff
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

#Data paths from running formatting.py first
path = r"archive\chest_xray\chest_xray"
train_dir = os.path.join(path, "train")
test_dir  = os.path.join(path, "test")

#resnet50 data gen processing
test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

test = test_datagen.flow_from_directory(
    test_dir,
    target_size = (224, 224),
    batch_size = 32,
    class_mode = "binary",
    shuffle = False )


train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
    validation_split = 0.2       
)
#aatempting 80/20 split
train = train_datagen.flow_from_directory(
    train_dir,
    target_size = (224, 224),
    batch_size = 32,
    class_mode = "binary",
    subset = "training",
    shuffle = True
)


val_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    validation_split = 0.2
)

val = val_datagen.flow_from_directory(
    train_dir,
    target_size = (224, 224),
    batch_size = 32,
    class_mode = "binary",
    subset = "validation",
    shuffle = False )


#Simple resnet50 model; code is very different than basic CNN bc it is big well trained boy
base_model = ResNet50(weights = "imagenet", include_top = False, input_shape = (224, 224, 3))

base_model.trainable = False   #freeze base layers since resnet50 already a trained model
#Cool note: cnn has nothing to freeze since it small scratch model

#no manual layers like CNN
x = GlobalAveragePooling2D()(base_model.output) #like flatten() for bigger models
x = Dense(256, activation = "relu")(x)

x = Dropout(0.5)(x) #same drop out as cnn for consistency
output = Dense(1, activation = "sigmoid")(x) #binary classification

model = Model(inputs = base_model.input, outputs = output)

model.compile(
    optimizer = Adam(learning_rate = 1e-4),
    loss = "binary_crossentropy",
    metrics = ["accuracy"] )


callbacks = [
    EarlyStopping(monitor = "val_loss", patience = 5, restore_best_weights = True),
    ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 5)
]


#Training; experimented with epochs. Epoch 8 is best; starts overfitting more than 8e epoch
history = model.fit( train, validation_data = val, epochs = 8, callbacks = callbacks)

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
y_pred = y_pred.reshape(-1)  # flattening shape (N,1) → (N,) 2D volumized vecotr to 1D for scikitlearn stuff

#Confusion Matrix
confusionm = confusion_matrix(y_true, y_pred)

plt.figure(figsize = (6,4))
sns.heatmap(confusionm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = ['Normal', 'Pneumonia'], yticklabels = ['Normal', 'Pneumonia'])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
print("SHowing B confusion matrix...") #checker
plt.savefig("resnet50_confusion_matrix.png")
plt.show()

#Classification Matricx
print("\n\nClassification Report:")
report = classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"])

with open("resnet50_classification_report.txt", "w") as file:
    file.write(report)
    file.close()

print("Saved report to resneet50_classification_report.txt")
print(report)

#Plotting results; #math plots for accuracy/val accuracy; and loss/valueloss
plt.figure(figsize = (10,4))
plt.subplot(1, 2, 1)

plt.plot(history.history["accuracy"], label = "Train Accuracy")
plt.plot(history.history["val_accuracy"], label = "Validation Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)

print("Final Train Accuracy:", history.history['accuracy'][-1])
print("Final Validation Accuracy:", history.history['val_accuracy'][-1])

plt.plot(history.history["loss"], label = "Train")
plt.plot(history.history["val_loss"], label = "Val")
plt.legend()
plt.title("Loss")

plt.savefig("Resnet50modelB_training_plots.png")
plt.show()
