#Tiffany Lin SID#030451783
#CECS 456 - Sec 01
#Deep Learning Project 
#Due Date: Dec. 17 2025
#Program Description: formatting.py is intended to augment the Kaggle dataset: "Chest Xray (Pneumonia)"
#to ensure consistency between both models A and Model B for analysis purposes.


from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Data augmenting code
train_datagen = ImageDataGenerator(
    rescale = 1/255,
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1
)

test_datagen = ImageDataGenerator(rescale = 1/255)

#Datasets; defining the train, val, and test datasets for consistency and easy usage

#Unified 224x224 images and 32 batch size
#Using a binary cassification model since the dataset is healthy or pneumonia chest images.
test = test_datagen.flow_from_directory(
    r"archive/chest_xray/chest_xray/test",
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'binary',
    shuffle = False)
#shuffle is false to preserve testing datasamples

#ONLY TRAIN DATA IS AUGMENTED; TEST/VAL UNTOUCHED
train = train_datagen.flow_from_directory(
    r"archive/chest_xray/chest_xray/train",
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'binary')

val = test_datagen.flow_from_directory(
    r"archive/chest_xray/chest_xray/val",
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'binary',
    shuffle = False)

print(test.class_indices)
print(train.class_indices)
print(val.class_indices)