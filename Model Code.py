import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense , GlobalAveragePooling2D , Dropout
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam
import os 

Image_Size = [224, 224]

Batch_Size = 32

Train_DIR = 'dry and wet waste.v1i.multiclass\sorted_train'

VAL_DIR = 'dry and wet waste.v1i.multiclass\sorted_train'

datagen = ImageDataGenerator(
    rescale = 1./225 , 
    validation_split = 0.2 , 
    width_shift_range = 0.2 , 
    height_shift_range = 0.2 , 
    horizontal_flip = True , 
    zoom_range = 0.2 , 
    rotation_range = 20
    )

val_datagen = ImageDataGenerator(rescale = 1./255)

train_data = datagen.flow_from_directory(
    Train_DIR , 
    target_size = Image_Size ,
    batch_size = Batch_Size ,
    class_mode = 'binary' ,
    subset = 'training'
)

val_data = datagen.flow_from_directory(
    VAL_DIR ,
    target_size = Image_Size ,
    batch_size = Batch_Size ,
    class_mode = 'binary',
    subset = 'validation'
)

base_model = ResNet50(weights = 'imagenet' ,include_top = False , input_shape = Image_Size + [3])

for layer in base_model.layers[:-10]:
    layer.trainable = False 

for layer in base_model.layers[-10:]:
    layer.trainable = True

x = base_model.output 
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128 , activation= 'relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1 ,  activation = 'sigmoid')(x)
model = Model(inputs = base_model.input , outputs = predictions)

model.compile (
    optimizer = Adam(learning_rate = 1e-5) ,
    loss = 'binary_crossentropy' , 
    metrics = ['accuracy']
)

EPOCHS = 15 

history = model.fit(
    train_data , 
    validation_data = val_data ,
    epochs = EPOCHS 
)

print('Training Done!')

model.save("Dry_Wet_classifier.h5")

