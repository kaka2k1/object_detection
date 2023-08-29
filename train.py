import os
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

# Define the model
def create_model(input_shape=(128, 128, 3), num_classes=7, num_boxes=4):
    x_input = Input(input_shape)

    x = Conv2D(32, (3, 3), padding="same", activation='relu')(x_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    output = Dense(num_boxes*4 + num_classes, activation='sigmoid')(x)

    model = Model(x_input, output)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print(model.summary())
    
    return model


# Define the Data Generator
def data_generator(images_directory, labels_directory, batch_size=4, num_boxes=4):
    while True:
        images = []
        labels = []
        boxes = []
        for img_file in os.listdir(images_directory):
            img_path = os.path.join(images_directory, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            images.append(img)

            label_file = os.path.join(labels_directory, os.path.splitext(img_file)[0] + '.txt')
            if os.path.isfile(label_file):
                with open(label_file, 'r') as f:
                    label_str = f.readline()
                    if label_str:
                        label_parts = label_str.strip().split(' ')
                        boxes.append([float(coord) for coord in label_parts[1:5]] * num_boxes)  # assuming each box has 4 coordinates
                        labels.append(int(label_parts[0]))

            if len(images) == batch_size:
                yield (np.array(images) / 255.0, np.concatenate((np.array(boxes).reshape(-1, num_boxes*4), to_categorical(labels, num_classes=7)), axis=1))
                images = []
                labels = []
                boxes = []

        if len(images) > 0:
            yield (np.array(images) / 255.0, np.concatenate((np.array(boxes).reshape(-1, num_boxes*4), to_categorical(labels, num_classes=7)), axis=1))
# Define the paths to your directories 
data_directory = "/content/data3"
images_directory = os.path.join(data_directory, "images")
labels_directory = os.path.join(data_directory, "labels")

# Create your model
model = create_model()

# Define your generators
train_generator = data_generator(images_directory, labels_directory, batch_size=4, num_boxes=4)
val_generator = data_generator(images_directory, labels_directory, batch_size=4, num_boxes=4)

# Training parameters
batch_size = 4
steps_per_epoch = len(os.listdir(images_directory)) // batch_size

checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)

model.fit(train_generator,
          steps_per_epoch=steps_per_epoch,
          epochs=30,
          validation_data=val_generator,
          validation_steps=steps_per_epoch // 5,
          callbacks=[checkpoint])

# Save the model weights
model.save_weights('final_model_weights.h5')
