# import os
# import cv2
# import numpy as np
# from keras.models import Model, load_model
# from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
# from keras.utils import to_categorical
# from matplotlib import pyplot as plt

# def non_max_suppression(boxes, probs, overlapThresh=0.3):
#     if len(boxes) == 0:
#         return []

#     if boxes.dtype.kind == "i":
#         boxes = boxes.astype("float")

#     pick = []

#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]

#     area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
#     if probs is None: 
#         idxs = np.arange(0, len(boxes))
#     else:
#         idxs = np.argsort(probs)

#     while len(idxs) > 0:
#         last = len(idxs) - 1
#         i = idxs[last]
#         pick.append(i)

#         xx1 = np.maximum(x1[i], x1[idxs[:last]])
#         yy1 = np.maximum(y1[i], y1[idxs[:last]])
#         xx2 = np.minimum(x2[i], x2[idxs[:last]])
#         yy2 = np.minimum(y2[i], y2[idxs[:last]])

#         w = np.maximum(0, xx2 - xx1 + 1)
#         h = np.maximum(0, yy2 - yy1 + 1)

#         overlap = (w * h) / area[idxs[:last]]

#         idxs = np.delete(idxs, np.concatenate(([last],
#             np.where(overlap > overlapThresh)[0])))

#     return boxes[pick].astype("int")


# def detect_objects(image_path):
#     if not os.path.exists(image_path):
#         print("Image path not found.")
#         return

#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (128, 128))
#     input_img = np.expand_dims(img / 255.0, axis=0)

#     # Get predictions from the model
#     predictions = model.predict(input_img)

#     num_boxes = 4  # The same as in the training code
#     num_classes = 7  # The same as in the training code

#     # Separate the predictions into boxes and class probabilities
#     boxes = predictions[:, :num_boxes*4]
#     class_probs = predictions[:, num_boxes*4:]

#     # Reshape the boxes and class_probs
#     boxes = np.reshape(boxes, (-1, num_boxes, 4))
#     class_probs = np.reshape(class_probs, (-1, num_classes))

#     # Perform Non-Maximum Suppression to get the final predictions
#     selected_boxes, _, _ = non_max_suppression(boxes, class_probs, overlapThresh=0.5)

#     # Check if there are any selected boxes
#     if len(selected_boxes) == 0:
#         print("No objects detected.")
#         return

#     # Draw the bounding boxes and class labels on the image
#     for x, y, w, h in selected_boxes:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # Display the result
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.show()

# # Load the pre-trained model
# model = load_model('best_model.h5')

# # Perform object detection on a single image
# test_image_path = "hinhanh.jpg"
# detect_objects(test_image_path)

import os
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

def predict_image(model_path, image_path):
    # Load the model
    model = load_model(model_path)

    # Load the image and resize to the expected input shape (128,128)
    img = image.load_img(image_path, target_size=(128, 128))

    # Convert image to array and expand dimension
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    # Make a prediction
    predictions = model.predict(img_batch)

    # Remove bounding box predictions
    predictions = predictions[0][16:]

    # Get the class label of the highest prediction score
    class_labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train"]
    predicted_label = class_labels[np.argmax(predictions)]

    # Return predictions and predicted label
    return predictions, predicted_label


# Path to the model
model_path = 'best_model2.h5'

# Path to the image
image_path = 'hinhanh4.jpg'

# Predict and print predictions
if os.path.exists(image_path):
    predictions, label = predict_image(model_path, image_path)
    print(f"Predictions: {predictions}\nPredicted label: {label}")
else:
    print("Image path not found.")

