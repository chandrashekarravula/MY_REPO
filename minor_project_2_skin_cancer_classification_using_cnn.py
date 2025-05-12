# -*- coding: utf-8 -*-
!pip install tensorflow
import tensorflow as tf
from tensorflow.keras import layers,models,optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

from google.colab import drive
drive.mount('/content/drive')

train_dir="/content/drive/MyDrive/Skin cancer ISIC The International Skin Imaging Collaboration/Train"
test_dir="/content/drive/MyDrive/Skin cancer ISIC The International Skin Imaging Collaboration/Test"

image_height,image_width=150,150
batch_size=32

train_datagen= ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen=ImageDataGenerator(rescale=1.0/255.0)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_height,image_width),
    class_mode='categorical',
    batch_size=32
    )

test_generator=test_datagen.flow_from_directory(
    test_dir,
    target_size=(image_height,image_width),
    class_mode='categorical',
    batch_size=32
)

class_names = list(train_generator.class_indices.keys())
print(f"Classes: {class_names}")

print(train_generator.class_indices)
print(test_generator.class_indices)

print(train_generator.samples)
print(test_generator.samples)

import numpy as np
import matplotlib.pyplot as plt

# Assuming train_generator is already defined as in the previous code

num_images_per_class = 5

for class_name in class_names:
  images = []
  labels = []
  for x,y in train_generator:
    for i in range(len(y)):
      if np.argmax(y[i]) == train_generator.class_indices[class_name] and len(images) < num_images_per_class:
        images.append(x[i])
        labels.append(class_name)
    if len(images) == num_images_per_class:
      break

  plt.figure(figsize=(20, 5))
  plt.suptitle(class_name, fontsize=20)
  for i in range(len(images)):
      plt.subplot(1, num_images_per_class, i + 1)
      plt.imshow(images[i])
      plt.axis('off')
  plt.show()

from sklearn.model_selection import StratifiedKFold

X = np.random.rand(100, 150, 150, 3)
y = np.random.randint(0, 7, 100)


n_splits = 5  # Number of folds
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}/{n_splits}")

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Create data generators for this fold
    train_datagen_fold = ImageDataGenerator(
        rescale=1.0/255.0,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen_fold = ImageDataGenerator(rescale=1.0/255.0)

    model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(image_height,image_width,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
    ])
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


    # Train the model on this fold
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=test_generator
    )

import matplotlib.pyplot as plt
import numpy as np
# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Get the confus# prompt: write evaluation metrics for above codes.

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Get predictions for the test data
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get true labels
y_true = test_generator.classes

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Precision, Recall, F1-score, and support
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Plot training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

from tensorflow.keras.utils import load_img,img_to_array
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def predict_image(image_path):
    img=load_img(image_path,target_size=(image_height,image_width))
    img_array=img_to_array(img)/255.0
    img_array=np.expand_dims(img_array,axis=0)
    prediction=model.predict(img_array)
    class_idx=(np.argmax(prediction))
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show()
    print(f"\nPredicted Class: {class_names[class_idx]}")

predict_image("/content/drive/MyDrive/test_image_skin_cancer.webp")

# prompt: save model as .keras zip file

model.save('my_model.keras')
!zip -r my_model.zip my_model.keras
from google.colab import files
files.download('my_model.zip')

from tensorflow.keras.models import save_model

# Assuming 'model' is your trained CNN
save_model(model, 'skin_cancer_model.h5')  # Saves in HDF5 format
# OR (for newer TF versions)
model.save('skin_cancer_model.keras')

pip install streamlit pillow numpy tensorflow

# prompt: generate code to save above model and deploy using streamlit with UI as image uploading field and output field which is returned by model.

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the trained model
model = tf.keras.models.load_model('my_model.keras')  # Assuming 'my_model.keras' is in the same directory

# Define image dimensions (must match the model's input shape)
image_height, image_width = 150, 150
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'] # Replace with your actual class names

def predict_image(image):
    img = image.resize((image_height, image_width))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    predicted_class = class_names[class_idx]
    return predicted_class, prediction


st.title("Skin Cancer Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        predicted_class, prediction = predict_image(image)

        st.write(f"Predicted Class: {predicted_class}")


        # Display the prediction probabilities (optional)
        st.write("Prediction Probabilities:")
        for i, prob in enumerate(prediction[0]):
          st.write(f"{class_names[i]}: {prob:.4f}")

        # Plot the image (optional)
        # st.pyplot(plt)
        plt.imshow(image)
        plt.axis("off")
        st.pyplot(plt)

# prompt: where to view wesite

#The website is viewable by running the streamlit app.
#First, you need to save the code as a Python file (e.g., app.py).
#Then, in the terminal or command prompt, navigate to the directory where you saved the file and run the command:
#streamlit run app.py


#This will open the Streamlit app in your web browser. You can then interact with the app by uploading an image.
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Set page title and layout
st.set_page_config(page_title="Skin Cancer Classification", layout="wide")

# Load the saved model
@st.cache_resource  # Cache model for faster reloads
def load_model():
    return tf.keras.models.load_model('skin_cancer_model.h5')  # or .keras

model = load_model()

# Define class labels (modify based on your dataset)
CLASS_NAMES = [
    "Melanoma",
    "Melanocytic Nevus",
    "Basal Cell Carcinoma",
    "Actinic Keratosis",
    "Benign Keratosis",
    "Dermatofibroma",
    "Vascular Lesion"
]

# Preprocess image for model input
def preprocess_image(image):
    img = image.resize((224, 224))  # Match model's expected input
    img = np.array(img)
    img = img / 255.0  # Normalize (if your model expects [0,1])
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit UI
st.title("Skin Cancer Classification using CNN 🏥")
st.write("Upload an image of a skin lesion for classification.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess and predict
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Show prediction
    st.subheader("Prediction Result")
    st.success(f"**Class:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # Optional: Show prediction probabilities
    st.write("**Detailed Probabilities:**")
    for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, predictions[0])):
        st.write(f"{class_name}: {prob * 100:.2f}%")

#streamlit run app.py

def upload():
  uploaded = files.upload()
  for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(uploaded[fn])))
    predict_image(fn)
    print("\n\n")
    upload()

upload()