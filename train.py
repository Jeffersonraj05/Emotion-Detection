import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Define paths
train_dir = r'C:\Users\Jefferson\Desktop\PROJECTS\Emotion_Detection\FER2013\train'

# Data generator for training
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical'
)

# Function to create synthetic data
def create_synthetic_data(train_dir, emotion1, emotion2, num_samples):
    # Load images for emotion1 and emotion2
    emotion1_dir = os.path.join(train_dir, emotion1)
    emotion2_dir = os.path.join(train_dir, emotion2)
    
    emotion1_images = []
    emotion2_images = []
    
    for file in os.listdir(emotion1_dir):
        img = load_img(os.path.join(emotion1_dir, file), target_size=(48, 48), color_mode='grayscale')
        img_array = img_to_array(img)
        emotion1_images.append(img_array)
        
    for file in os.listdir(emotion2_dir):
        img = load_img(os.path.join(emotion2_dir, file), target_size=(48, 48), color_mode='grayscale')
        img_array = img_to_array(img)
        emotion2_images.append(img_array)
        
    # Create synthetic images by averaging
    synthetic_images = []
    for _ in range(num_samples):
        idx1 = np.random.randint(0, len(emotion1_images))
        idx2 = np.random.randint(0, len(emotion2_images))
        
        synthetic_image = (emotion1_images[idx1] + emotion2_images[idx2]) / 2
        synthetic_images.append(synthetic_image)
        
    return np.array(synthetic_images)

# Create synthetic data for anxiety (fear + sad) and embarrassed (sad + surprise)
anxiety_images = create_synthetic_data(train_dir, 'fear', 'sad', 2000)  # Increased samples
embarrassed_images = create_synthetic_data(train_dir, 'sad', 'surprise', 2000)  # Increased samples

# Normalize synthetic images
anxiety_images = anxiety_images / 255.0
embarrassed_images = embarrassed_images / 255.0

# Create labels for synthetic data
anxiety_labels = np.zeros((2000, 9))
anxiety_labels[:, 7] = 1  # Anxiety label

embarrassed_labels = np.zeros((2000, 9))
embarrassed_labels[:, 8] = 1  # Embarrassed label

# Combine all training images with synthetic data
X_train_combined = []
y_train_combined = []

for batch in train_generator:
    # Adjust labels to match the new structure (9 classes)
    batch_labels = np.zeros((batch[1].shape[0], 9))
    batch_labels[:, :7] = batch[1]
    
    X_train_combined.append(batch[0])
    y_train_combined.append(batch_labels)
    
    # Stop once we've processed all batches
    if len(X_train_combined) * 32 >= train_generator.samples:
        break

# Convert lists to numpy arrays
X_train_combined = np.concatenate(X_train_combined)
y_train_combined = np.concatenate(y_train_combined)

# Combine with synthetic data
X_train_combined = np.concatenate((X_train_combined, anxiety_images, embarrassed_images))
y_train_combined = np.concatenate((y_train_combined, anxiety_labels, embarrassed_labels))

# Define CNN model with more layers
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))  # Output layer for 9 classes

# Compile model using learning_rate instead of lr
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train model for more epochs if needed
history = model.fit(X_train_combined, y_train_combined, batch_size=32, epochs=20, verbose=1)

# Plot training accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.legend()
plt.show()

# Evaluate model
y_pred = model.predict(X_train_combined)
y_pred_class = np.argmax(y_pred, axis=1)
y_train_class = np.argmax(y_train_combined, axis=1)

# Evaluate accuracy
accuracy = accuracy_score(y_train_class, y_pred_class)
print(f"Training Accuracy: {accuracy:.2f}")

# Print classification report
print(classification_report(y_train_class, y_pred_class))

# Print confusion matrix
print(confusion_matrix(y_train_class, y_pred_class))

# Save the model with a new name to avoid overwriting previous models
model.save('emotion_detection_model_v2.h5')
