import glob
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout
from keras import optimizers
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import TimeDistributed, GlobalAveragePooling2D

labels_file = '/Users/euntaeklee/tensorflow-env/final_project/full.csv'
sequence_length = 30  # 2.5 seconds
num_classes = 2
num_epochs = 200
batch_size = 10

# Load labels
header_list = ["file", "label"]
labels = pd.read_csv(labels_file, names=header_list)

# Extract frames from video
def frame_extract(path):
    vidObj = cv2.VideoCapture(path)
    frames = []
    success = True
    while success:
        success, image = vidObj.read()
        if success:
            frames.append(image)
    return frames

# Load and preprocess video frames
def load_video_frames(video_path, sequence_length):
    frames = frame_extract(video_path)
    num_frames = len(frames)
    if num_frames < sequence_length:
        # Pad frames if the video is shorter than the desired sequence length
        frames += [frames[-1]] * (sequence_length - num_frames)
    elif num_frames > sequence_length:
        # Randomly select a sequence of frames from the video
        start_index = random.randint(0, num_frames - sequence_length)
        frames = frames[start_index: start_index + sequence_length]

    # Resize frames to (112, 112)
    resized_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, (224, 224)).astype(np.float32)
        resized_frames.append(resized_frame)

    # Convert frames to numpy array and preprocess
    frames_array = np.array(resized_frames, dtype=np.float32)
    frames_array = preprocess_input(frames_array)

    # Get label
    video_name = video_path.split('/')[-1]
    label = labels.loc[labels["file"] == video_name, "label"].values
    if len(label) == 0:
        return None, None
    label = label[0]
    if label == 'FAKE':
        label = 0
    elif label == 'REAL':
        label = 1

    return frames_array, label

# Load the data
video_files = glob.glob('/Users/euntaeklee/Downloads/FAKE_Face_only_data/*.mp4')
video_files += glob.glob('/Users/euntaeklee/Downloads/REAL_Face_only_data/*mp4')
random.shuffle(video_files)
random.shuffle(video_files)
print(len(video_files))
data = []
for video_path in video_files:
    frames, label = load_video_frames(video_path, sequence_length)
    if frames is not None and label is not None:
        data.append((frames, label))


if len(data) == 0:
    print("No valid data found.")
    exit()

X = np.array([item[0] for item in data])
y = to_categorical(np.array([item[1] for item in data]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
inputs = Input(shape=(sequence_length, 224, 224, 3))
resnet = ResNet50(include_top=False, weights='imagenet')
for layer in resnet.layers:
    layer.trainable = False
resnet = TimeDistributed(resnet)(inputs)
dropout = Dropout(0.5)(resnet)
resnet = TimeDistributed(GlobalAveragePooling2D())(dropout)
lstm = LSTM(512)(resnet)
dense = Dense(512, activation='relu')(lstm)
output = Dense(num_classes, activation='softmax')(dense)
model = Model(inputs=inputs, outputs=output)

# Train the model
model.compile(optimizer=optimizers.Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, batch_size=batch_size)

# Test the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss: {:.4f}'.format(loss))
print('Test Accuracy: {:.2f}%'.format(accuracy * 100))

# Predict the test set
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Plot the confusion matrix
labels = ['FAKE', 'REAL']
cm = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_model_train.png')


