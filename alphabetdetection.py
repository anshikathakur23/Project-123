import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image

data = np.load ('image.npz')
images = data ['arr_0']
labels = pd.read_csv ('labels.csv')

images_scaled = images / 255.0
X = images_scaled.reshape (images_scaled.shape [0], -1)
y = labels.values.ravel ()

xtrain, xtest, ytrain, ytest = train_test_split (X, y, test_size = 0.2, random_state = 42)

model = LogisticRegression (max_iter = 1000)
model.fit (xtrain, ytrain)

y_pred = model.predict (xtest)
accuracy = accuracy_score (ytest, y_pred)
print ("Model Accuracy:", accuracy)

cap = cv2.VideoCapture (0)

while True:
    ret, frame = cap.read ()
    if not ret:
        break

    height, width, _ = frame.shape
    box_size = 200
    top_left = (width // 2 - box_size // 2, height // 2 - box_size // 2)
    bottom_right = (width // 2 + box_size // 2, height // 2 + box_size // 2)
    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

    roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    grayscale_image = pil_image.convert("L")
    grayscale_array = np.array(grayscale_image.resize((28, 28)))

    grayscale_array = np.clip(grayscale_array, 0, 255)

    test_sample = grayscale_array / 255.0
    test_sample = test_sample.flatten().reshape(1, -1)

    prediction = model.predict(test_sample)

    cv2.putText(frame, f"Prediction: {prediction[0]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Alphabet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()