from sklearn.svm import SVC
import pickle
from data_load import DatasetLoader
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = DatasetLoader()

image_shape = (128, 128)

image_paths = 'data/train_after'
(data, labels) = data.load(image_paths, image_shape)

data = data.reshape(data.shape[0], image_shape[0]*image_shape[1])

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, random_state=42)

model = SVC(kernel='poly')

model = model.fit(trainX, trainY)

pred = model.predict(testX)

score = accuracy_score(testY, pred)

print(score)

#pickle.dump(model, open('rotate_180.pkl', "wb"))

