from sklearn.svm import SVC
import pickle
from data_load import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

image_shape = (128, 128)

data_folder = 'data_180'

(data, labels) = load(data_folder, image_shape)
print('Load data complete')
data = data.reshape(data.shape[0], image_shape[0]*image_shape[1])

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, random_state=42)

model = SVC(kernel='poly')

model = model.fit(trainX, trainY)
print('Train complete')
pred = model.predict(testX)

score = accuracy_score(testY, pred)

print('Accuracy:', score)

#pickle.dump(model, open('rotate_180.pkl', "wb"))

