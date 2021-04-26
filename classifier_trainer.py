from sklearn.svm import SVC
import numpy as np
from glob import glob
from os.path import basename
from micromlgen import port

def load_features(folder):
    dataset = None
    classmap = {}
    for class_idx, filename in enumerate(glob('%s/*.csv' % folder)):
        class_name = basename(filename)[:-4]
        classmap[class_idx] = class_name
        samples = np.loadtxt(filename, dtype=float, delimiter=',')
        labels = np.ones((len(samples), 1)) * class_idx
        samples = np.hstack((samples, labels))
        dataset = samples if dataset is None else np.vstack((dataset, samples))
    return dataset, classmap

features, classmap = load_features(r'C:\Users\chait\Desktop\My Folder\Electronics Projects\Object Detection Using ESP32 CAM\samples_csv')
X, y = features[:, :-1], features[:, -1]
classifier = SVC(kernel='rbf', gamma=0.001).fit(X, y)
c_code = port(classifier)
print(c_code)