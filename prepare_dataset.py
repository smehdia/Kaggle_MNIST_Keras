import numpy as np
import csv
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# read train.csv and save images as numpy data
def read_and_split_dataset():
    images = []
    labels = []
    with open('./data/train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in tqdm(csv_reader):
            row = list(map(int, row))
            label = row[0]
            image = np.array(row[1:]).reshape([28, 28])
            images.append(image)
            labels.append(label)

    labels = np.array(labels).reshape([-1])
    images = np.array(images).reshape([-1, 28, 28, 1])
    # split data to train-validation and test set
    X, X_test, y, y_test = train_test_split(images, labels, test_size=0.2, random_state=1)

    print 'Number of Data in Each Class:'
    for i in range(10):
        print '{}'.format(i) + str(y[y==i].shape)

    print '--------------------------------------------'

    return X, X_test, y, y_test
''
if __name__ == "__main__":
    # save data
    X, X_test, y, y_test = read_and_split_dataset()
    np.save('data/X.npy', X)
    np.save('data/y.npy', y)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)
