from keras.models import load_model
import csv
import numpy as np
from tqdm import tqdm
from glob import glob
from scipy.stats import mode

if __name__ == "__main__":

    # read test images
    images = []
    with open('./data/test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in tqdm(csv_reader):
            row = list(map(int, row))
            image = np.array(row).reshape([28, 28])
            images.append(image)

    images = np.array(images).reshape([-1, 28, 28, 1]).astype('float32')
    # scale images
    images = (images - 127.5) / 127.5
    # get prediction for each model (for ensemble models)
    labels = []
    for model_name in glob('models/*'):
        print 'Model Name: ' + model_name
        model = load_model(model_name)
        labels.append(model.predict(images))
    # convert one hot predictions to integets
    labels = np.array(labels)
    labels = np.argmax(labels, axis=2)
    # write predictions in submission.csv
    f = open('submission.csv', 'w')
    f.write('ImageID,label\n')
    for i in tqdm(range(labels.shape[1])):
        l = labels[:, i]
        # voting between different models predictions
        f.write('{},{}\n'.format(i+1, mode(l)[0][0]))

    f.close()



