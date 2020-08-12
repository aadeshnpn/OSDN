import keras
from keras.datasets import mnist
from keras import backend as K

from utils.evt_fitting import weibull_tailfitting
from utils.compute_openmax import recalibrate_scores

import scipy.spatial.distance as spd

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
# from nepali_characters import *

# train_x,train_y,test_x,text_y,valid_x,valid_y = split(0.9,0.05,0.05)

label = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]


def seperate_data(x, y):
    ind = y.argsort()
    sort_x = x[ind[::-1]]
    sort_y = y[ind[::-1]]

    dataset_x = []
    dataset_y = []
    mark = 0

    for a in range(len(sort_y)-1):
        if sort_y[a] != sort_y[a+1]:
            dataset_x.append(np.array(sort_x[mark:a]))
            dataset_y.append(np.array(sort_y[mark:a]))
            mark = a + 1    # here mark should be updated to the next index.
        if a == len(sort_y)-2:
            dataset_x.append(np.array(sort_x[mark:len(sort_y)]))
            dataset_y.append(np.array(sort_y[mark:len(sort_y)]))
    return dataset_x, dataset_y


def compute_feature(x, model):
    score = get_activations(model, 8, x)
    fc8 = get_activations(model, 7, x)
    return score, fc8


def compute_mean_vector(feature):
    return np.mean(feature, axis=0)


def compute_distances(mean_feature, feature, category_name):
    eucos_dist, eu_dist, cos_dist = [], [], []
    eu_dist, cos_dist, eucos_dist = [], [], []
    for feat in feature:
        eu_dist += [spd.euclidean(mean_feature, feat)]
        cos_dist += [spd.cosine(mean_feature, feat)]
        eucos_dist += [spd.euclidean(mean_feature, feat)/200. + spd.cosine(
            mean_feature, feat)]
    distances = {'eucos': eucos_dist, 'cosine': cos_dist, 'euclidean': eu_dist}
    return distances


def get_train_test():
    # batch_size = 128
    num_classes = 10
    # epochs = 50

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # print (x_train.shape,y_train.shape)

    # sep_x,sep_y = seperate_data(x_test,y_test)

    # emnist = emnist.read_data_sets('EMNIST_data',one_hot=True)
    # x_train, y_train = emnist.train.images,emnist.train.labels
    # x_test, y_test = emnist.test.images,emnist.test.labels
    # x_valid, y_valid = emnist.validation.images,emnist.validation.labels

    # print (x_train.shape,y_train.shape)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        # x_valid = x_valid.reshape(x_valid.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        # x_valid = x_valid.reshape(x_valid.shape[0],img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)   # noqa: F841

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # x_valid = x_valid.astype('float32')
    x_train /= 255
    x_test /= 255
    # x_valid /= 255
    # print('x_train shape:', x_train.shape)
    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')
    # print(x_valid.shape[0], 'valid samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test


# x_train, x_test, y_train, y_test = get_train_test()


def get_activations(model, layer, X_batch):
    # print (model.layers[6].output)
    get_activations = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[layer].output])
    activations = get_activations([X_batch, 0])[0]
    # print (activations.shape)
    return activations


def get_correct_classified(pred, y):
    pred = (pred > 0.5) * 1
    res = np.all(pred == y, axis=1)
    return res


def create_model(model, data):
    # output = model.layers[-1]

    # Combining the train and test set
    # print (x_train.shape,x_test.shape)
    # exit()
    # x_train, x_test, y_train, y_test = get_train_test()
    x_train, x_test, y_train, y_test = data
    x_all = np.concatenate((x_train, x_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    pred = model.predict(x_all)
    index = get_correct_classified(pred, y_all)
    x1_test = x_all[index]
    y1_test = y_all[index]

    y1_test1 = y1_test.argmax(1)

    sep_x, sep_y = seperate_data(x1_test, y1_test1)

    feature = {}
    feature["score"] = []
    feature["fc8"] = []
    weibull_model = {}
    feature_mean = []
    feature_distance = []

    for i in range(len(sep_y)):
        print(i, sep_x[i].shape)
        weibull_model[label[i]] = {}
        score, fc8 = compute_feature(sep_x[i], model)
        mean = compute_mean_vector(fc8)
        distance = compute_distances(mean, fc8, sep_y)
        feature_mean.append(mean)
        feature_distance.append(distance)
    np.save('mean', feature_mean)
    np.save('distance', feature_distance)


def build_weibull(mean, distance, tail):
    weibull_model = {}
    for i in range(len(mean)):
        weibull_model[label[i]] = {}
        weibull = weibull_tailfitting(mean[i], distance[i], tailsize=tail)
        weibull_model[label[i]] = weibull
    return weibull_model


def compute_openmax(model, imagearr):
    mean = np.load('mean.npy', allow_pickle=True)
    distance = np.load('distance.npy', allow_pickle=True)
    # Use loop to find the good parameters
    # alpharank_list = [1,2,3,4,5,5,6,7,8,9,10]
    # tail_list = list(range(0,21))

    alpharank_list = [10]
    # tail_list = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    tail_list = [5]
    # total = 0
    for alpha in alpharank_list:
        weibull_model = {}
        openmax = None
        softmax = None
        for tail in tail_list:
            # print ('Alpha ',alpha,' Tail ',tail)
            # print ('++++++++++++++++++++++++++++')
            weibull_model = build_weibull(mean, distance, tail)
            openmax, softmax = recalibrate_scores(
                weibull_model, label, imagearr, alpharank=alpha)

            # print ('Openmax: ',np.argmax(openmax))
            # print ('Softmax: ',np.argmax(softmax))
            # print ('opemax lenght',openmax.shape)
            # print ('openmax',np.argmax(openmax))
            # print ('openmax',openmax)
            # print ('softmax',softmax.shape)
            # print ('softmax',np.argmax(softmax))
            # if np.argmax(openmax) == np.argmax(softmax):
            # if np.argmax(openmax) == 0 and np.argmax(softmax) == 0:
            # print ('########## Parameters found ############')
            # print ('Alpha ',alpha,' Tail ',tail)
            # print ('########## Parameters found ############')
            #    total += 1
            # print ('----------------------------')
    return np.argmax(softmax), np.argmax(openmax)


def process_input(model, ind, data):
    x_train, x_test, y_train, y_test = data
    imagearr = {}
    plt.imshow(np.squeeze(x_train[ind]))
    plt.show()
    image = np.reshape(x_train[ind], (1, 28, 28, 1))
    score5, fc85 = compute_feature(image, model)
    imagearr['scores'] = score5
    imagearr['fc8'] = fc85
    # print (score5)
    return imagearr


def compute_activation(model, img):
    imagearr = {}
    # img = np.squeeze(img)
    img = np.array(
        Image.fromarray(
            (np.squeeze(img)).astype(np.uint8)).resize((28, 28)))
    # img = scipy.misc.imresize(np.squeeze(img),(28,28))
    # img = img[:,0:28*28]
    img = np.reshape(img, (1, 28, 28, 1))
    score5, fc85 = compute_feature(img, model)
    imagearr['scores'] = score5
    imagearr['fc8'] = fc85
    return imagearr


def image_show(img, label):
    # print(img.shape)
    # img = scipy.misc.imresize(np.squeeze(img), (28, 28))
    # img = np.array(
    #     Image.fromarray(
    #         (np.squeeze(img)).astype(np.uint8)).resize((28, 28)))
    # print(img.shape)
    # img = img[:, 0:28*28]
    plt.imshow(np.squeeze(img), cmap='gray')
    # print ('Character Label: ',np.argmax(label))
    plt.show()


# def openmax_unknown_class(model):
#     f = h5py.File('HWDB1.1subset.hdf5','r')
#     # total = 0
#     i = np.random.randint(0, len(f['tst/y']))
#     print('label', np.argmax(f['tst/y'][i]))
#     print(f['tst/x'][i].shape)
#     # exit()
#     imagearr = process_other_input(model, f['tst/x'][i])
#     compute_openmax(model, imagearr)
    #     if compute_openmax(model, imagearr)    >= 4:
    #        total += 1
    # print ('correctly classified',total,'total set',len(y2))


def openmax_known_class(model, y, data):
    x_train, x_test, y_train, y_test = data
    # total = 0
    for i in range(15):
        # print ('label', y[i])
        j = np.random.randint(0, len(y_train[i]))
        imagearr = process_input(model, j)
        print(compute_openmax(model, imagearr))
        #    total += 1
    # print ('correct classified',total,'total set',len(y))


"""
def main():
    #model = load_model("MNIST_CNN_tanh.h5")
    model = load_model("MNIST_CNN.h5")
    #create_model(model)
    #openmax_known_class(model,y_test)
    openmax_unknown_class(model)

if __name__ == '__main__':
    main()
"""