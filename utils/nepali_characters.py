"""
API for Nepali Character Recognition dataset
"""
import scipy.io as sio
from sklearn.utils import shuffle
import numpy as np
# from PIL import Image


def read_data(filename='data/nepali_numbers_6.mat'):
    # filename = 'nepaliChars_dataset_v7.3.mat'
    dataset = sio.loadmat(filename)
    labels = dataset['Y']
    data = dataset['X'].astype(float)
    # print ('data',data[:1])
    return data, labels


def normalize_data(data):
    for row in range(len(data)):
        # data[row]=(data[row]-data[row].mean())/data[row].std()
        data_mean = data[row]-data[row].mean()
        data_std = data[row].std()
        data[row] = np.divide(
            data_mean, data_std,
            out=np.ones_like(data_mean)*0.001, where=data_std != 0)
        np.nan_to_num(data[row])
    # print ('normalize data',data[:1] )
    return data


def output_labels():
    output = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    return output


def data_class_length():
    data_class = [
        (0, 4788), (4788, 4872), (9660, 5124), (14784, 4676), (19460, 4844),
        (24304, 4760), (29064, 4956), (34020, 5012), (39032, 4284),
        (43316, 1372)
    ]
    return data_class


def one_hot_encoding(n=10):
    output = []
    for a in range(n):
        label = [a*0 for a in range(10)]
        label[a] = 1
        output.append(label)
    return output


def conv_labels(labels):
    labels1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    output = output_labels()
    for data in range(len(labels)):
        if labels[data] == 0:
            labels1 = np.vstack((labels1, np.array(output[0])))
        elif labels[data] == 1:
            labels1 = np.vstack((labels1, np.array(output[1])))
        elif labels[data] == 2:
            labels1 = np.vstack((labels1, np.array(output[2])))
        elif labels[data] == 3:
            labels1 = np.vstack((labels1, np.array(output[3])))
        elif labels[data] == 4:
            labels1 = np.vstack((labels1, np.array(output[4])))
        elif labels[data] == 5:
            labels1 = np.vstack((labels1, np.array(output[5])))
        elif labels[data] == 6:
            labels1 = np.vstack((labels1, np.array(output[6])))
        elif labels[data] == 7:
            labels1 = np.vstack((labels1, np.array(output[7])))
        elif labels[data] == 8:
            labels1 = np.vstack((labels1, np.array(output[8])))
        elif labels[data] == 9:
            labels1 = np.vstack((labels1, np.array(output[9])))
    labels1 = np.delete(labels1, 0, axis=0)
    return labels1


def get_label(label):
    output = output_labels()
    # print ('l',label)
    return output.index(label)


# Create input data / Label
def pre_process():
    data, labels = read_data()
    data = normalize_data(data)
    # labels=conv_labels(labels)
    # print (labels[:5])
    return data, labels


def shuffled_data(X, Y):
    # assert len(X) == len(Y)
    # p=numpy.random.permutation(len(X))
    # return
    return shuffle(X, Y, random_state=0)


def split(training_per=0.6, test_per=0.2, validation_per=0.2):
    data, labels = pre_process()
    # print (np.shape(data))
    class_lenght = data_class_length()
    training_data = np.zeros(1024)
    training_label = np.zeros(1)
    test_data = np.zeros(1024)
    test_label = np.zeros(1)
    validation_data = np.zeros(1024)
    validation_label = np.zeros(1)
    for size in class_lenght:
        test_index = size[0]+int(test_per*size[1])
        test_data = np.vstack((test_data, data[size[0]:test_index]))
        test_label = np.vstack((test_label, labels[size[0]:test_index]))
        validation_index = test_index+int(validation_per*size[1])
        validation_data = np.vstack(
            (validation_data, data[test_index:validation_index]))
        validation_label = np.vstack(
            (validation_label, labels[test_index:validation_index]))
        training_index = size[0]+size[1]
        training_data = np.vstack(
            (training_data, data[validation_index:training_index]))
        training_label = np.vstack(
            (training_label, labels[validation_index:training_index]))

    training_data = np.delete(training_data, 0, axis=0)
    validation_data = np.delete(validation_data, 0, axis=0)
    test_data = np.delete(test_data, 0, axis=0)
    training_label = np.delete(training_label, 0, axis=0)
    validation_label = np.delete(validation_label, 0, axis=0)
    test_label = np.delete(test_label, 0, axis=0)
    training_data, training_label = shuffled_data(
        training_data, training_label)
    test_data, test_label = shuffled_data(test_data, test_label)
    validation_data, validation_label = shuffled_data(
        validation_data, validation_label)

    return (
        training_data, np.squeeze(training_label), test_data,
        np.squeeze(test_label), validation_data, np.squeeze(validation_label))


"""
#def get_next_batch(n=100,data,label):
a,b,c,d,e,f=split()
#print ('Training',np.shape(a),np.shape(b))
#print ('Test',np.shape(c),np.shape(d))
#print ('Validation',np.shape(e),np.shape(f))
#print ('Training',b[:5])

for _ in range(10):
    index=np.random.random_integers(5730)
    image_data=c[index].reshape((32,32))
    img=Image.fromarray(image_data.T)
    print (d[index])
    img.show(title=d[index])
    a=input("Just testing")
"""