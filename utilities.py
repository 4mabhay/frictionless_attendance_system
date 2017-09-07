import config as cfg
import os
import numpy as np
import pickle

def load_and_prepare_data():
    """
    loads the encodings from the encoding directory,
    converts them into train data and training labels.
    Saves the label dictionary into the label_dictionary.
    :return: return training data and training label
    """
    ecoding_dir = cfg.encoding_dir
    label_dictionary = cfg.label_dictionary
    classes = {}
    train_data = []
    label_data = []
    for index, files in enumerate(os.listdir(ecoding_dir)):
        class_name = os.path.splitext(files)[0]
        class_data = np.load(os.path.join(ecoding_dir, files))
        if class_data.shape[0] != 0:
            classes[class_name] = index
            print class_name, class_data.shape
            no_of_rows = class_data.shape[0]
            label_list = [classes[class_name]] * no_of_rows
            assert no_of_rows == len(label_list)
            label_list = np.array(label_list)[:, np.newaxis]
            train_data.append(class_data)
            label_data.append(label_list)
    assert len(train_data) == len(label_data)
    train_data_npy = np.vstack(train_data)
    label_data_npy = np.vstack(label_data)
    pickle.dump(classes,open(label_dictionary, 'wb'))
    print "total no of training data %d  and label_data %d " % (train_data_npy.shape[0], label_data_npy.shape[0])
    return train_data_npy, label_data_npy , classes

def initialise_W(training_vector,class_dict):
    num_classes = len(class_dict.keys())
    dim = training_vector.shape[1]
    # Generate a random softmax weight matrix and use it to compute the loss.
    W = np.random.randn(dim , num_classes) * 0.0001
    return W

def get_train_val_test(train_size=0.6,val_size=0.2,test_size=0.2):
    X_total, y_total, class_dict = load_and_prepare_data()

    total_example = X_total.shape[0]
    total_indices = np.arange(0,total_example)

    np.random.seed(1234)
    np.random.shuffle(total_indices)

    X_total = X_total[total_indices]
    y_total = y_total[total_indices]

    num_training = int(train_size * total_example)
    num_validation = int(val_size * total_example)
    num_test = int(test_size * total_example)

    mask_train_index = np.arange(0, num_training)
    mask_val_index = np.arange(num_training,num_training+num_validation)
    mask_test_index = np.arange(total_example-num_validation,total_example)

    # Our training set will be the first num_train points from the original
    # training set.
    X_train = X_total[mask_train_index]
    y_train = y_total[mask_train_index]

    # Our validation set will be num_validation points from the original
    # training set.
    X_val = X_total[mask_val_index]
    y_val = y_total[mask_val_index]

    # We use the first num_test points of the original test set as our
    # test set.
    X_test = X_total[mask_test_index]
    y_test = y_total[mask_test_index]


    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image


    # add bias dimension and transform into columns
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])


    print 'Train data shape: ', X_train.shape
    print 'Train labels shape: ', y_train.shape
    print 'Validation data shape: ', X_val.shape
    print 'Validation labels shape: ', y_val.shape
    print 'Test data shape: ', X_test.shape
    print 'Test labels shape: ', y_test.shape

    return X_train,y_train,X_val,y_val,X_test,y_test , class_dict


if __name__=="__main__":
    from softmax import softmax_loss_vectorized
    import time

    X_train, y_train, X_val, y_val, X_test, y_test , class_dict = get_train_val_test()

    W = initialise_W(X_train,  class_dict)

    tic = time.time()
    loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_train, y_train, 0.00001)
    toc = time.time()
    # As a rough sanity check, our loss should be something close to -log(0.1).
    print 'loss: %f' % loss_vectorized
    print 'sanity check: %f' % (-np.log(0.1))
