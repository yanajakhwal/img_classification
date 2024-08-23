import pickle
import numpy as np


def load_cifar10_batch(filename):
    with open(filename, 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
    data = batch['data']
    labels = batch['labels']
    
    if data.size == 0:
        raise ValueError("Loaded data is empty, check file path or data integrity.")
    
    data = data.reshape(-1, 3, 32, 32)
    data = data.transpose(0, 2, 3, 1) 

    return data, np.array(labels)


def load_cifar10(root_dir):
    train_data = []
    train_labels = []

    for i in range(1, 6):
        file_path = root_dir + '//data_batch_' + str(i)
       # print(file_path)
        data, labels = load_cifar10_batch(file_path)
        train_data.append(data)
        train_labels.append(labels)

    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)
    
    test_data, test_labels = load_cifar10_batch(root_dir + '//test_batch')

    return train_data, train_labels, test_data, test_labels


root_dir = "//Users//yanajakhwal//Desktop//Projects//Image_Classification//cifar10_data//cifar-10-batches-py"
# x_train, y_train, x_test, y_test = load_cifar10(root_dir)

# print("Training data shape:", x_train.shape)
# print("Training labels shape:", y_train.shape)
# print("Test data shape:", x_test.shape)
# print("Test labels shape:", y_test.shape)

'''
should print:
    Training data shape: (50000, 32, 32, 3)
    Training labels shape: (50000,)
    Test data shape: (10000, 32, 32, 3)
    Test labels shape: (10000,)
'''

'''
	•	Training data shape: (50000, 32, 32, 3)
	•	50000: The number of training images in the CIFAR-10 dataset.
	•	32, 32: The height and width of each image (32x32 pixels).
	•	3: The number of color channels in each image (3 channels for RGB: Red, Green, Blue).
	•	Training labels shape: (50000,)
	•	50000: The number of labels corresponding to the 50,000 training images.
	•	Test data shape: (10000, 32, 32, 3)
	•	10000: The number of test images in the CIFAR-10 dataset.
	•	32, 32: The height and width of each image (32x32 pixels).
	•	3: The number of color channels in each image (3 channels for RGB).
	•	Test labels shape: (10000,)
	•	10000: The number of labels corresponding to the 10,000 test images.
'''