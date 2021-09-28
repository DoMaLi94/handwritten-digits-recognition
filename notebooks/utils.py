import numpy as np
import h5py
import cv2

import matplotlib.pyplot as plt


def plot_random_samples(x, y, num_samples=3, title=""):
    
    # Reshape the images for plotting
    side_length = int(np.sqrt(x.shape[1]))
    x = x.reshape(x.shape[0], side_length, side_length)
    
    # Create subplots
    fig, ax = plt.subplots(num_samples, 10, sharex=True, sharey=True,
                           figsize=(15, num_samples), constrained_layout=True)
    
    # Set title
    fig.suptitle(title, fontsize=16)
    
    # Sample from images and plot them
    for label in range(10):
        class_idxs = np.where(y == label)
        for i, idx in enumerate(np.random.randint(0, class_idxs[0].shape[0], num_samples)):
            ax[i, label].set_axis_off()
            if i == 0:
                ax[i, label].set_title('label: %i' % label)
            ax[i, label].imshow(x[class_idxs[0][idx]],
                                cmap=plt.cm.gray, interpolation='nearest')
    
    # Plot images
    plt.show()
    
    # Print total samples
    print(f"Total number of samples: {x.shape[0]}")
            


def plot_info(x, y, name):
    
    print(f"..... Info: {name} Dataset .....")
    print("---------------------------------")
    print(f"    feature shape: {x.shape}")
    print(f"    min value: {np.min(x)}, max value: {np.max(x)}")
    print("")
    print(f"    target shape: {y.shape}")
    print(f"   Classes: {np.unique(y)}")
    print("-----------------------------------")
    print("")


def store_hdf5(hdf_file, x_train, y_train, x_test, y_test):
    print(f"Creating HDF5 dataset and sotre it to: {hdf_file} ...")
    
    # Open file in write mode
    f = h5py.File(hdf_file, 'w')
    
    # Create data structure and save it
    group = f.create_group('training')
    group.create_dataset(name='images', data=x_train, compression='gzip')
    group.create_dataset(name='labels', data=y_train, compression='gzip')

    group = f.create_group('testing')
    group.create_dataset(name='images', data=x_test, compression='gzip')
    group.create_dataset(name='labels', data=y_test, compression='gzip')
    
    # Important: Close open file
    f.close()
    print("Done.")
    
    
def load_data(datasets: dict) -> dict:
    
    """
        Return multidimensional dictanary with:
        1d: key: dataset name
            value: dataset
            
        2d: key: training or testing
            value: training or testing datasets
            
        3d: key: x or y
            value: x: 2d np.array of 1d:samples, 2d:pixels
                   y: 1d np.array of classes
    """
    
    # Create dict for datasets
    data = {}
    
    # Load and store dataset in dict
    for dataset in datasets.keys():
        print(f"Loading {dataset}...")
        
        # Open hdf5 file, read data and close file
        f = h5py.File(datasets[dataset], mode='r')
        x_tr = np.array(f["training"]["images"])
        y_tr = np.array(f["training"]["labels"])
        x_te = np.array(f["testing"]["images"])
        y_te = np.array(f["testing"]["labels"])
        f.close()
        
        # Append dataset to data dict
        data.update({dataset: {"train": {"x": x_tr, "y": y_tr}, \
                               "test": {"x": x_te, "y": y_te}}})
        
    print("Done.")
    # return a dict of datasets
    return data


def select_dataset(data, dataset):
    x_tr = data[dataset]["train"]["x"]
    x_te = data[dataset]["test"]["x"]
    y_tr = data[dataset]["train"]["y"]
    y_te = data[dataset]["test"]["y"]
    return x_tr, x_te, y_tr, y_te


def resize_images(img_arr:np.array, width=16, height=16, interpolation = cv2.INTER_LINEAR):
    # Resize images to smaler size
    
    # Create numpy array for resized images
    images = np.zeros((img_arr.shape[0], width * height))
    
    # Reshape img_arr to 3d
    img_arr = img_arr.reshape(img_arr.shape[0], \
                              np.sqrt(img_arr.shape[1]).astype(np.int), \
                              np.sqrt(img_arr.shape[1]).astype(np.int))
    
    for i, img in enumerate(img_arr):
        # Resize images using OpenCV
        img = cv2.resize(img, (width, height), interpolation=interpolation)
        # Reshape to 2d nampy array
        images[i] = img.reshape(1, width * height)
    
    # Return images (samples x images as 2d array)
    return images