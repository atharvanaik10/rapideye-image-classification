from PIL import Image
import numpy as np
import dbf
import pandas as pd

# TODO function to load images
def load_image_as_numpy(filepath):
    """Takes in a filepath and returns a numpy array representation of the image

    Args:
        filepath (string): filepath of image

    Returns:
        np.array: numpy array of the image
    """
    with Image.open(filepath) as img:
        numpy_image = np.array(img)
    return numpy_image

# TODO define a function to get pixel wise y_train labels
def load_labels(cdl_filepath, dbf_filepath):
    # Get the CDL image
    cdl_image = load_image_as_numpy(cdl_filepath)

    # Get the dbf table as a pandas DataFrame
    table = dbf.Table(dbf_filepath)
    table.open()
    df = pd.DataFrame(iter(table))
    table.close()

    label_map = {}
    for index, row in df.iterrows():
        if row['VALUE'] == 1:
            label_map[row['crop_id']] = 1 # corn
        elif row['VALUE'] == 5:
            label_map[row['crop_id']] = 2 # soybean
        else:
            label_map[row['crop_id']] = 0 # other
    
    labels = np.vectorize(label_map.get)(cdl_image)

    return labels


# TODO function to initialize the model


# TODO function to train the model

# TODO function to run the model

# TODO function to evaluate and output


