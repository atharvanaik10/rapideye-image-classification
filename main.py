import tifffile
import numpy as np
from dbfread import DBF
import pandas as pd

# TODO define a function to get pixel wise y_train labels
def load_labels(cdl_filepath, dbf_filepath, corn_label, soybean_label):
    # Get the CDL image
    cdl_image = tifffile.imread(cdl_filepath)
    # Convert to 1 (corn), 5 (soybean), and 0 (other) for labels
    cdl_image[(cdl_image != corn_label) & (cdl_image != soybean_label)] = 0

    # Get the dbf table as a pandas DataFrame
    table = DBF(dbf_filepath)
    df = pd.DataFrame(iter(table))

    return cdl_image, df


# TODO function to initialize the model

# TODO function to train the model

# TODO function to run the model

# TODO function to evaluate and output


if __name__ == "__main__":
    # Global variables
    north_image = "data/20130824_RE3_3A_Analytic_Champaign_north.tif"
    cdl_image = "data/CDL_2013_Champaign_north.tif"
    cdl_dbf = "data/CDL_2013_clip_20170525181724_1012622514.tif.vat.dbf"
    corn_label = 1
    soybean_label = 5


    print("Running image classification...")
    print("Loading training data")
    train_x = tifffile.imread(north_image)
    train_y, dbf_table = load_labels(cdl_image, cdl_dbf, corn_label, soybean_label)
    print(train_y)
