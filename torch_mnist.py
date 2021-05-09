import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.datasets import ImageFolder
import torch.nn as nn
import tf_mnist

def main():

    # take a look at the dataset csv
    df_chinese_mnist = pd.read_csv('chinese-mnist/chinese_mnist.csv')
    print(df_chinese_mnist)


    # see what the data picture looks like
    image = Image.open(f"chinese-mnist/data/data/input_1_1_1.jpg")
    plt.imshow(np.array(image), cmap='gray')
    # plt.show()

    SEED = 42
    EPOCHS = 50
    BATCH_SIZE = 32
    IMG_SIZE = 64
    IMG_PATH = 'chinese-mnist/data/data/'

    torch.manual_seed(SEED)
    data_len = len(df_chinese_mnist)
    test_len = int(0.2*data_len)
    train_val_ds_len = data_len - test_len
    valid_len = int(0.1 * train_val_ds_len)
    # split dataset into test and train and validation
    test_ds, train_val_ds = torch.utils.data.random_split(df_chinese_mnist, (test_len, data_len - test_len))
    val_ds, train_ds = torch.utils.data.random_split(train_val_ds, (valid_len, train_val_ds_len - valid_len))

    print(val_ds[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize model
    model = ConvolutionNetwork()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.CrossEntropyLoss()


if __name__ == '__main__':
    main()