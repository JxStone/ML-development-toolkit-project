import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.datasets import ImageFolder
import torch.nn as nn
from tf_mnist import *
import torch_model
from sklearn.model_selection import train_test_split


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
    data_df = pd.read_csv('chinese-mnist/chinese_mnist.csv')

    torch.manual_seed(SEED)
    # data_len = len(df_chinese_mnist)
    # test_len = int(0.2*data_len)
    # train_val_ds_len = data_len - test_len
    # valid_len = int(0.1 * train_val_ds_len)
    # # split dataset into test and train and validation
    # test_ds, train_val_ds = torch.utils.data.random_split(df_chinese_mnist, (test_len, data_len - test_len))
    # val_ds, train_ds = torch.utils.data.random_split(train_val_ds, (valid_len, train_val_ds_len - valid_len))

    # create dataset
    train_df, test_df = train_test_split(data_df,
                                         test_size=0.2,
                                         random_state=SEED,
                                         stratify=data_df['character'].values)
    train_df, val_df = train_test_split(train_df,
                                        test_size=0.1,
                                        random_state=SEED,
                                        stratify=train_df['character'].values)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_X, train_y = create_datasets(train_df, IMG_PATH, IMG_SIZE, 1)
    val_X, val_y = create_datasets(val_df, IMG_PATH, IMG_SIZE, 1)
    test_X, test_y = create_datasets(test_df, IMG_PATH, IMG_SIZE, 1)

    # initialize model
    model = torch_model.ConvolutionNetwork()
    model = model.to(device)

    optimizer = torch.optim.SparseAdam(model.parameters(), lr=10e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.CrossEntropyLoss()


if __name__ == '__main__':
    main()