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

# define model
class ConvolutionNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionNetwork, self).__init__()

        # Define the convolutions

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.conv2_bn = nn.BatchNorm2d(16)
        self.maxP1 = nn.MaxPool2d(3)
        self.drop1 = nn.Dropout2d(0.2)

        self.conv3 = nn.Conv2d(1, 16, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3)
        self.relu4 = nn.ReLU()
        self.conv4_bn = nn.BatchNorm2d(16)
        self.maxP2 = nn.MaxPool2d(3)
        self.drop2 = nn.Dropout2d(0.2)

        self.flat = nn.Flatten()
        # self.dense = nn.Linear(, 15)
        self.soft = nn.Softmax()


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv2_bn(x)
        x = self.maxP1(x)
        x = self.drop1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv4_bn(x)
        x = self.maxP2(x)
        x = self.drop2(x)

        x = self.flat(x)
        print(x.size)
        x = self.soft(x)
        return x

if __name__ == '__main__':
    main()