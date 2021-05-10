import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.datasets import ImageFolder
import torch.nn as nn
from tf_mnist import *
from torch_model import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def create_datasets(df, img_root, img_size):
    imgs = []
    for filename in tqdm(df['file']):
        img = io.imread(img_root + filename)
        img = transform.resize(img, (1, img_size, img_size))
        imgs.append(img)

    X = np.array(imgs)
    y = df['code'].to_numpy()
    return X, y

def main():

    # # take a look at the dataset csv
    # df_chinese_mnist = pd.read_csv('chinese-mnist/chinese_mnist.csv')
    # print(df_chinese_mnist)
    #
    #
    # # see what the data picture looks like
    # image = Image.open(f"chinese-mnist/data/data/input_1_1_1.jpg")
    # plt.imshow(np.array(image), cmap='gray')
    # plt.show()

    SEED = 42
    EPOCHS = 50
    BATCH_SIZE = 32
    IMG_SIZE = 64
    IMG_PATH = 'chinese-mnist/data/data/'
    data_df = pd.read_csv('chinese-mnist/chinese_mnist.csv')
    data_df["file"] = data_df.apply(create_file_name, axis=1)

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
    train_X, train_y = create_datasets(train_df, IMG_PATH, IMG_SIZE)
    val_X, val_y = create_datasets(val_df, IMG_PATH, IMG_SIZE)
    test_X, test_y = create_datasets(test_df, IMG_PATH, IMG_SIZE)


    # initialize model
    model = ConvolutionNetwork()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.CrossEntropyLoss()

    train_loss = []
    valid_loss = []
    accuracy = []
    # train the model and evaluate using validation set
    model.train()
    train_dataloader = torch.utils.data.DataLoader(Dataset(train_X, train_y), batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(Dataset(val_X, val_y), batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(EPOCHS):
        # keep the loss for each epoch
        train_eploss = 0
        for i, data in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            data_label = data
            inputs_data, batch_label = data_label
            output = model(inputs_data.to(device).type(torch.float))
            batch_label = torch.stack(list(batch_label), dim=0)
            loss = criterion(output, batch_label.to(device))
            train_eploss += loss.item()
            loss.backward()
            optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i+1) * BATCH_SIZE, len(train_dataloader.dataset),
                100. * i / len(train_dataloader), loss.item()))
        train_loss.append(train_eploss/len(train_dataloader.dataset))
        # validation set
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_dataloader:
                data, target = data.to(device).type(torch.float), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy.append(correct/len(val_dataloader.dataset))
        val_loss /= len(val_dataloader.dataset)
        valid_loss.append(val_loss)
        print("Validation set loss is " + str(val_loss))


    # test
    test_dataloader = torch.utils.data.DataLoader(Dataset(test_X, test_y), batch_size=BATCH_SIZE, shuffle=True)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device).type(torch.float), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))


    # plot the data
    epochs = np.arange(0, EPOCHS)
    print(train_loss)
    print(valid_loss)
    print(epochs)
    plt.figure(0)
    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, valid_loss, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(1)
    plt.plot(epochs, accuracy)
    plt.show()

if __name__ == '__main__':
    main()