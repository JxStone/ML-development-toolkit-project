import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import cv2 as cv
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tqdm import tqdm
from skimage import io, transform
from sklearn.model_selection import train_test_split
from sklearn import metrics


# set random seed
def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

def create_file_name(x):
    file_name = f'input_{x[0]}_{x[1]}_{x[2]}.jpg'
    return file_name

# check for missing data
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

def create_datasets(df, img_root, img_size, n):
    imgs = []
    for filename in tqdm(df['file']):
        img = io.imread(img_root+filename)
        img = transform.resize(img, (img_size,img_size,n))
        imgs.append(img)
        
    X = np.array(imgs)
    y = pd.get_dummies(df['character'])
    return X, y

def get_lr_callback(batch_size=32, plot=False, epochs=50):
    lr_start   = 0.003
    lr_max     = 0.00125 * batch_size
    lr_min     = 0.001
    lr_ramp_ep = 20
    lr_sus_ep  = 0
    lr_decay   = 0.8
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    if plot == True:
        rng = [i for i in range(epochs)]
        y = [lrfn(x) for x in rng]
        plt.plot(rng, y)
        plt.xlabel('epoch', size=14); plt.ylabel('learning_rate', size=14)
        plt.title('Training Schedule', size=16)
        plt.show()
    return lr_callback

def show_images(df, isTest=False, path='chinese-mnist/data/data/'):
    f, ax = plt.subplots(10,15, figsize=(15,10))
    for i,idx in enumerate(df.index):
        dd = df.iloc[idx]
        image_name = dd['file']
        image_path = os.path.join(path, image_name)
        img_data = cv.imread(image_path)
        ax[i//15, i%15].imshow(img_data)
        ax[i//15, i%15].axis('off')
    plt.show()

def main():
    # set configuration and read metadata
    SEED = 42
    EPOCHS = 50
    BATCH_SIZE = 32
    IMG_SIZE = 64
    IMG_PATH = 'chinese-mnist/data/data/'
    data_df = pd.read_csv('chinese-mnist/chinese_mnist.csv')

    seed_everything(SEED)

    # check for missing values
    print(missing_data(data_df))

    # explore image data
    image_files = list(os.listdir(IMG_PATH))
    print("Number of image files: {}".format(len(image_files)))

    data_df["file"] = data_df.apply(create_file_name, axis=1)

    # check suites of the image
    print(f"Number of suites: {data_df.suite_id.nunique()}")
    print(f"Samples: {data_df.sample_id.nunique()}: {list(data_df.sample_id.unique())}")
    print(f"Characters codes: {data_df.code.nunique()}: {list(data_df.code.unique())}")
    print(f"Characters: {data_df.character.nunique()}: {list(data_df.character.unique())}")
    print(f"Numbers: {data_df.value.nunique()}: {list(data_df.value.unique())}")

    # show sample images
    df = data_df.loc[data_df.suite_id==20].sort_values(by=["sample_id","value"]).reset_index()
    show_images(df)

    # create dataset
    train_df, test_df = train_test_split(data_df, 
                                     test_size=0.2,
                                     random_state=SEED,
                                     stratify=data_df['character'].values) 
    train_df, val_df = train_test_split(train_df,
                                    test_size=0.1,
                                    random_state=SEED,
                                    stratify=train_df['character'].values)

    train_X, train_y = create_datasets(train_df, IMG_PATH, IMG_SIZE, 1)
    val_X, val_y = create_datasets(val_df, IMG_PATH, IMG_SIZE, 1)
    test_X, test_y = create_datasets(test_df, IMG_PATH, IMG_SIZE, 1)

    # build a neural network model, add layers, apply label smoothing and add learning rate scheduler
    input_shape = (IMG_SIZE, IMG_SIZE, 1)

    model = Sequential()
    # model.add(Conv2D(16, kernel_size=3, input_shape=input_shape, activation='relu', padding='same'))
    # model.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'))
    # model.add(MaxPool2D(2))
    # model.add(Dropout(0.5))
    # model.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'))
    # model.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'))
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(Dense(train_y.columns.size, activation='softmax'))

    model.add(Conv2D(16, kernel_size=3, padding='same', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(3))
    model.add(Dropout(0.2))
    model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(3))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(15, activation='softmax'))
    opt = tfa.optimizers.LazyAdam()
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.025)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    print(model.summary())

    get_lr_callback(plot=True, epochs=EPOCHS)

    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, 
                                               verbose=1, 
                                               restore_best_weights=True)

    # fit the model with training set and validation set
    train_model = model.fit(train_X, 
                        train_y, 
                        batch_size=BATCH_SIZE, 
                        epochs=EPOCHS, 
                        callbacks=[es_callback, get_lr_callback(BATCH_SIZE)],
                        validation_data=(val_X, val_y))

    # plot accuracy and loss
    df1 = pd.DataFrame(train_model.history)[['accuracy', 'val_accuracy']].plot()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Epoch vs. Accuracy')
    plt.legend()

    df2 = pd.DataFrame(train_model.history)[['loss', 'val_loss']].plot()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs. Loss')
    plt.legend()

    plt.show()

    # evaluation metrics
    pred = model.predict(test_X)
    y_pred = np.argmax(pred, axis=1)
    y_true = np.argmax(test_y.values, axis=1)
    print(metrics.classification_report(y_true, y_pred, target_names=test_y.columns))

    result = model.evaluate(test_X, test_y)
    print('Loss function: %s, accuracy:' % result[0], result[1])

if __name__ == '__main__':
    main()
