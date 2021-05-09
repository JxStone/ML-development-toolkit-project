import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# take a look at the dataset csv
df_chinese_mnist = pd.read_csv('chinese-mnist/chinese_mnist.csv')
print(df_chinese_mnist)


# see what the data picture looks like
image = Image.open(f"chinese-mnist/data/data/input_1_1_1.jpg")
plt.imshow(np.array(image), cmap='gray')
plt.show()

