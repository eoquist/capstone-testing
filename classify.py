# 1) Resize all images to same size.You can loop over each image and resize and save.
# 2) get the pixel vector of each image and create the dataset.
#    As a example if your cat images are in "Cat" folder and Dog images are in "Dog" folder,
#    iterate over all images inside the folder and get the pixel values.
#    Label the data as "cat"(cat=1) and "non-cat"(non-cat=0)

import os
import imageio
import pandas as pd

catimages = os.listdir("Cat")
dogimages = os.listdir("Dog")
catVec = []
dogVec = []
for img in catimages:
    img = imageio.imread(f"Cat/{img}")
    ar = img.flatten()
    catVec.append(ar)
catdf = pd.DataFrame(catVec)
catdf.insert(loc=0, column="label", value=1)

for img in dogimages:
    img = imageio.imread(f"Dog/{img}")
    ar = img.flatten()
    dogVec.append(ar)
dogdf = pd.DataFrame(dogVec)
dogdf.insert(loc=0, column="label", value=0)

# 3) concat catdf and dogdf and shuffle the dataframe

data = pd.concat([catdf, dogdf])
data = data.sample(frac=1)

# now you have dataset with lable for your images.
# 4) split dataset to train and test and fit to the model.
