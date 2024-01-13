"""Contains the functions used in the main script.

"""
# pylint: disable=E1101 (no-member)
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import img_to_array

from config.base import import_tensorflow

tf = import_tensorflow()


def multiclass_predict_from_image_with_comparison(
    model, image, image_size, class_names, comparison_path, picture_title
):
    plot_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plot_image_2 = cv2.resize(plot_image, (400, 400))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ## Preprocessing
    image = tf.image.resize(image, image_size)
    image_array = img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)

    ## Predicting model
    score = model.predict(image_array)
    # print(f'The score list is {score}')
    index = np.argmax(score)
    score = np.ravel(score)
    rounded_score = np.round(score[index], 4)
    print(f"rounded off score is: {rounded_score}")

    breed = class_names[index]

    comparison_folder = os.path.join(comparison_path, class_names[index])

    print("File Count inside the comparison path:")
    print(
        len(
            [
                entry
                for entry in os.listdir(comparison_folder)
                if os.path.isfile(os.path.join(comparison_folder, entry))
            ]
        )
    )

    image_list = []

    for images in os.listdir(comparison_folder):
        image_list.append(images)

    # read all the images
    # we are going to take 4 images only
    image1 = cv2.imread(os.path.join(comparison_folder, image_list[0]))
    image2 = cv2.imread(os.path.join(comparison_folder, image_list[1]))
    image3 = cv2.imread(os.path.join(comparison_folder, image_list[2]))
    image4 = cv2.imread(os.path.join(comparison_folder, image_list[3]))

    # make all the images of same size
    # so we will use resize function
    image1 = cv2.resize(image1, (200, 200))
    image2 = cv2.resize(image2, (200, 200))
    image3 = cv2.resize(image3, (200, 200))
    image4 = cv2.resize(image4, (200, 200))

    horizontal1 = np.hstack([image1, image2])
    horizontal2 = np.hstack([image3, image4])

    vertical_attachment = np.vstack([horizontal1, horizontal2])

    final_image = cv2.cvtColor(vertical_attachment, cv2.COLOR_BGR2RGB)

    ## Plotting it
    fig, axes = plt.subplots(  # pylint: disable=W0612
        1,  ## Number of columns
        2,  ## Number of rows
        figsize=(12, 6),  ## Figure Size
    )

    ## Plotting the Subplots
    axes[0].imshow(plot_image_2)
    axes[0].title.set_text(f"Predicted Class: {breed}")
    axes[0].axis("off")

    axes[1].imshow(final_image)
    axes[1].title.set_text(breed)
    axes[1].axis("off")

    ## If you have more than 1 columns and want to put title in those specific columns
    plt.suptitle(picture_title)

    plt.show()


def read_image(path):
    image = cv2.imread(path)
    return image
