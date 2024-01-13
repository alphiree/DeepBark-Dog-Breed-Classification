"""Main Script for DeepBark Dog Breed Classification"""

import os

from keras.models import load_model

from config.base import Config
from deeplearning_functions import (
    multiclass_predict_from_image_with_comparison,
    read_image,
)

model = load_model(
    os.path.join(f"{os.getcwd()}", f"{Config.model_folder}", "mobilenetV2.h5")
)

print("")
print("DeepBark: Dog Breed Classification by: Alphire :)")
print("")
# Get input string from user
input_string = input("Enter the file path: ")
# Add another backslash to the string
new_string = input_string.replace("\\", "\\\\")
# Remove double quotes from the string
new_string = new_string.replace('"', "")

image_read = read_image(new_string)


if __name__ == "__main__":
    multiclass_predict_from_image_with_comparison(
        model,
        image=image_read,
        image_size=(224, 224),
        class_names=Config.class_names,
        comparison_path="train",
        picture_title="Dog Breed Classification",
    )
