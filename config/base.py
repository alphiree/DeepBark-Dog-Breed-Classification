import logging
import os
import warnings

import tensorflow as tf


def import_tensorflow():
    """Filter tensorflow version warnings"""

    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=Warning)

    tf.get_logger().setLevel("INFO")
    tf.autograph.set_verbosity(0)

    tf.get_logger().setLevel(logging.ERROR)
    return tf


class Config:
    model_folder = "saved-models"

    class_names = [
        "Afghan",
        "African Wild Dog",
        "Airedale",
        "American Hairless",
        "American Spaniel",
        "Basenji",
        "Basset",
        "Beagle",
        "Bearded Collie",
        "Bermaise",
        "Bichon Frise",
        "Blenheim",
        "Bloodhound",
        "Bluetick",
        "Border Collie",
        "Borzoi",
        "Boston Terrier",
        "Boxer",
        "Bull Mastiff",
        "Bull Terrier",
        "Bulldog",
        "Cairn",
        "Chihuahua",
        "Chinese Crested",
        "Chow",
        "Clumber",
        "Cockapoo",
        "Cocker",
        "Collie",
        "Corgi",
        "Coyote",
        "Dalmation",
        "Dhole",
        "Dingo",
        "Doberman",
        "Elk Hound",
        "French Bulldog",
        "German Sheperd",
        "Golden Retriever",
        "Great Dane",
        "Great Perenees",
        "Greyhound",
        "Groenendael",
        "Irish Spaniel",
        "Irish Wolfhound",
        "Japanese Spaniel",
        "Komondor",
        "Labradoodle",
        "Labrador",
        "Lhasa",
        "Malinois",
        "Maltese",
        "Mex Hairless",
        "Newfoundland",
        "Pekinese",
        "Pit Bull",
        "Pomeranian",
        "Poodle",
        "Pug",
        "Rhodesian",
        "Rottweiler",
        "Saint Bernard",
        "Schnauzer",
        "Scotch Terrier",
        "Shar_Pei",
        "Shiba Inu",
        "Shih-Tzu",
        "Siberian Husky",
        "Vizsla",
        "Yorkie",
    ]
