from keras_preprocessing.image import ImageDataGenerator


def get_augmentation(rotation=0, width_sr=0.0, height_sr=0.0, shear=0.0, zoom=0):
    return ImageDataGenerator(
        rotation_range=rotation,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=width_sr,  # randomly shift images vertically (fraction of total height)
        height_shift_range=height_sr,  # randomly shift images vertically (fraction of total height)
        shear_range=shear,  # set range for random shear
        zoom_range=zoom)  # set range for random zoom
