from keras.callbacks import ModelCheckpoint, Callback

import glob, re, os
from utils import divider
from constants import WEIGHTS_BASE_DIR, WEIGHTS_EXTENSION, VALIDATION_SPLIT


def train_model(model, train_data, train_labels, val_data, batch_size, epochs, augment, model_name, verbose=1):
    """
    Training wrappper. Trains a given model with given params. Can be set to augment if needed. Checkpoints the best
    weights, and cleans them up when finished training.

    :param model: The model being trained
    :param train_data: Training data
    :param train_labels: Training labels
    :param val_data: Validation data. If none, 'val_split' times of the training data will be used as validation data
    :param batch_size: Batch size used when training
    :param epochs: Epochs to train
    :param augment: Augmentation object
    :param model_name: Name of the model. Model will be saved with this name
    :param verbose: Level of output
    :return:
    """
    # Callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        WEIGHTS_BASE_DIR + model_name + "_{epoch:03d}" + WEIGHTS_EXTENSION,
        verbose=1,
        save_best_only=True,
        monitor="val_loss"
    )

    # Callback to cleanup the checkpoints
    delete_checkpoint = CheckpointCleanup(model_name)

    # Use some of the training data as validation data if there is no validation data specified
    validation_split = VALIDATION_SPLIT if not val_data or len(val_data) <= 0 else 0

    # Fit the model according to params
    if augment is None:
        model.fit(train_data,
                  train_labels,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=verbose,
                  validation_data=val_data,
                  validation_split=validation_split
                  )
    else:
        # Set the augmentation and append it to the training data
        datagen = augment
        datagen.fit(train_data)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(train_data, train_labels, batch_size=batch_size),
                            epochs=epochs,
                            steps_per_epoch=len(train_data) // batch_size,
                            validation_steps=len(train_data) // batch_size,
                            callbacks=[checkpoint_callback, delete_checkpoint]
                            )


class CheckpointCleanup(Callback):
    """
    Custom callback to cleanup all excess checkpoint. Primarely used on the end of a training
    """

    def __init__(self, model_name):
        super(CheckpointCleanup, self).__init__()
        self.model_name = model_name

    def on_train_end(self, batch, logs={}):
        divider(70)
        print("Deleting all excess checkpoints")
        files = [file for file in glob.glob(WEIGHTS_BASE_DIR + self.model_name + '*' + WEIGHTS_EXTENSION)]
        numbs = []
        for file in files:
            try:
                num = int(re.search("\d{3}", file).group())
                numbs.append(num)
            except AttributeError:
                pass

        max_numb = max(numbs) if len(numbs) > 0 else 0
        max_numb_str = "{:03d}".format(max_numb)
        print("Best checkpoint was in epoch {}".format(max_numb_str))
        for file in files:
            if max_numb_str in file:
                os.rename(file, WEIGHTS_BASE_DIR + self.model_name + WEIGHTS_EXTENSION)
            else:
                os.remove(file)
