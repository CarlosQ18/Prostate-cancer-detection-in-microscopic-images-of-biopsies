import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


def Predicted_model(ruta_directorio,model):

    CLASSID = ['0','3','4','5']

    Imagenes = tf.keras.utils.image_dataset_from_directory(
        directory=ruta_directorio,
        labels=None,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=32,
        image_size=(512, 512),
        seed=123
    )

    predicted = []
    labels = []

    for im in Imagenes:
        predictions = model(im)
        if np.any(predictions):
            predicted.append(predictions)

    if predicted:
        pred = np.concatenate([np.argmax(p, axis=-1).flatten() for p in predicted])
        decoded_labels = np.array(CLASSID)[pred].tolist()
        return decoded_labels
    else:
        return ["No predictions"]


##model1 = load_model(r"C:\Users\Carlos F. Quintero\Desktop\Test Definitivo\EfficientNetB0\EfficientNetB0")
##ruta_directorio1 = r"C:\Users\Carlos F. Quintero\Desktop\Test Definitivo\Grado 0\3faad4959dbc001b2529a158822193df\Parches\Original"
##
##print(Predicted_model(ruta_directorio1,model1))
