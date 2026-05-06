import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import os

DATASET_PATH = r"C:\twizy\BDD_Roboflow_Tazi\BDD_Roboflow_Tazi\train"
DATASET_PATH2 = r"C:\twizy\BDD_Roboflow_Tazi\BDD_Roboflow_Tazi\test"
CSV_PATH = r"C:\twizy\BDD_Roboflow_Tazi\BDD_Roboflow_Tazi\train\_classes.csv"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32


df = pd.read_csv(CSV_PATH)

class_names = df.columns[1:]
print("Classes :", list(class_names))

CSV_PATH_TEST = r"C:\twizy\BDD_Roboflow_Tazi\BDD_Roboflow_Tazi\test\_classes.csv"

df_test = pd.read_csv(CSV_PATH_TEST)

filepaths2 = df_test['filename'].apply(lambda x: os.path.join(DATASET_PATH2, x)).values
labels2 = df_test[class_names].values.astype("float32")

filepaths = df['filename'].apply(lambda x: os.path.join(DATASET_PATH, x)).values
labels = df[class_names].values.astype("float32")


dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
dataset2 = tf.data.Dataset.from_tensor_slices((filepaths2, labels2))

dataset = dataset.shuffle(
    buffer_size=len(filepaths),
    seed=42,
    reshuffle_each_iteration=False
)


train_dataset = dataset
val_dataset = dataset2

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, label

train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])




model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    data_augmentation,

    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(),

    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(),

    
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='sigmoid')
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),
    ]
)


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        patience=2,
        factor=0.5
    )
]


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    callbacks=callbacks
)

results = model.evaluate(val_dataset)

print("Perte :", results[0])
print("Accuracy :", results[1])

test_path = ""

def predict_image(model, image_path, class_names):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0

    img = tf.expand_dims(img, axis=0)

    probs = model.predict(img)[0]

    print("\nProbabilités par classe :")
    for name, p in zip(class_names, probs):
        print(f"{name}: {p:.4f}")

    best_idx = np.argmax(probs)
    print("\nPrédiction finale :", class_names[best_idx], f"({probs[best_idx]:.2f})")