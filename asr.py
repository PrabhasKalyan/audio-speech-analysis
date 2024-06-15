import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


audio="/Users/prabhaskalyan/Desktop/asr/Audio"

#loads data from directory
train_ds=tf.keras.utils.audio_dataset_from_directory(
    directory=audio,
    batch_size=64, #loads particular no.of audio files to ram to train the model
    seed=0, #using a fixed seed value makes sure that data is not shuffled
    output_sequence_length=16000
)

#converts audio signal from wave form to spectrogram
def get_spectro(wave):
    spectrogram=tf.signal.stft(wave,frame_length=256,frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram

audio_data,label=next(iter(train_ds))
audio_data=np.reshape(audio_data,(64,16000))
for i in range(64):
    spectrogram=get_spectro(audio_data)


model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(64,124,129)),
    tf.keras.layers.Resizing(32, 32),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(spectrogram,label,epochs=7)

model.summary()




