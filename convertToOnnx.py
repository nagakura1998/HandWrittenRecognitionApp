from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import load_model
import tf2onnx
import tensorflow as tf
import onnx


model = tf.keras.Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))

model.add(Dense(36,activation ="softmax"))

model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model = load_model("model_hand.h5")

def export_to_onnx(model):
   # convert to onnx model
   input_signature = [tf.TensorSpec([1, 28, 28], tf.float32, name='x')]
   onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
   onnx.save(onnx_model, "C:\\Users\\trong\\Downloads\\test.onnx")

export_to_onnx(model)