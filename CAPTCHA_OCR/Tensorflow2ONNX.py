import tf2onnx
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CTCLayer(layers.Layer):
    def __init__(self, name=None, ** kwargs):
        super().__init__(name=name, ** kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):

        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred

# 기 저장된 모델 불러오기(from tenorflow)
new_model = keras.models.load_model('model1.h5', custom_objects={'CTCLayer':CTCLayer})

# 예측 모델 정의하기
prediction_model = keras.models.Model(
    new_model.get_layer(name="image").input, new_model.get_layer(name="dense2").output
)

# .onnx로 전환
spec = (tf.TensorSpec((None, 200, 50, 1), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(prediction_model, input_signature=spec, output_path = "./ONNX/onnx_model.onnx")

