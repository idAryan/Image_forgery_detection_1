import cv2
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist
from tensorflow import keras
from keras.saving import register_keras_serializable

# Register the custom ResizeLayer used in your CNN model
@register_keras_serializable()
class ResizeLayer(keras.layers.Layer):
    def __init__(self, target_size, **kwargs):
        super().__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size)

    def get_config(self):
        config = super().get_config()
        config.update({"target_size": self.target_size})
        return config


class MultiScaleMultiStageCNN:
    """Loads and applies a trained CNN model for forgery prediction."""
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.load(model_path)

    def load(self, model_path):
        """Load a trained Keras model with custom objects."""
        self.model = keras.models.load_model(
            model_path,
            custom_objects={"ResizeLayer": ResizeLayer}
        )

    def predict_forgery_mask(self, image_normalized):
        """Predict a forgery mask using the CNN model."""
        if self.model is None:
            raise ValueError("CNN model not loaded.")
        input_tensor = np.expand_dims(image_normalized, axis=0)  # Shape: (1, 256, 256, 3)
        prediction = self.model.predict(input_tensor, verbose=0)[0]  # Shape: (256, 256, 1)
        if prediction.ndim == 3:
            prediction = prediction[:, :, 0]
        return prediction  # Float mask (values between 0 and 1)


class CopyMoveForgeryDetector:
    """Hybrid detection system combining traditional and deep learning approaches."""

    def __init__(self, model_path=None):
        self.block_size = 16
        self.overlap = 8
        self.similarity_threshold = 0.85
        self.cnn_model = MultiScaleMultiStageCNN(model_path)

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (256, 256))
        image_normalized = image_resized.astype(np.float32) / 255.0
        return image, image_resized, image_normalized

    def block_based_detection(self, image):
        h, w = image.shape[:2]
        blocks, positions = [], []

        for i in range(0, h - self.block_size + 1, self.overlap):
            for j in range(0, w - self.block_size + 1, self.overlap):
                block = image[i:i+self.block_size, j:j+self.block_size]
                if block.shape[:2] == (self.block_size, self.block_size):
                    gray_block = cv2.cvtColor(block, cv2.COLOR_RGB2GRAY)
                    blocks.append(gray_block.flatten())
                    positions.append((i, j))

        if len(blocks) < 2:
            return np.zeros((h, w), dtype=np.uint8)

        blocks = np.array(blocks)
        distances = cdist(blocks, blocks, metric='euclidean')
        forgery_mask = np.zeros((h, w), dtype=np.uint8)

        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                pos1, pos2 = positions[i], positions[j]
                distance_between_blocks = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                if (distances[i, j] < (1 - self.similarity_threshold) * np.max(distances) and
                        distance_between_blocks > self.block_size * 2):
                    forgery_mask[pos1[0]:pos1[0]+self.block_size, pos1[1]:pos1[1]+self.block_size] = 255
                    forgery_mask[pos2[0]:pos2[0]+self.block_size, pos2[1]:pos2[1]+self.block_size] = 255

        return forgery_mask

    def deep_learning_detection(self, image_normalized):
        forgery_mask = self.cnn_model.predict_forgery_mask(image_normalized)
        return (forgery_mask > 0.5).astype(np.uint8) * 255  # Binary mask

    def hybrid_detection(self, image, image_normalized):
        traditional_mask = self.block_based_detection(image)
        dl_mask = self.deep_learning_detection(image_normalized)

        if dl_mask.shape != traditional_mask.shape:
            dl_mask = cv2.resize(dl_mask, (traditional_mask.shape[1], traditional_mask.shape[0]))

        combined_mask = cv2.bitwise_and(traditional_mask, dl_mask)
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        return traditional_mask, dl_mask, combined_mask

    def detect_forgery(self, image_path):
        """Full pipeline: Preprocessing → Traditional + DL → Decision"""
        image, resized, normalized = self.preprocess_image(image_path)
        traditional_mask, dl_mask, combined_mask = self.hybrid_detection(resized, normalized)

        forgery_pixels = np.sum(combined_mask > 0)
        total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
        confidence = min((forgery_pixels / total_pixels) * 100, 100)
        is_forged = confidence > 5.0

        return {
            'is_forged': is_forged,
            'confidence_score': confidence
        }
