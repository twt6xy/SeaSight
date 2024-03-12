import matplotlib.pyplot as plt
import tensorflow as tf


class TFRecordPlots:
    feature_description = {"image/encoded": tf.io.FixedLenFeature([], tf.string)}

    @staticmethod
    def _parse_function(example_proto):
        return tf.io.parse_single_example(
            example_proto, TFRecordPlots.feature_description
        )

    @staticmethod
    def display_random_image_from_tfrecord(tfrecord_path):
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        parsed_dataset = raw_dataset.map(TFRecordPlots._parse_function)

        for raw_record in parsed_dataset.shuffle(buffer_size=1000).take(1):
            image = tf.image.decode_jpeg(raw_record["image/encoded"].numpy())
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis("off")
            plt.show()
