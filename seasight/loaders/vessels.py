from typing import Dict, Literal, Tuple

import tensorflow as tf


class VesselDatasetLoader:
    """
    A loader for creating TensorFlow datasets from TFRecord files for vessel detection.

    Attributes:
        dataset_dir (str): Directory containing the TFRecord files.
        batch_size (int): Number of samples per batch.
        autotune (tf.data.experimental.AUTOTUNE): Constant used to denote that the number of parallel calls is to be determined at runtime based on available CPU.
        label_map (dict): Mapping from label IDs to human-readable string labels.
    """

    def __init__(
        self,
        dataset_dir: str,
        batch_size: int = 32,
        autotune: int = tf.data.experimental.AUTOTUNE,
    ):
        """
        Initializes the vessel dataset loader class.
        """
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.autotune = autotune
        self.label_map = {
            1: "Fishing Boat",
            2: "Merchant Ship",
            3: "Military Ship",
            4: "Patrol Boat",
            5: "Sails Boat",
            6: "Submarine",
            7: "Tugboat",
            8: "Yacht",
        }

    def parse_tfrecord_fn(
        self, example: tf.Tensor
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Parses a single example from a TFRecord file.

        Parameters:
            example (tf.Tensor): A serialized example from a TFRecord file.

        Returns:
            tuple: A tuple containing the decoded image and a dictionary with the corresponding labels and bounding boxes.
        """
        feature_description = {
            "image/encoded": tf.io.FixedLenFeature([], tf.string),
            "image/object/class/label": tf.io.VarLenFeature(tf.int64),
            "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
        }
        example = tf.io.parse_single_example(example, feature_description)
        image = tf.image.decode_jpeg(example["image/encoded"], channels=3)
        labels = tf.sparse.to_dense(example["image/object/class/label"])
        xmin = tf.sparse.to_dense(example["image/object/bbox/xmin"])
        xmax = tf.sparse.to_dense(example["image/object/bbox/xmax"])
        ymin = tf.sparse.to_dense(example["image/object/bbox/ymin"])
        ymax = tf.sparse.to_dense(example["image/object/bbox/ymax"])
        bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
        return image, {"labels": labels, "bboxes": bboxes}

    def create_dataset(
        self, subset: Literal["test", "train", "valid"]
    ) -> tf.data.Dataset:
        """
        Creates a TensorFlow dataset for a specific subset of the data (train, validation, test).

        Parameters:
            subset (str): The subset of the dataset to load ('train', 'valid', 'test').

        Returns:
            tf.data.Dataset: The constructed TensorFlow dataset.
        """
        tfrecord_path = f"{self.dataset_dir}/{subset}/Sea-Vessels.tfrecord"
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        parsed_dataset = raw_dataset.map(
            self.parse_tfrecord_fn, num_parallel_calls=self.autotune
        )
        return parsed_dataset

    def get_dataloader(
        self, subset: Literal["test", "train", "valid"]
    ) -> tf.data.Dataset:
        """
        Prepares and returns a DataLoader for a given subset of the dataset.

        Parameters:
            subset (str): The subset for which to prepare the DataLoader ('train', 'valid', 'test').

        Returns:
            tf.data.Dataset: A DataLoader object ready for model training or evaluation.
        """
        dataset = self.create_dataset(subset)
        if subset == "train":
            dataset = dataset.shuffle(1000)

        padded_shapes = ([None, None, 3], {"labels": [None], "bboxes": [None, 4]})
        padding_values = (
            tf.constant(0, dtype=tf.uint8),
            {
                "labels": tf.constant(-1, dtype=tf.int64),
                "bboxes": tf.constant(0.0, dtype=tf.float32),
            },
        )

        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=False,
        ).prefetch(buffer_size=self.autotune)

        return dataset
