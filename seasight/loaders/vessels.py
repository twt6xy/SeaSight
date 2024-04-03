from typing import Dict, Literal

import keras_cv
import tensorflow as tf


class VesselDatasetLoader:
    def __init__(self, dataset_dir: str, label_map_path: str):
        self.dataset_dir = dataset_dir
        self.label_map = self.parse_label_map(label_map_path)
        self.autotune = tf.data.AUTOTUNE

    def parse_label_map(self, label_map_path: str):
        """
        Parses the label mapping from tthe pbtxt file

        Parameters:
            label_map_path: (str): Path to one of the pbtxt files.

        Returns:
            dict: A dictionary containing the label mapping
        """
        class_table = {}
        with open(label_map_path, "r") as file:
            for line in file:
                line = line.strip()
                if line.startswith("name:"):
                    class_name = line.split('"')[1]
                elif line.startswith("id:"):
                    class_id = int(line.split(":")[1].strip().rstrip(","))
                    class_table[class_id] = class_name
        return class_table

    def parse_tfrecord_fn(self, example: tf.Tensor, bounding_box_format: str) -> Dict:
        """
        Parses a TFRecord example into formatted images and bounding boxes.

        Parameters:
        - example: A serialized TFRecord example.
        - bounding_box_format: The target format for bounding boxes.

        Returns:
        A dictionary with keys 'images' and 'bounding_boxes'.
        """
        # Feature description for TFRecord example parsing
        feature_description = {
            "image/encoded": tf.io.FixedLenFeature([], tf.string),
            "image/object/class/label": tf.io.VarLenFeature(tf.int64),
            "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
        }
        parsed_example = tf.io.parse_single_example(example, feature_description)

        # Decode the image
        image = tf.image.decode_jpeg(parsed_example["image/encoded"], channels=3)
        image = tf.cast(image, tf.float32)
        image /= 255.0

        # Prepare bounding boxes
        xmin = tf.sparse.to_dense(parsed_example["image/object/bbox/xmin"])
        xmax = tf.sparse.to_dense(parsed_example["image/object/bbox/xmax"])
        ymin = tf.sparse.to_dense(parsed_example["image/object/bbox/ymin"])
        ymax = tf.sparse.to_dense(parsed_example["image/object/bbox/ymax"])
        boxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)

        # Convert labels to dense and format bounding boxes
        labels = tf.sparse.to_dense(parsed_example["image/object/class/label"])
        boxes = keras_cv.bounding_box.convert_format(
            boxes, source="yxyx", target=bounding_box_format, images=image
        )

        # Construct the return dictionary
        return {
            "images": image,
            "bounding_boxes": {
                "boxes": boxes,
                "classes": labels,
            },
        }

    def parse_tfrecord_fn_one_hot(
        self, example: tf.Tensor, bounding_box_format: str
    ) -> Dict:
        """
        Parses a TFRecord example into formatted images and bounding boxes, with labels as one-hot-encoded vectors.

        Parameters:
        - example: A serialized TFRecord example.
        - bounding_box_format: The target format for bounding boxes.

        Returns:
        A dictionary with keys 'images' and 'bounding_boxes'.
        """
        # Feature description for TFRecord example parsing
        feature_description = {
            "image/encoded": tf.io.FixedLenFeature([], tf.string),
            "image/object/class/label": tf.io.VarLenFeature(tf.int64),
            "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
        }
        parsed_example = tf.io.parse_single_example(example, feature_description)

        # Decode the image
        image = tf.image.decode_jpeg(parsed_example["image/encoded"], channels=3)
        image = tf.cast(image, tf.float32) / 255.0  # Normalize image pixels to [0, 1]

        # Prepare bounding boxes
        xmin = tf.sparse.to_dense(parsed_example["image/object/bbox/xmin"])
        xmax = tf.sparse.to_dense(parsed_example["image/object/bbox/xmax"])
        ymin = tf.sparse.to_dense(parsed_example["image/object/bbox/ymin"])
        ymax = tf.sparse.to_dense(parsed_example["image/object/bbox/ymax"])
        boxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)

        # Convert labels to one-hot-encoded vectors
        labels = tf.sparse.to_dense(parsed_example["image/object/class/label"])
        one_hot_labels = tf.one_hot(labels, depth=8)  # Assuming 8 classes

        # Optional: If you need to adjust bounding boxes format
        if bounding_box_format != "yxyx":
            boxes = keras_cv.bounding_box.convert_format(
                boxes, source="yxyx", target=bounding_box_format, images=image
            )

        # Construct the return dictionary
        return {
            "images": image,
            "bounding_boxes": {
                "boxes": boxes,
                "classes": one_hot_labels,
            },
        }

    def load_dataset(
        self,
        split: Literal["train", "test", "valid"],
        bounding_box_format: str,
        batch_size: int,
    ) -> tf.data.Dataset:
        """
        Prepares and returns a DataLoader for a given subset of the dataset.

        Parameters:
            subset (str): The subset for which to prepare the DataLoader ('train', 'valid', 'test').

        Returns:
            tf.data.Dataset: A DataLoader object ready for model training or evaluation.
        """
        tfrecord_path = f"{self.dataset_dir}/{split}/Sea-Vessels.tfrecord"
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.map(
            lambda x: self.parse_tfrecord_fn_one_hot(x, bounding_box_format),
            num_parallel_calls=self.autotune,
        )

        if split == "train":
            dataset = dataset.shuffle(batch_size * 4)

        dataset = dataset.ragged_batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self.autotune)

        return dataset

    def load_padded_dataset(
        self,
        split: Literal["train", "test", "valid"],
        bounding_box_format: str,
        batch_size: int,
    ) -> tf.data.Dataset:
        """
        Prepares and returns a DataLoader for a given subset of the dataset.
        """
        tfrecord_path = f"{self.dataset_dir}/{split}/Sea-Vessels.tfrecord"
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.map(
            lambda x: self.parse_tfrecord_fn(x, bounding_box_format),
            num_parallel_calls=self.autotune,
        )

        if split == "train":
            dataset = dataset.shuffle(batch_size * 4)

        # Adjusted padded_shapes for flexibility in the number of boxes and labels
        padded_shapes = {
            "images": [416, 416, 3],  # Assuming images can be of variable size
            "bounding_boxes": {
                "boxes": [None, 4],
                "classes": [None],
            },  # Allow any number of boxes
        }
        padding_values = {
            "images": 0.0,  # Assuming images are normalized to [0, 1]
            "bounding_boxes": {"boxes": 0.0, "classes": tf.constant(0, dtype=tf.int64)},
        }

        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=True,
        )

        dataset = dataset.prefetch(self.autotune)
        return dataset

    def load_dataset_images_labels(
        self,
        split: Literal["train", "test", "valid"],
        batch_size: int,
        num_classes: int,
    ) -> tf.data.Dataset:
        """
        Prepares and returns a DataLoader for a given subset of the dataset, including only images and their one-hot encoded labels.

        Parameters:
            split (str): The subset for which to prepare the DataLoader ('train', 'valid', 'test').
            batch_size (int): The size of the batches to return.
            num_classes (int): The number of distinct classes.

        Returns:
            tf.data.Dataset: A DataLoader object ready for model training or evaluation, yielding images and one-hot encoded labels.
        """
        tfrecord_path = f"{self.dataset_dir}/{split}/Sea-Vessels.tfrecord"
        dataset = tf.data.TFRecordDataset(tfrecord_path)

        def parse_tfrecord_fn_images_labels(example: tf.Tensor) -> Dict[str, tf.Tensor]:
            # Feature description for TFRecord example parsing
            feature_description = {
                "image/encoded": tf.io.FixedLenFeature([], tf.string),
                "image/object/class/label": tf.io.VarLenFeature(tf.int64),
            }
            parsed_example = tf.io.parse_single_example(example, feature_description)

            # Decode the image
            image = tf.image.decode_jpeg(parsed_example["image/encoded"], channels=3)
            image = (
                tf.cast(image, tf.float32) / 255.0
            )  # Normalize image pixels to [0, 1]

            # Convert labels to dense and one-hot-encoded vectors
            labels = tf.sparse.to_dense(parsed_example["image/object/class/label"])
            one_hot_labels = tf.reduce_max(
                tf.one_hot(labels, depth=num_classes), axis=0
            )

            return image, one_hot_labels

        dataset = dataset.map(
            parse_tfrecord_fn_images_labels, num_parallel_calls=self.autotune
        )

        if split == "train":
            dataset = dataset.shuffle(batch_size * 4)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(self.autotune)

        return dataset
