from typing import Any, Dict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class TFRecordPlots:
    feature_description = {"image/encoded": tf.io.FixedLenFeature([], tf.string)}

    @staticmethod
    def _parse_function(example_proto: tf.Tensor) -> Dict[str, tf.Tensor]:
        return tf.io.parse_single_example(
            example_proto, TFRecordPlots.feature_description
        )

    @staticmethod
    def display_random_image_from_tfrecord(tfrecord_path: str) -> None:
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        parsed_dataset = raw_dataset.map(TFRecordPlots._parse_function)

        for raw_record in parsed_dataset.shuffle(buffer_size=1000).take(1):
            image = tf.image.decode_jpeg(raw_record["image/encoded"].numpy())
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis("off")
            plt.show()

    @staticmethod
    def show_batch_with_bboxes(
        batch_data: Dict[str, Any],
        label_map: Dict[int, str],  # Assuming label_map is {int: str}
    ) -> None:
        """
        Display a batch of images with bounding boxes and labels.

        Parameters:
        - batch_data: A batch of data containing 'images' and 'bounding_boxes'.
        - label_map: A dictionary mapping label IDs to label names.
        """
        if isinstance(batch_data, tuple):
            image_batch = batch_data[0]
            bounding_boxes_batch = batch_data[1]
        else:
            image_batch = batch_data["images"]
            bounding_boxes_batch = batch_data["bounding_boxes"]

        # Use nrows() for RaggedTensor to find the batch size
        # batch_size = image_batch.nrows()
        batch_size = image_batch.shape[0]

        plt.figure(figsize=(15, 15))
        for n in range(min(batch_size, 16)):  # Display up to 16 images from the batch
            ax = plt.subplot(4, 4, n + 1)
            img = (image_batch[n].numpy() * 255).astype("uint8")
            plt.imshow(img)
            plt.axis("off")

            if isinstance(batch_data, tuple):
                bboxes = bounding_boxes_batch[0][n].numpy()
                labels = bounding_boxes_batch[1][n].numpy()
            else:
                bboxes = bounding_boxes_batch["boxes"][n].numpy()
                labels = bounding_boxes_batch["classes"][n].numpy()

            for label, bbox in zip(labels, bboxes):
                ymin, xmin, ymax, xmax = bbox
                rect = patches.Rectangle(
                    (xmin * img.shape[1], ymin * img.shape[0]),
                    (xmax - xmin) * img.shape[1],
                    (ymax - ymin) * img.shape[0],
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)
                plt.text(
                    xmin * img.shape[1],
                    ymin * img.shape[0] - 2,
                    label_map.get(label, ""),
                    bbox=dict(facecolor="red", alpha=0.5),
                    fontsize=10,
                    color="white",
                )
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_batch_predictions(
        image_batch: tf.Tensor,
        y_pred: Dict[str, np.ndarray],
        label_map: Dict[int, str],
    ) -> None:
        """
        Plot the predicted bounding boxes and labels for a batch of images.

        Parameters:
        - image_batch: A batch of images from the dataset.
        - y_pred: A dictionary containing the predicted bounding boxes, confidence scores, classes, and number of detections.
        - label_map: A dictionary mapping label IDs to label names.
        """
        plt.figure(figsize=(12, 12))
        num_images = len(image_batch)
        num_rows = int(np.ceil(np.sqrt(num_images)))
        num_cols = int(np.ceil(num_images / num_rows))

        for i in range(num_images):
            ax = plt.subplot(num_rows, num_cols, i + 1)
            img = (image_batch[i].numpy() * 255).astype("uint8")
            plt.imshow(img)
            plt.axis("off")

            boxes = y_pred["boxes"][i]
            confidence = y_pred["confidence"][i]
            classes = y_pred["classes"][i]
            num_detections = y_pred["num_detections"][i]

            for j in range(num_detections):
                if confidence[j] > 0:
                    ymin, xmin, ymax, xmax = boxes[j]
                    label = label_map.get(classes[j], "Unknown")
                    score = confidence[j]

                    if xmin >= 0 and ymin >= 0:  # Check if coordinates are non-negative
                        rect = patches.Rectangle(
                            (xmin, ymin),
                            xmax - xmin,
                            ymax - ymin,
                            linewidth=2,
                            edgecolor="r",
                            facecolor="none",
                        )
                        ax.add_patch(rect)
                        ax.text(
                            xmin,
                            ymin - 2,
                            f"{label}: {score:.2f}",
                            bbox=dict(facecolor="red", alpha=0.5),
                            fontsize=10,
                            color="white",
                        )

        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_instance_with_predictions(
        model,
        image: tf.Tensor,
        annotations: Dict[str, tf.Tensor],
        label_map: Dict[int, str],
        threshold: float = 0.5,
    ) -> None:
        """
        Display a single image with predicted segments, bounding boxes, and labels.

        Parameters:
        - model: The trained model.
        - image: A single image from the dataset.
        - annotations: Annotations for the image, including 'labels' and 'bboxes'.
        - label_map: A dictionary mapping label IDs to label names.
        - threshold: The threshold for considering a prediction as positive.
        """
        plt.figure(figsize=(10, 10))

        # Make predictions
        classification_output, segmentation_output = model.predict(
            tf.expand_dims(image, axis=0)
        )
        predicted_labels = np.where(classification_output[0] > threshold)[0]
        predicted_segments = segmentation_output[0] > threshold

        # Display the image
        ax = plt.subplot(1, 2, 1)
        plt.imshow(image.numpy().astype("uint8"))
        plt.axis("off")

        # Display the true bounding boxes and labels
        labels = annotations["labels"].numpy()
        bboxes = annotations["bboxes"].numpy()
        for label, bbox in zip(labels, bboxes):
            ymin, xmin, ymax, xmax = bbox
            rect = patches.Rectangle(
                (xmin * image.shape[1], ymin * image.shape[0]),
                (xmax - xmin) * image.shape[1],
                (ymax - ymin) * image.shape[0],
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
            plt.text(
                xmin * image.shape[1],
                ymin * image.shape[0] - 2,
                label_map.get(label, "Unknown"),
                bbox=dict(facecolor="red", alpha=0.5),
                fontsize=10,
                color="white",
            )

        # Display the predicted segments
        ax = plt.subplot(1, 2, 2)
        plt.imshow(predicted_segments[:, :, 0], cmap="gray")
        plt.axis("off")

        # Display the predicted labels
        plt.title(
            f"Predicted Labels: {', '.join([label_map.get(label, 'Unknown') for label in predicted_labels])}"
        )

        plt.tight_layout()
        plt.show()
