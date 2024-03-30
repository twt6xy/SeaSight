from typing import Dict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
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
        image_batch: tf.Tensor,
        annotations: Dict[str, tf.Tensor],
        label_map: Dict[int, str],
    ) -> None:
        """
        Display a batch of images with bounding boxes and labels.

        Parameters:
        - image_batch: A batch of images from the dataset.
        - annotations: A batch of annotations, where each annotation includes 'labels' and 'bboxes'.
        - label_map: A dictionary mapping label IDs to label names.
        """
        plt.figure(figsize=(15, 15))
        for n in range(
            min(len(image_batch), 16)
        ):  # Display up to 16 images from the batch
            ax = plt.subplot(4, 4, n + 1)
            img = image_batch[n].numpy().astype("uint8")
            plt.imshow(img)
            plt.axis("off")

            labels = annotations["labels"][n].numpy()
            bboxes = annotations["bboxes"][n].numpy()

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
                    label_map.get(label, "Unknown"),
                    bbox=dict(facecolor="red", alpha=0.5),
                    fontsize=10,
                    color="white",
                )
        plt.tight_layout()
        plt.show()
