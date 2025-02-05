import os
import numpy as np
import tensorflow as tf
import cv2
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class MaskValidation:
    def __init__(self):

        self.GT_list = []           #Ground Truth images
        self.prediction_list = []   #Predicted images
        self.filenames = []         #Filenames

    def load_images(self, GT_folder_path, prediction_folder_path):


        for filename in sorted(os.listdir(prediction_folder_path)):
            if filename.endswith('.png'):
                if  os.path.exists(os.path.join(GT_folder_path, filename)):
                    # loading predictions
                    prediction = np.array(cv2.imread(os.path.join(prediction_folder_path, filename)))
                    prediction = np.where(prediction[:, :, 1] >= 100, 1, 0)
                    prediction = self.crop_and_padd(prediction,(256, 256))
                    self.prediction_list.append(prediction)

                    # loading grand truth (GT)
                    GT = np.array(cv2.imread(os.path.join(GT_folder_path, filename)))
                    GT = np.where(GT[:, :, 1] >= 100, 1, 0)
                    GT = self.crop_and_padd(GT,(256, 256))
                    self.GT_list.append(GT)
                    self.filenames.append((filename))
                else:
                    print(f"For prediction:{filename}, grand truth does not exists ")

        if self.prediction_list is None:
            print(f"Warning: Unable to load images")
        else:
            print(f"{len(self.GT_list)} samples loaded")

    def crop_and_padd(self, image, dest_shape=(256, 256)):

        rows = image.shape[0]
        cols = image.shape[1]

        before_row = int((dest_shape[0] - rows) / 2)
        after_row = dest_shape[0] - rows - before_row
        before_col = int((dest_shape[1] - cols) / 2)
        after_col = dest_shape[1] - cols - before_col

        paddings = tf.constant([[before_row, after_row], [before_col, after_col]])

        image_padded = tf.pad(image, paddings, mode="CONSTANT", constant_values=0)

        return image_padded

    def calculate_map(self, iou_threshold):
        # https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html

        map_metric = MeanAveragePrecision(iou_thresholds=iou_threshold, iou_type="segm")
        predictions = []
        GT = []

        for pred_mask, gt_mask in zip(np.array(self.prediction_list), np.array(self.GT_list)):
            # Convert the 2D masks to bounding boxes and scores
            pred_mask_tensor = torch.tensor(pred_mask, dtype=torch.bool).unsqueeze(0)
            gt_mask_tensor = torch.tensor(gt_mask, dtype=torch.bool).unsqueeze(0)


            #A list consisting of dictionaries each containing the key-values
            # (each dictionary corresponds to a single image):#

            # Predictions for a single class
            predictions.append({
                "masks": pred_mask_tensor,  # Boolean masks
                "scores": torch.tensor([1.0]),
                "labels": torch.tensor([0], dtype=torch.int64),  # Single class (0)
            })

            # Ground truth for a single class
            GT.append({
                "masks": gt_mask_tensor,  # Boolean masks
                "labels": torch.tensor([0], dtype=torch.int64),  # Single class (0)
            })

        # Update the metric with the predictions and GT
        map_metric.update(predictions,GT)

        # Compute the mAP
        map_value = map_metric.compute()
        self.print_metrics(map_value['map'].item(), "mAP", f"IoU threshold {iou_threshold}")

    def compute_iou(self, GT, prediction):
        intersection = np.logical_and(GT, prediction)
        union = np.logical_or(GT, prediction)
        iou = np.sum(intersection) / np.sum(union)
        return iou

    def proces_GT_mask_for_1_pix_error(self):

        modified_masks = []
        for mask in (self.GT_list):
            if not isinstance(mask, np.ndarray):
                mask = mask.numpy()  # Convert TensorFlow tensor to NumPy array
            mask = mask.astype(np.uint8)  # Ensure it's uint8

            # Create a 3x3 kernel for dilation
            kernel = np.ones((3, 3), np.uint8)

            # Apply dilation to add 1 pixel in every direction
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)

            # Convert dilated mask back to binary if necessary (0/1)
            modified_masks.append((dilated_mask))

        return modified_masks

    def calculate_1_px_error(self):

        modified_ious = []
        modified_masks = self.proces_GT_mask_for_1_pix_error()

        for i in range(len(self.GT_list)):
            modified_ious.append(self.compute_iou(self.GT_list[i], modified_masks [i]))

        self.print_metrics(modified_ious,"1-pixel error", "1-pixel error - drop of IoU from 1.0 - potential error in ground truth "
                                                          "annotation: a 1-pixel wide border around entire mask.")

        return np.array(modified_ious)

    def print_metrics(self, metrics, metric_name= "IoU",comment=""):

        if isinstance(metrics, (list, np.ndarray)) and len(metrics) > 1:
            mean_value = np.mean(metrics)
            std_value = np.std(metrics)
            print(f"{metric_name} - Mean: {mean_value:.4f}, Std: {std_value:.4f}")
        else:
            print(f"{metric_name}: {metrics:.4f}")
        if comment:
            print("Comment:", comment)

    def worst_samples(self, metrix, metrix_name = ""):

        #metrix must be list - not a scalar
        worst_indices = np.argsort(metrix)[:10]
        worst_samples = [self.filenames[i] for i in worst_indices]
        worst_scores = metrix[worst_indices]
        print("Samples with the worst", metrix_name, " scores:")
        for name, score in zip(worst_samples, worst_scores):
            print(f"{name}: {score:.4f}")

    def metrix(self):

        iou = []
        for GT, prediction in zip(self.GT_list, self.prediction_list):
            iou.append(self.compute_iou(GT, prediction))

        self.print_metrics(iou, metric_name="IoU", comment="")

        #mAP with different thresholds
        iou_threshold = None
        self.calculate_map(iou_threshold)

        iou_threshold = [0.90]
        self.calculate_map(iou_threshold)

        #metric - 1pixel wide border error in annotating of image
        self.calculate_1_px_error()
    validator.metrix()



