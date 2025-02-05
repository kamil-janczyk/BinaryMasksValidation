# BinaryMasksValidation

Overview:

MaskValidation is a Python class for evaluating segmentation masks. It compares predicted masks with ground truth masks and computes various metrics such as Intersection over Union (IoU), Mean Average Precision (mAP), and proposed 1-pixel error metric.

Features:

Load Images: Reads ground truth and predicted masks from directories.

Preprocessing: Crops and pads images to a fixed size (256x256).

IoU Calculation: Computes the intersection-over-union metric.

mAP Calculation: Uses TorchMetrics to compute mean average precision for segmentation.

1-Pixel Error Analysis: Shows how slight errors in annotation can influence potential results by dilating ground truth masks.

Worst Samples Identification: Finds the worst-performing samples based on IoU.

Dependencies:

Used dependencies are in requirements.txt file.

Example of use:

    if __name__ == "__main__":
        # Paths to the ground truth and prediction folders
        gt_folder = "/path/to/ground_truth_images/folder/"
        pred_folder = "/path/to/predicted_images/folder/"
    
        # Initialize the MaskValidation class
        validator = MaskValidation()
         
        # Load the images from the specified folders
        # In method load_images you can change target image size
        validator.load_images(gt_folder, pred_folder)
        
        # Remember to attune iou_threshold for mAP 
        # Compute and display the metrics
        validator.metrics()
