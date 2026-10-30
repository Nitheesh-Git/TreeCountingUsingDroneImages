Example images - Tree Counting Using Drone Images (Tree-Top-View, 416x416 NEON aerial tiles)

1_training_set/          3 raw images from the 594-image training subset used for YOLOv9t/YOLOv12n
2_test_dataset/          3 raw images from the unseen 55-image test split
3_labelled_ground_truth/ the 3 training images with ground-truth tree boxes (green), shown at 2x
4_model_predictions/     the 3 test images: ground truth (green) vs YOLOv9t and YOLOv12n
                         detections (orange, conf 0.25), shown at 2x

Images were chosen with a fixed seed (0): one sparse, one medium and one dense image,
each from a different NEON site. File names give the site and the ground-truth tree count.
Regenerate with: python "External Validation v9 and v12/common/export_examples.py"
