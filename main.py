import os
import sys
import cv2
import torch
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QTextEdit, QHBoxLayout, QSplitter, QFrame
)
from PySide6.QtGui import QPixmap, QImage, QColor
from PySide6.QtCore import Qt
from torch import parse_ir
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry


def cv_to_pixmap(cv_img):
    height, width, channel = cv_img.shape
    bytes_per_line = 3 * width
    q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
    return QPixmap.fromImage(q_img)


def calculate_centroid(mask):
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy

def compute_iou(mask1, mask2):
    """Compute Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def is_in_list(mask, mask_list):
    return any(np.array_equal(mask, m) for m in mask_list)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Strip Compliance Checker")
        self.resize(1200, 800)

        self.yolo_model = YOLO("last.pt")
        self.sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth").cuda()
        self.predictor = SamPredictor(self.sam)

        self.sample_mask_data = []
        self.sample_image_display = None
        self.sample_image = None
        self.image_folder_paths = []
        self.current_image_index = 0

        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)

        self.left_panel = QVBoxLayout()

        self.load_sample_btn = QPushButton("Load Sample Image")
        self.load_sample_btn.clicked.connect(self.load_sample_image)
        self.left_panel.addWidget(self.load_sample_btn)

        self.load_folder_btn = QPushButton("Load Image Folder")
        self.load_folder_btn.clicked.connect(self.load_image_folder)
        self.left_panel.addWidget(self.load_folder_btn)

        self.next_image_btn = QPushButton("Next Image")
        self.next_image_btn.clicked.connect(self.process_next_image)
        self.next_image_btn.setEnabled(False)
        self.left_panel.addWidget(self.next_image_btn)

        self.left_panel.addStretch()

        self.log_panel = QTextEdit()
        self.log_panel.setFixedHeight(120)
        self.log_panel.setReadOnly(True)

        self.viewer_panel = QVBoxLayout()
        self.image_label = QLabel("Top View")
        self.result_label = QLabel("Bottom View")
        for label in [self.image_label, self.result_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.viewer_panel.addWidget(self.image_label)
        self.viewer_panel.addWidget(self.result_label)

        splitter = QSplitter(Qt.Vertical)
        viewer_container = QWidget()
        viewer_container.setLayout(self.viewer_panel)
        splitter.addWidget(viewer_container)
        log_container = QWidget()
        log_container.setLayout(QVBoxLayout())
        log_container.layout().addWidget(self.log_panel)
        splitter.addWidget(log_container)

        layout.addLayout(self.left_panel, 1)
        layout.addWidget(splitter, 4)

    def log(self, text):
        self.log_panel.append(text)

    def load_sample_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Sample Image", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return

        img, masks = self.process_image(path)
        self.sample_mask_data = masks
        self.sample_image_display = img.copy()
        self.sample_image = img.copy()

        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(self.sample_image_display, contours, -1, (0, 255, 255), 2)

        self.image_label.setPixmap(cv_to_pixmap(self.sample_image_display))
        self.result_label.clear()
        self.log(f"Loaded sample image with {len(masks)} masks")

    def load_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        self.image_folder_paths = [os.path.join(folder, f) for f in os.listdir(folder)
                                   if f.lower().endswith(('.jpg', '.png'))]
        self.current_image_index = 0
        self.next_image_btn.setEnabled(True)
        self.process_next_image()

    def process_next_image(self):
        if self.current_image_index >= len(self.image_folder_paths):
            self.log("All images processed.")
            self.next_image_btn.setEnabled(False)
            return

        path = self.image_folder_paths[self.current_image_index]
        img, masks = self.process_image(path)

        print(len(self.sample_mask_data), len(masks))
        matched, missing, extra = self.compare_by_shape(self.sample_mask_data, masks, img)

        annotated = img.copy()
        msg = []
        if missing:
            msg.append(f"Missing: {len(missing)}")
        if extra:
            msg.append(f"Extra: {len(extra)}")
        if not missing and not extra:
            msg.append("All matched ✓")

        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = [0, 255, 0] if is_in_list(mask, matched) else [0, 0, 255]
            cv2.drawContours(annotated, contours, -1, color, 2)

        self.result_label.setPixmap(cv_to_pixmap(annotated))
        self.image_label.setPixmap(cv_to_pixmap(self.sample_image_display))
        self.log(f"Processed {os.path.basename(path)} | {' | '.join(msg)}")

        self.current_image_index += 1

    def process_image(self, path):
        img_bgr = cv2.imread(path)[1300:2650, 1200:2824]
        img_bgr = cv2.resize(img_bgr, (512, 384))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.yolo_model.predict(source=img_rgb, imgsz=img_rgb.shape[:2], iou=0.1, conf=0.25, verbose=False)[0]

        masks = []
        if results.masks is not None:
            self.predictor.set_image(img_rgb)
            for mask_tensor, box in zip(results.masks.data, results.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box.tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                input_point = np.array([[cx, cy]])
                input_label = np.array([1])
                sam_masks, scores, _ = self.predictor.predict(point_coords=input_point, point_labels=input_label,
                                                               multimask_output=False)
                refined = sam_masks[0].astype(np.uint8) * 255
                # masks.append(refined)
                # Check IoU with existing masks
                is_duplicate = False
                for existing in masks:
                    iou = compute_iou(refined, existing)
                    if iou > 0.9:  # IoU threshold
                        is_duplicate = True
                        break

                if not is_duplicate:
                    masks.append(refined)

        return img_bgr, masks


    def compare_by_shape(self, sample_masks, target_masks, img, shape_thresh=0.15, color_thresh=65):
        matched = []
        missing = []
        extra_indices = set(range(len(target_masks)))

        def get_largest_contour(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return max(contours, key=cv2.contourArea) if contours else None

        def mean_color(mask, image):
            """Return mean BGR color inside the binary mask."""
            if mask.sum() == 0:
                return np.array([0, 0, 0])
            return cv2.mean(image, mask)[0:3]  # BGR

        sample_contours = [get_largest_contour(m) for m in sample_masks]
        target_contours = [get_largest_contour(m) for m in target_masks]

        for i, sc in enumerate(sample_contours):
            if sc is None:
                missing.append(sample_masks[i])
                continue

            found = False
            for j in list(extra_indices):
                tc = target_contours[j]
                if tc is None:
                    continue

                shape_score = cv2.matchShapes(sc, tc, cv2.CONTOURS_MATCH_I1, 0.0)
                print(f"Shape score (sample {i} vs target {j}): {shape_score}")

                if shape_score < shape_thresh:
                    # Check color similarity
                    sample_color = mean_color(sample_masks[i], self.sample_image)
                    target_color = mean_color(target_masks[j], img)
                    color_diff = np.linalg.norm(np.array(sample_color) - np.array(target_color))
                    print(f"Color diff (sample {i} vs target {j}): {color_diff}")

                    if color_diff < color_thresh:
                        matched.append(target_masks[j])
                        extra_indices.remove(j)
                        found = True
                        break  # Stop after first valid match

            if not found:
                missing.append(sample_masks[i])

        extra = [target_masks[j] for j in extra_indices]
        return matched, missing, extra

    @staticmethod
    def calculate_iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
