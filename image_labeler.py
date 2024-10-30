import sys
import os
import numpy as np
import csv
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QLabel
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.patches as patches
from skimage.metrics import structural_similarity as ssim
from image_similarity_measures.quality_metrics import fsim

patch_size = 64
patch_half = patch_size // 2

class ImageLabeler(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.folder_path = ""
        self.files = []
        self.current_index = 0

    def initUI(self):
        self.setWindowTitle("Image Labeler")
        self.setGeometry(100, 100, 700, 600)
        self.setStyleSheet("""
            QMainWindow {background-color: #ECF0F1;}
            QPushButton { 
                font-size: 16px; 
                padding: 10px; 
                color: white; 
                background-color: #5DADE2; 
                border-radius: 8px;
                border: 2px solid #3949AB;
            }
            QPushButton:hover {
                background-color: #3949AB;
            }
            QLabel {
                font-size: 24px; 
                color: #17202A;
                font-weight: bold;
                padding: 20px;
            }
            QLabel#subtitle {
            font-size: 18px;
            font-weight: normal;
            font-style: italic;
            color: #626567;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setAlignment(Qt.AlignCenter)

        title_label = QLabel("IQT Image Labeler (UCL)")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont('Arial', 24, QFont.Bold))

        # Create subtitle label
        subtitle_label = QLabel("Developed by S. Kim. Only supports .npy files. Features that do not exist in the ground truth, which can lead to different clinical interpretations, are considered hallucinations.")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setFont(QFont('Arial', 14, QFont.StyleItalic))
        subtitle_label.setObjectName("subtitle")  # Correctly reference the subtitle QLabel for styling

        btn_load_folder = QPushButton('Load Folder', self)
        btn_load_folder.setIcon(QIcon('folder_icon.png'))  # Ensure you have an icon in the same directory or provide the path
        btn_load_folder.clicked.connect(self.load_folder)  # Ensure connection to load_folder method

        # Matplotlib Figures
        self.figure = Figure(edgecolor='k', facecolor='#D1F2EB')
        self.canvas = FigureCanvas(self.figure)
        self.ax1 = self.figure.add_subplot(141)
        self.ax2 = self.figure.add_subplot(142)
        self.ax3 = self.figure.add_subplot(143)
        self.ax4 = self.figure.add_subplot(144)

        # Horizontal layout for the buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        button_layout.setAlignment(Qt.AlignCenter)

        self.btn_label_hallucination1 = QPushButton('Mild Hallucination', self)
        self.btn_label_hallucination2 = QPushButton('Severe Hallucination', self)
        # self.btn_label_hallucination3 = QPushButton('Texture Hallucination', self)
        # self.btn_label_hallucination4 = QPushButton('Artifact Hallucination', self)
        self.btn_label_normal = QPushButton('Normal', self)
        self.btn_label_hallucination1.clicked.connect(lambda: self.save_label('mild_hallucination'))
        self.btn_label_hallucination2.clicked.connect(lambda: self.save_label('severe_hallucination'))
        # self.btn_label_hallucination3.clicked.connect(lambda: self.save_label('texture_hallucination'))
        # self.btn_label_hallucination4.clicked.connect(lambda: self.save_label('artifact_hallucination'))
        self.btn_label_normal.clicked.connect(lambda: self.save_label('normal'))

        # Add buttons to the horizontal layout
        button_layout.addWidget(self.btn_label_hallucination1)
        button_layout.addWidget(self.btn_label_hallucination2)
        # button_layout.addWidget(self.btn_label_hallucination3)
        # button_layout.addWidget(self.btn_label_hallucination4)
        button_layout.addWidget(self.btn_label_normal)

        # Adding widgets and layouts to main layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)
        main_layout.addWidget(btn_load_folder)
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(button_layout)
        # main_layout.addWidget(self.btn_label_hallucination1)
        # main_layout.addWidget(self.btn_label_hallucination2)
        # main_layout.addWidget(self.btn_label_hallucination3)
        # main_layout.addWidget(self.btn_label_hallucination4)
        # main_layout.addWidget(self.btn_label_normal)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def load_folder(self):
        self.folder_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if self.folder_path:  # Check if a folder was selected
            self.files = [f for f in os.listdir(self.folder_path) if f.endswith('.npy') and 'gt' in f]
            self.files.sort()
            self.current_index = 0  # Reset index when a new folder is loaded
            self.show_images()

    def show_images(self):
        if self.current_index < len(self.files):
            
            # if the file is already labeled in the csv file, skip it
            if os.path.exists(os.path.join(self.folder_path, 'labels.csv')):
                with open(os.path.join(self.folder_path, 'labels.csv'), 'r') as file:
                    reader = csv.reader(file)
                    labeled_files = [row[0] for row in reader]
                    while self.files[self.current_index] in labeled_files:
                        self.current_index += 1
                        if self.current_index >= len(self.files):
                            break
            
            gt_file = os.path.join(self.folder_path, self.files[self.current_index])
            pred_file = os.path.join(self.folder_path, self.files[self.current_index].replace('gt', 'pred'))

            gt_image = np.load(gt_file)
            pred_image = np.load(pred_file)

            gt_patch = gt_image
            pred_patch = pred_image

            gt_image = (gt_image - np.min(gt_image)) / (np.max(gt_image) - np.min(gt_image))
            pred_image = (pred_image - np.min(pred_image)) / (np.max(pred_image) - np.min(pred_image))

            gt_patch = (gt_patch - np.min(gt_patch)) / (np.max(gt_patch) - np.min(gt_patch))
            pred_patch = (pred_patch - np.min(pred_patch)) / (np.max(pred_patch) - np.min(pred_patch))

            # Calcuclate FSIM score
            self.ssim_score = ssim(gt_patch, pred_patch, data_range=1.0)
            self.fsim_score = fsim(np.expand_dims(gt_patch, axis=-1), np.expand_dims(pred_patch, axis=-1))

            # Clear previous figures
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()

            # Display images using Matplotlib
            self.ax1.imshow(gt_image, cmap='gray')
            # Create a rectangle patch to highlight the center of the image of size 16x16
            # rect = patches.Rectangle((gt_image.shape[1]//2 - patch_half, gt_image.shape[0]//2 - patch_half), patch_size, patch_size, linewidth=2, edgecolor='r', facecolor='none')
            # self.ax1.add_patch(rect)
            self.ax1.set_title('Ground Truth')
            self.ax1.axis('off')
            self.ax2.imshow(pred_image, cmap='gray')
            # Create a rectangle patch to highlight the center of the image of size 16x16
            # rect = patches.Rectangle((pred_image.shape[1]//2 - patch_half, pred_image.shape[0]//2 - patch_half), patch_size, patch_size, linewidth=2, edgecolor='r', facecolor='none')
            # self.ax2.add_patch(rect)
            self.ax2.set_title(f'Predicted -> SSIM: {self.ssim_score:.3f} FSIM: {self.fsim_score:.3f}')
            self.ax2.axis('off')
            
            # Display the difference image
            im1 = self.ax3.imshow(np.abs(gt_image - pred_image), cmap='gray')
            self.ax3.set_title('Difference')
            self.figure.colorbar(im1, ax=self.ax3)
            self.ax3.axis('off')
            
            # Display patch-wise FSIM score (16x16)
            self.patch_fsim_score = []
            fsim_patch_size = 16
            for i in range(0, gt_image.shape[0], fsim_patch_size):
                for j in range(0, gt_image.shape[1], fsim_patch_size):
                    gt_patch = gt_image[i:i+fsim_patch_size, j:j+fsim_patch_size]
                    pred_patch = pred_image[i:i+fsim_patch_size, j:j+fsim_patch_size]
                    self.patch_fsim_score.append(fsim(np.expand_dims(gt_patch, axis=-1), np.expand_dims(pred_patch, axis=-1)))
            self.patch_fsim_score = np.array(self.patch_fsim_score).reshape(gt_image.shape[0]//fsim_patch_size, gt_image.shape[1]//fsim_patch_size)
            print(self.patch_fsim_score.shape)
            #nearest interpolation
            im1 = self.ax4.imshow(pred_image, cmap='gray', extent=(0, pred_image.shape[1], pred_image.shape[0], 0))
            im2 = self.ax4.imshow(self.patch_fsim_score, cmap='jet', alpha=0.5, extent=(0, pred_image.shape[1], pred_image.shape[0], 0))
            self.figure.colorbar(im2, ax=self.ax4)
            self.ax4.set_title('Patch-wise FSIM')
            self.ax4.axis('off')

            # Refresh canvas
            self.canvas.draw()

            self.current_index += 1

    def save_label(self, label):
        if not os.path.exists(os.path.join(self.folder_path, 'labels.csv')):
            with open(os.path.join(self.folder_path, 'labels.csv'), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['filename', 'label', 'ssim_score'])
        # # else write to the existing file but skip already labeled images
        # with open(os.path.join(self.folder_path, 'labels.csv'), 'r') as file:
        #     reader = csv.reader(file)
        #     labeled_files = [row[0] for row in reader]
        #     if self.files[self.current_index-1] in labeled_files:
        #         return

        with open(os.path.join(self.folder_path, 'labels.csv'), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.files[self.current_index-1], label, self.ssim_score])

        self.show_images()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageLabeler()
    ex.show()
    sys.exit(app.exec_())
