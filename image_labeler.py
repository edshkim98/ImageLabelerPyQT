import sys
import os
import numpy as np
import csv
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLabel
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.patches as patches

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
        subtitle_label = QLabel("Developed by S. Kim. Only supports .npy files.")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setFont(QFont('Arial', 14, QFont.StyleItalic))
        subtitle_label.setObjectName("subtitle")  # Correctly reference the subtitle QLabel for styling

        btn_load_folder = QPushButton('Load Folder', self)
        btn_load_folder.setIcon(QIcon('folder_icon.png'))  # Ensure you have an icon in the same directory or provide the path
        btn_load_folder.clicked.connect(self.load_folder)  # Ensure connection to load_folder method

        # Matplotlib Figures
        self.figure = Figure(edgecolor='k', facecolor='#D1F2EB')
        self.canvas = FigureCanvas(self.figure)
        self.ax1 = self.figure.add_subplot(121)
        self.ax2 = self.figure.add_subplot(122)

        self.btn_label_hallucination = QPushButton('Hallucination', self)
        self.btn_label_normal = QPushButton('Normal', self)
        self.btn_label_hallucination.clicked.connect(lambda: self.save_label('hallucination'))
        self.btn_label_normal.clicked.connect(lambda: self.save_label('normal'))

        # Adding widgets and layouts to main layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)
        main_layout.addWidget(btn_load_folder)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.btn_label_hallucination)
        main_layout.addWidget(self.btn_label_normal)

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
            gt_file = os.path.join(self.folder_path, self.files[self.current_index])
            pred_file = os.path.join(self.folder_path, self.files[self.current_index].replace('gt', 'pred'))

            gt_image = np.load(gt_file)
            pred_image = np.load(pred_file)

            # Clear previous figures
            self.ax1.clear()
            self.ax2.clear()

            # Display images using Matplotlib
            self.ax1.imshow(gt_image, cmap='gray')
            # Create a rectangle patch to highlight the center of the image of size 16x16
            rect = patches.Rectangle((gt_image.shape[1]//2 - 8, gt_image.shape[0]//2 - 8), 16, 16, linewidth=2, edgecolor='r', facecolor='none')
            self.ax1.add_patch(rect)
            self.ax1.set_title('Ground Truth')
            self.ax1.axis('off')
            self.ax2.imshow(pred_image, cmap='gray')
            # Create a rectangle patch to highlight the center of the image of size 16x16
            rect = patches.Rectangle((pred_image.shape[1]//2 - 8, pred_image.shape[0]//2 - 8), 16, 16, linewidth=2, edgecolor='r', facecolor='none')
            self.ax2.add_patch(rect)
            self.ax2.set_title('Predicted')
            self.ax2.axis('off')

            # Refresh canvas
            self.canvas.draw()

            self.current_index += 1

    def save_label(self, label):
        with open(os.path.join(self.folder_path, 'labels.csv'), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.files[self.current_index-1], label])
        self.show_images()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageLabeler()
    ex.show()
    sys.exit(app.exec_())
