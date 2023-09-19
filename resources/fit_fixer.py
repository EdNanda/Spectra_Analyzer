import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QSpinBox, QHBoxLayout, \
    QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class CurveEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selected_point: int | None = None
        self.selected_dataset: int = 0
        self.adjacent_points: int = 3  # Default number of adjacent points
        self.initUI()

    def initUI(self):
        self.canvas = FigureCanvas(Figure())
        self.ax = self.canvas.figure.add_subplot(111)
        self.canvas.mpl_connect('button_press_event', self.mouse_press_event)
        self.canvas.mpl_connect('button_release_event', self.mouse_release_event)
        self.canvas.mpl_connect('motion_notify_event', self.mouse_move_event)

        self.load_data()
        self.plot_data()

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        controls_layout = QHBoxLayout()
        self.dataset_selector = QComboBox()
        for i in range(len(self.y_data_sets)):
            self.dataset_selector.addItem(f'Dataset {i + 1}')
        self.dataset_selector.currentIndexChanged.connect(self.change_dataset)
        controls_layout.addWidget(self.dataset_selector)

        spin_box = QSpinBox()
        spin_box.setValue(self.adjacent_points)
        spin_box.setMaximum(len(self.x_data) - 1)
        spin_box.valueChanged.connect(self.set_adjacent_points)
        controls_layout.addWidget(spin_box)

        save_button = QPushButton('Save Curves')
        save_button.clicked.connect(self.save_curves)
        controls_layout.addWidget(save_button)
        layout.addLayout(controls_layout)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def load_data(self):
        self.x_data = list(range(10))
        # Example: 3 datasets that share the same x-axis
        self.y_data_sets = [
            [i ** 2 for i in self.x_data],
            [i ** 1.5 for i in self.x_data],
            [-i ** 2 for i in self.x_data]
        ]

    def plot_data(self):
        self.ax.clear()
        for y_data in self.y_data_sets:
            self.ax.plot(self.x_data, y_data, marker='o', linestyle='-')
        self.canvas.draw()

    def save_curves(self):
        # Save the modified curves (e.g., to a file)
        pass

    def set_adjacent_points(self, value: int):
        self.adjacent_points = value

    def change_dataset(self, index: int):
        self.selected_dataset = index

    def mouse_press_event(self, event):
        if event.inaxes is not self.ax:
            return
        distances = np.sqrt(
            (self.x_data - event.xdata) ** 2 + (self.y_data_sets[self.selected_dataset] - event.ydata) ** 2)
        if np.min(distances) < 0.5:
            self.selected_point = np.argmin(distances)

    def mouse_move_event(self, event):
        if self.selected_point is None or event.inaxes is not self.ax:
            return
        dy = event.ydata - self.y_data_sets[self.selected_dataset][self.selected_point]
        for i in range(-self.adjacent_points, self.adjacent_points + 1):
            index = self.selected_point + i
            if 0 <= index < len(self.x_data):
                factor = (1 - abs(i) / (self.adjacent_points + 1))
                self.y_data_sets[self.selected_dataset][index] += dy * factor
        self.plot_data()

    def mouse_release_event(self, event):
        self.selected_point = None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CurveEditor()
    ex.show()
    sys.exit(app.exec_())