import sys
import os
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QSlider, QPushButton, QLineEdit
from PyQt5.QtCore import QRegularExpression
from PyQt5.QtGui import QPixmap, QRegularExpressionValidator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from scipy.signal import butter, filtfilt

ico = os.path.join(sys._MEIPASS, "ico.ico") if getattr(sys, 'frozen', False) else "ico.ico"
png = os.path.join(sys._MEIPASS, "scale_1200.png") if getattr(sys, 'frozen', False) else "scale_1200.png"
app1 = QtWidgets.QApplication(sys.argv)
app1.setWindowIcon(QtGui.QIcon(ico))

class SuperheterodyneReceiver(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Супергетеродинный Приёмник")
        self.setGeometry(100, 100, 1200, 600)  # Increased width for accommodating frequency plots
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.signal_frequency_label = QLabel("Частота сигнала")
        self.layout.addWidget(self.signal_frequency_label)
        self.signal_frequency_slider = QLineEdit()
        self.regular2 = QRegularExpressionValidator(QRegularExpression("^(?:[1-9]|[12]\d|30)$"))
        self.signal_frequency_slider.setValidator(self.regular2)
        self.signal_frequency_slider.setText('15')
        self.signal_frequency_slider.textChanged.connect(self.plot_signal)
        self.layout.addWidget(self.signal_frequency_slider)
        self.heterodyne_frequency_label = QLabel("Частота гетеродина")
        self.layout.addWidget(self.heterodyne_frequency_label)
        self.heterodyne_frequency_slider = QLineEdit()
        self.regular = QRegularExpressionValidator(QRegularExpression("^(?:6[0-9]|7[0-9]|8[0-9]|90|100|110|120)$"))
        self.heterodyne_frequency_slider.setValidator(self.regular)
        self.heterodyne_frequency_slider.setText('90')
        self.heterodyne_frequency_slider.textChanged.connect(self.plot_signal)
        self.layout.addWidget(self.heterodyne_frequency_slider)
        self.noise_amplitude_label = QLabel("Амплитуда шума")
        self.layout.addWidget(self.noise_amplitude_label)
        self.noise_amplitude_slider = QLineEdit()
        self.regular1 = QRegularExpressionValidator(QRegularExpression("^(2[0-9]|3[0-9]|40|50|60)$"))
        self.noise_amplitude_slider.setValidator(self.regular1)
        self.noise_amplitude_slider.setText('20')
        self.noise_amplitude_slider.textChanged.connect(self.plot_signal)
        self.layout.addWidget(self.noise_amplitude_slider)
        self.intermediate_frequency_label = QLabel("Промежуточная частота")
        self.layout.addWidget(self.intermediate_frequency_label)
        self.open_img = QPushButton('Показать схему супергетеродинного приёмника')
        self.open_img.clicked.connect(self.show_img)
        self.layout.addWidget(self.open_img)        
        self.intermediate_frequency_value = QLabel()
        self.layout.addWidget(self.intermediate_frequency_value)
        self.canvas_signal = FigureCanvas(plt.Figure())
        self.layout.addWidget(self.canvas_signal)
        self.toolbar_signal = NavigationToolbar(self.canvas_signal, self)
        self.layout.addWidget(self.toolbar_signal)
        self.canvas_mixed_signal = FigureCanvas(plt.Figure())
        self.layout.addWidget(self.canvas_mixed_signal)
        self.toolbar_mixed_signal = NavigationToolbar(self.canvas_mixed_signal, self)
        self.layout.addWidget(self.toolbar_mixed_signal)
        self.canvas_filtered_signal = FigureCanvas(plt.Figure())
        self.layout.addWidget(self.canvas_filtered_signal)
        self.toolbar_filtered_signal = NavigationToolbar(self.canvas_filtered_signal, self)
        self.layout.addWidget(self.toolbar_filtered_signal)
        self.plot_signal()

    def plot_signal(self):
        if self.signal_frequency_slider.text() == '':
            return
        elif self.heterodyne_frequency_slider.text() == '':
            return
        elif self.noise_amplitude_slider.text() == '':
            return
        else:
            signal_freq = int(self.signal_frequency_slider.text())
            heterodyne_freq = int(self.heterodyne_frequency_slider.text())
            noise_amplitude = int(self.noise_amplitude_slider.text()) / 100.0
            intermediate_freq = np.abs(signal_freq - heterodyne_freq) 
            time = np.linspace(0, 1, 500)
            signal = np.sin(2 * np.pi * signal_freq * time)
            heterodyne = np.sin(2 * np.pi * heterodyne_freq * time)
            mixed_signal = signal * heterodyne
            #noise = np.random.normal(0, noise_amplitude, len(time))
            noise = noise_amplitude * np.sin(signal_freq * len(time))
            mixed_signal_with_noise = mixed_signal + noise
            filtered_signal = self.filter_signal(mixed_signal_with_noise)
            self.clear_subplots()
            self.plot_time_domain_signals(time, signal, mixed_signal_with_noise, filtered_signal)
            self.plot_frequency_domain_signals(signal, mixed_signal_with_noise, filtered_signal)
            self.draw_canvases(intermediate_freq)

    def clear_subplots(self):
        self.canvas_signal.figure.clear()
        self.canvas_mixed_signal.figure.clear()
        self.canvas_filtered_signal.figure.clear()

    def plot_time_domain_signals(self, time, signal, mixed_signal_with_noise, filtered_signal):
        ax1 = self.canvas_signal.figure.add_subplot(121)
        ax2 = self.canvas_mixed_signal.figure.add_subplot(121)
        ax3 = self.canvas_filtered_signal.figure.add_subplot(121)
        ax1.plot(time, signal)
        ax2.plot(time, mixed_signal_with_noise)
        ax3.plot(time, filtered_signal)
        self.set_common_properties(ax1, 'Исходный сигнал')
        self.set_common_properties(ax2, 'Смешанный сигнал с шумом')
        self.set_common_properties(ax3, 'Отфильтрованный сигнал')

    def plot_frequency_domain_signals(self, signal, mixed_signal_with_noise, filtered_signal):
        ax4 = self.canvas_signal.figure.add_subplot(122)
        ax5 = self.canvas_mixed_signal.figure.add_subplot(122)
        ax6 = self.canvas_filtered_signal.figure.add_subplot(122)
        sin_f_signal = np.sin(signal)
        sin_mis = np.sin(mixed_signal_with_noise)
        sin_mis_filtered = np.sin(filtered_signal)
        ax4.magnitude_spectrum(sin_f_signal)
        ax5.magnitude_spectrum(sin_mis)
        ax6.magnitude_spectrum(sin_mis_filtered)
        self.set_common_properties(ax4, 'Спектр исходного сигнала')
        self.set_common_properties(ax5, 'Спектр смешанного сигнала с шумом')
        self.set_common_properties(ax6, 'Спектр отфильтрованного сигнала')

    def set_common_properties(self, ax, title):
        ax.set_xlabel('Частота (Гц)' if title.startswith('Спектр') else 'Время')
        ax.set_ylabel('Амплитуда' if title.startswith('Спектр') else 'Амплитуда')
        ax.set_title(title)

        ax.grid(True)

    def draw_canvases(self, intermediate_freq):
        self.canvas_signal.draw()
        self.canvas_mixed_signal.draw()
        self.canvas_filtered_signal.draw()
        intermediate_freq_str = "{:.2f}".format(intermediate_freq)
        self.intermediate_frequency_value.setText(intermediate_freq_str)

    def filter_signal(self, signal):
        nyquist_freq = 0.5
        cutoff_freq = 0.2
        order = 2
        normal_cutoff = cutoff_freq / nyquist_freq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    
    def show_img(self):
        self.image_window = QWidget()
        self.image_window.setWindowTitle("Cхема супергетеродинного приёмника")
        image_label = QLabel(self.image_window)
        pixmap = QPixmap(png)  
        image_label.setPixmap(pixmap)
        layout = QVBoxLayout()
        layout.addWidget(image_label)
        self.image_window.setLayout(layout)
        self.image_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SuperheterodyneReceiver()
    window.showMaximized()
    sys.exit(app.exec_())
