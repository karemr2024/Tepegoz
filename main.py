# from PIL import Image
# import PySpin
# import time
# import myspincam
import pulse_generator as pulser
# import os
import experiment_cmds
import sys
import PWM_Acquisition
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QMainWindow
from PyQt5.QtGui import QIcon

if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = PWM_Acquisition.Ui_Dialog()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
    # experiment_cmds.run_experiment_nogui(5000, "CHIP87_7ms_30FPS_1000IMG_BFS-U3-17S7M-C_Deltest", "CUSTOM", 0, 30, 7000)
    # experiment_cmds.run_experiment_nogui(500, "TRY2", "CUSTOM", 0, 40, 4000)
    # experiment_cmds.run_experiment_nogui(500, "TRY3", "CUSTOM", 0, 40, 4000)
    # experiment_cmds.run_experiment_nogui(500, "TRY4", "CUSTOM", 0, 40, 4000)
    # experiment_cmds.run_experiment_nogui(500, "TRY5", "CUSTOM", 0, 40, 4000)

