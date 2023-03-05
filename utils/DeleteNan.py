from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
from PyQt5.QtWebEngineWidgets import QWebEngineView
import os

from utils import Constants
from utils.PlotGraphics import Plots


class DeleteNanWindow(QtWidgets.QMainWindow):

    submitClicked = QtCore.pyqtSignal(pd.DataFrame)

    def __init__(self, data):
        super().__init__()

        self.data = data

        self.setWindowTitle("Delete NaN values")
        self.resize(400, 300)

        self.centralwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.setCentralWidget(self.centralwidget)

        self.web = QWebEngineView(self.centralwidget)

        self.show_button = QtWidgets.QPushButton(self.centralwidget)
        self.show_button.setText("Распределение NaN значений")
        self.setStyleSheet(Constants.SELECTALL_BUTTON)
        self.show_button.setGeometry(30, 30, 300, 30)
        self.show_button.clicked.connect(self.showDistributionNanValues)

        self.softDelete = QtWidgets.QLabel(self.centralwidget)
        self.softDelete.setText("<b>Удалить всю строку, если есть хотя бы одно значение NaN")
        self.softDelete.setGeometry(10, 100, 300, 30)
        self.softDelete.setWordWrap(True)

        self.softDelete = QtWidgets.QLabel(self.centralwidget)
        self.softDelete.setText("<b>Удалить всю строку, если все значения в строке NaN")
        self.softDelete.setGeometry(10, 180, 300, 30)
        self.softDelete.setWordWrap(True)

        self.hardCheckBox = QtWidgets.QCheckBox(self.centralwidget)
        self.hardCheckBox.setGeometry(350, 100, 40, 40)

        self.softCheckBox = QtWidgets.QCheckBox(self.centralwidget)
        self.softCheckBox.setGeometry(350, 180, 40, 40)
        self.softCheckBox.setChecked(True)

        self.buttonApplyChanges = QtWidgets.QPushButton(self.centralwidget)
        self.buttonApplyChanges.setText("Применить изменения")
        self.buttonApplyChanges.setStyleSheet(Constants.UPDATE_TABLE_BUTTON_STYLE)
        self.buttonApplyChanges.setGeometry(180, 240, 200, 30)
        self.buttonApplyChanges.clicked.connect(self.applyChanges)

    def applyChanges(self):
        if self.data is not None:
            if self.hardCheckBox.isChecked():
                self.data = self.data.dropna(how="any")
            if self.softCheckBox.isChecked():
                self.data = self.data.dropna(how="all")
            self.data = self.data.reset_index(drop=True)
            self.submitClicked.emit(self.data)
            QtWidgets.QMessageBox.about(self, "INFO", "Изменения применены!")
        else:
            QtWidgets.QMessageBox.about(self, "ERROR", "Ошибка!")

    def showDistributionNanValues(self):
        if self.data is not None:
            Plots.plotDistributionNanValues(self.data)
            file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Nan.html"))
            self.web.load(QtCore.QUrl.fromLocalFile(file_path))
            self.web.show()
        else:
            QtWidgets.QMessageBox.about(self, "ERROR", "Ошибка!")

    def displayInfo(self):
        self.show()
