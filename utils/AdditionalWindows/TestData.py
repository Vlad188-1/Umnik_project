from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
# from PyQt5.QtWebEngineWidgets import QWebEngineView
import yaml
from pathlib import Path

import torch
# import cv2
from sklearn.metrics import classification_report

# My imports
from src import NN, AutoEncoder
from utils.plots import plot_test_markup, plot_curve_testing
from utils import Constants


class TestDataWindow(QtWidgets.QMainWindow):
    # submitClicked = QtCore.pyqtSignal(pd.DataFrame)

    def __init__(self, data):
        super().__init__()

        self.data: pd.DataFrame = data
        self.x_test: pd.DataFrame = pd.DataFrame({})
        self.y_test: pd.Series = pd.Series({})
        self.temp_dir = Path(__file__).parent / "temp_dir_191j1"
        self.config: dict = {}
        self.name_target_variable: str
        self.AE_model: AutoEncoder
        self.NN_model: NN

        self.setWindowTitle("Test data window")
        self.resize(900, 700)

        self.temp_dir.mkdir(exist_ok=True)
        # with tempfile.TemporaryDirectory(dir=str(Path(__file__).parent), prefix="temp_mask_") as temp:
        #     self.temp_dir = temp
        # print(__file__)
        # print(self.temp_dir)

        self.centralwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.setCentralWidget(self.centralwidget)

        # self.web = QWebEngineView(self.centralwidget)

        self.button_open_yaml = QtWidgets.QPushButton(self.centralwidget)
        self.button_open_yaml.setGeometry(10, 10, 250, 50)
        self.button_open_yaml.setText("Открыть config файл и веса модели")
        self.button_open_yaml.clicked.connect(self.openConfigFile)

        self.button_apply_yaml = QtWidgets.QPushButton(self.centralwidget)
        self.button_apply_yaml.setGeometry(10, 60, 250, 50)
        self.button_apply_yaml.setText("Применить config файл к данным")
        self.button_apply_yaml.clicked.connect(self.applyConfig)

        self.targetVariableLabel = QtWidgets.QLabel(self.centralwidget)
        self.targetVariableLabel.setText("<b>Целевая переменная</b>")
        self.targetVariableLabel.setGeometry(15, 110, 200, 50)
        self.comboboxSelectTargetVariable = QtWidgets.QComboBox(self.centralwidget)
        self.comboboxSelectTargetVariable.setGeometry(10, 140, 150, 50)

        self.depthLabel = QtWidgets.QLabel(self.centralwidget)
        self.depthLabel.setText("<b>Выбрать высоту</b>")
        self.depthLabel.setGeometry(15, 180, 250, 50)
        self.depthCombobox = QtWidgets.QComboBox(self.centralwidget)
        self.depthCombobox.setGeometry(10, 210, 150, 50)

        self.variableForMarkup = QtWidgets.QLabel(self.centralwidget)
        self.variableForMarkup.setText("<b>Выбрать переменную, по которой будет строиться разметка</b>")
        self.variableForMarkup.setWordWrap(True)
        self.variableForMarkup.setGeometry(15, 260, 250, 50)

        self.markupVariableCombobox = QtWidgets.QComboBox(self.centralwidget)
        self.markupVariableCombobox.setGeometry(10, 300, 150, 50)

        self.device_label = QtWidgets.QLabel(self.centralwidget)
        self.device_label.setText("<b>Вычислитель")
        self.device_label.setGeometry(10, 350, 200, 30)

        self.combobox_devices = QtWidgets.QComboBox(self.centralwidget)
        self.combobox_devices.setGeometry(110, 343, 100, 50)
        self.combobox_devices.addItems(["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
        self.combobox_devices.setCurrentIndex(0)

        self.testButton = QtWidgets.QPushButton(self.centralwidget)
        self.testButton.setGeometry(10, 400, 250, 30)
        self.testButton.setText("Начать тестирование")
        self.testButton.clicked.connect(self.startTest)
        self.testButton.setStyleSheet(Constants.UPDATE_TABLE_BUTTON_STYLE)

        self.testButtonMarkup = QtWidgets.QPushButton(self.centralwidget)
        self.testButtonMarkup.setGeometry(10, 450, 250, 30)
        self.testButtonMarkup.setText("Отобразить результаты разметки")
        self.testButtonMarkup.clicked.connect(self.showResult)
        self.testButtonMarkup.setStyleSheet(Constants.UPDATE_TABLE_BUTTON_STYLE)

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(300, 10, 400, 550)

        self.exit = QtWidgets.QAction("Exit Application", self) #shortcut=QtGui.QKeySequence("Ctrl+q"), triggered=lambda: self.exit_app)
        self.exit.triggered.connect(self.exit_app)
        self.addAction(self.exit)

        # image = cv2.imread("/Users/vladislavefremov/Disk/Vlad/Гугл диск/Диск/Диплом/Папка с кодами/Для написания в диплом/Umnik_project/projects/proba_sorted_features_2/train_val_curve.jpg")
        # image = cv2.resize(image, (550, 400), interpolation = cv2.INTER_AREA)
        # image = QtGui.QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3, QtGui.QImage.Format_BGR888)
        # self.im = QtGui.QPixmap(image)
        #
        # self.label = QtWidgets.QLabel(self.centralwidget)
        # self.label.setGeometry(300, 10, 550, 400)
        # self.label.setPixmap(self.im)

        # self.list_checkboxes = QtWidgets.QListWidget(self.centralwidget)
        # self.list_checkboxes.setDragDropMode(self.list_checkboxes.InternalMove)
        # self.list_checkboxes.setGeometry(10, 310, 200, 150)

    def exit_app(self):
        print("closed")
        self.temp_dir.rmdir()
        self.close()

    # Load config
    def openConfigFile(self):
        # filename, _ = QtWidgets.QFileDialog.getOpenFileName(filter="*.yml")
        project_path = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select Folder')
        all_files = sorted(list(Path(project_path).glob("*")))
        for file in all_files:
            if Path(file).suffix == ".yml":
                try:
                    with open(file, "r") as yml:
                        self.config = yaml.safe_load(yml)
                        print(self.config)
                except FileNotFoundError:
                    QtWidgets.QMessageBox.warning(self, "WARNING", "Файлы не загружены")
            elif file.suffix == ".pt":
                if "AE" in file.stem:
                    self.AE_model = torch.load(file)
                elif "NN" in file.stem:
                    self.NN_model = torch.load(file)
                else:
                    QtWidgets.QMessageBox.warning(self, "ERROR", "Неизвестное название модели в проекте!")
        if (self.config is not None) and (self.AE_model is not None) and (self.NN_model is not None):
            QtWidgets.QMessageBox.information(self, "SUCCESS", "Config подготовки данных, Модель автоэнкодера и полносвязной сети успешно загружены!")

    def applyConfig(self):
        if (self.data is not None) and (self.config is not None):
            for k in sorted(self.config.keys()):
                if k == "^2":
                    for feature in self.config[k]:
                        self.data[feature + "^2"] = self.data[feature] ** 2
                elif k == "^3":
                    for feature in self.config[k]:
                        self.data[feature + "^3"] = self.data[feature] ** 3
                elif k == "*":
                    for feature in self.config[k]:
                        if len(feature) == 2:
                            self.data["*".join(feature)] = self.data[feature[0]] * self.data[feature[1]]
                        elif len(feature) == 3:
                            self.data["*".join(feature)] = self.data[feature[0]] * self.data[feature[1]] * self.data[feature[2]]
                        elif len(feature) == 4:
                            self.data["*".join(feature)] = self.data[feature[0]] * self.data[feature[1]] * self.data[feature[2]] \
                                                      * self.data[feature[3]]
                elif k == "Target variable":
                    self.name_target_variable = self.config[k]

            self.comboboxSelectTargetVariable.addItems(self.data.columns)
            self.depthCombobox.addItems(self.data.columns)
            self.markupVariableCombobox.addItems(self.data.columns)

            self.data.dropna(inplace=True)
        else:
            QtWidgets.QMessageBox.about(self, "Error",
                                              "Не загружены данные или не выбран config подготовки данных!")

    def startTest(self):

        self.data.sort_index(axis=1, inplace=True)
        self.y_test = self.data[self.comboboxSelectTargetVariable.currentText()].values
        self.x_test = self.data.drop(
            [self.comboboxSelectTargetVariable.currentText(), self.depthCombobox.currentText()], axis=1)

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = self.combobox_devices.customEvent()

        if (self.x_test is not None) and (self.y_test is not None):
            self.x_test = torch.from_numpy(self.x_test.values).to(torch.float32)
            self.AE_model = self.AE_model.to(device)
            self.NN_model = self.NN_model.to(device)

            x_test_encoder = self.AE_model.encoder(self.x_test.to(device))
            # outputs = self.NN_model(x_test_encoder)
            y_pred = self.NN_model.predict(x_test_encoder).cpu().detach().numpy()
            # y_pred = torch.sigmoid(outputs).cpu().detach().numpy()

            classification_report(self.y_test, y_pred.round()) #, target_names=["no_oil", "oil"])

            print("Построение Precision-Recall кривой")
            plot_curve_testing(self.y_test, y_pred)
            plot_test_markup(self.data[[self.markupVariableCombobox.currentText(), self.depthCombobox.currentText()]],
                             self.y_test, y_pred,
                             depth_name=self.depthCombobox.currentText(),
                             variable_for_markup=self.markupVariableCombobox.currentText()
                             )

    def showResult(self):
        pass
        # def resize_image(image, window_height=500):
        #     aspect_ratio = float(image.shape[1]) / float(image.shape[0])
        #     window_width = window_height / aspect_ratio
        #     image = cv2.resize(image, (int(window_height), int(window_width)))
        #     return image

        # image = cv2.imread("/Users/vladislavefremov/Disk/Vlad/Гугл диск/Диск/Диплом/Папка с кодами/Для написания в диплом/Umnik_project/Markup.jpg")
        # image = resize_image(image, window_height=400)
        # image = QtGui.QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3, QtGui.QImage.Format_BGR888)
        # self.im = QtGui.QPixmap(image)
        # self.label.setPixmap(self.im)

    def displayInfo(self):
        # self.data = data
        self.show()
