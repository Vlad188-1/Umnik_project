# Additional imports
import pandas as pd
from math import ceil
import os
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime

# PyQt5 imports
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QPushButton, QLabel, QLineEdit, QComboBox
# from PyQt5.QtWebEngineWidgets import QWebEngineView

# My imports
from utils.AdditionalWindows import FeatureEngineWindow, DeleteNanWindow, TestDataWindow
from utils import Process, Constants
from src import CustomDataset, NN, train_and_validation, AutoEncoder, train_and_validation_autoencoder
from utils.PlotGraphics import Plots

# Import for training
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Vertical:
                return str(self._data.index[section])


class TrainModel(QThread):
    def __init__(self, train_function, training_params):
        super(QThread, self).__init__()
        self.train_function = train_function
        self.training_params = training_params

    def run(self):
        self.train_function(**self.training_params)


class Ui_MainWindow():

    def __init__(self):
        self.filename = ""
        self.data = None
        # self.depth = None
        self.model_table = None
        self.separators = [":", ",", ";"]
        self.true_checkboxes_text = []
        self.data_copy = None
        self.true_cols = []
        self.false_cols = []

        # For prepared data
        self.project_path = ""

        self.x_train = None
        self.x_valid = None
        self.y_train = None
        self.y_valid = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setMouseTracking(True)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("assets/oil-pump.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        # MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        # MainWindow.setAnimated(True)
        # MainWindow.setDocumentMode(False)
        # MainWindow.setDockNestingEnabled(False)
        # MainWindow.setDockOptions(QtWidgets.QMainWindow.AllowTabbedDocks|QtWidgets.QMainWindow.AnimatedDocks)
        # MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        MainWindow.resize(1320, 770)

        self.tabs = QtWidgets.QTabWidget(MainWindow)
        MainWindow.setCentralWidget(self.tabs)
        # Main widget
        self.centralwidgetTrain = QtWidgets.QWidget(MainWindow)
        self.centralwidgetInference = QtWidgets.QWidget(MainWindow)
        # self.centralwidget.setObjectName("centralwidget")

        self.tabs.addTab(self.centralwidgetTrain, "Обучение моделей")
        self.tabs.addTab(self.centralwidgetInference, "Инференс моделей")
        # MainWindow.setCentralWidget(self.tabs)

        # self.web = QWebEngineView(self.centralwidgetTrain)

        # self.web = QWebEngineView(self.centralwidget)
        # self.browser = QtWebEngineWidgets.QWebEngineView(self.centralwidget)
        # self.browser.setGeometry(800, 350, 450, 300)
        # self.show_button = QPushButton(self.centralwidget)
        # self.show_button.setText("Распределение NaN значений")
        # self.show_button.setGeometry(800, 10, 200, 30)
        # self.show_button.clicked.connect(self.showDistributionNanValues)

        # Buttons
        self.ButtonProcessData = QPushButton(self.centralwidgetTrain)
        self.ButtonProcessData.setText("Обработка данных")
        self.ButtonProcessData.setGeometry(300, 20, 150, 30)
        self.ButtonProcessData.setObjectName("pushButton")
        self.ButtonProcessData.setStyleSheet(Constants.SELECTALL_BUTTON)
        self.ButtonProcessData.clicked.connect(self.preProcess)

        self.line = QtWidgets.QFrame(self.centralwidgetTrain)
        self.line.setGeometry(QtCore.QRect(730, 20, 40, 640))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)

        # TABLE: Layout for table
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidgetTrain)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 70, 700, 411))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        # TABLE: Table
        self.table = QtWidgets.QTableView()
        self.verticalLayout.addWidget(self.table)

        # TABLE: Text with number of rows and columns
        self.num_rows = QtWidgets.QLabel("<b>Количество строк: </b>" + "0")
        self.verticalLayout.addWidget(self.num_rows)
        self.num_columns = QtWidgets.QLabel()
        self.num_columns.setText("<b>Количество столбцов: </b>" + "0")
        self.verticalLayout.addWidget(self.num_columns)

        # Checkbox for test
        # self.checkbox_test = QtWidgets.QCheckBox()
        # self.checkbox_test.stateChanged.connect(self.checkBoxChangedAction)
        # self.labelA = QtWidgets.QLabel("Not slected.")
        # self.verticalLayout.addWidget(self.checkbox_test)
        # self.verticalLayout.addWidget(self.labelA)

        # Buttons
        # self.button4 = QtWidgets.QPushButton("Four")
        # self.button5 = QtWidgets.QPushButton("Five")
        # self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        # self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(800, 100, 300, 200))
        # self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        # self.verticalLayoutWidget_2.setStyleSheet(Constants.PROCESSING_BUTTONS_STYLES)

        # TABLE: Combobx separator
        self.name_combobox = QtWidgets.QLabel(self.centralwidgetTrain)
        self.name_combobox.setText("<b>Выбрать сепаратор</b>")
        self.name_combobox.setGeometry(10, 10, 150, 50)

        self.commobox_separators = QComboBox(self.centralwidgetTrain)
        self.commobox_separators.setGeometry(QtCore.QRect(170, 10, 100, 50))
        self.commobox_separators.addItems(self.separators)
        self.commobox_separators.setObjectName("combox")
        self.commobox_separators.currentIndexChanged['QString'].connect(self.updateSeparatorTable)

        # MENU: Menubar
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 24))
        self.menubar.setNativeMenuBar(False)
        self.menubar.setObjectName("menubar")
        self.menubar.setStyleSheet("border :1px solid grey;")

        font_menu_bar = QFont()
        font_menu_bar.setBold(True)
        self.menubar.setFont(font_menu_bar)

        # MENU: Open menu
        self.menuOpen = QtWidgets.QMenu(self.menubar)
        self.menuOpen.setObjectName("menuFile")

        # MENU: Edit Menu
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        MainWindow.setMenuBar(self.menubar)

        # MENU: Action open file
        self.actionOpen = QtWidgets.QAction(MainWindow)
        iconOpen = QtGui.QIcon()
        iconOpen.addPixmap(QtGui.QPixmap("assets/open-folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpen.setIcon(iconOpen)
        self.actionOpen.setObjectName("actionOpen")
        self.actionOpen.setFont(QFont("", 14))

        # MENU: Action save file
        self.actionSaveFile = QtWidgets.QAction(MainWindow)
        self.actionSaveFile.setObjectName("saveFile")
        iconSave = QtGui.QIcon()
        iconSave.addPixmap(QtGui.QPixmap("assets/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSaveFile.setIcon(iconSave)
        self.actionSaveFile.setFont(QFont("", 14))

        # MENU: Status bar
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuOpen.menuAction())

        # Progress bar for processing
        self.pbar_process = QtWidgets.QProgressBar(self.statusbar)
        self.pbar_process.setGeometry(0, 0, 500, 25)
        self.pbar_process.setStyleSheet("QProgressBar {border: 2px solid grey;border-radius:8px;padding:1px}"
                                        "QProgressBar::chunk {background:lightgreen}")
        self.pbar_process.setVisible(False)
        self.pbar_process.setAlignment(Qt.AlignCenter)
        self.pbar_process.setMaximum(100)

        # Layout for buttons
        self.verticalLayoutWidgetTable = QtWidgets.QWidget(self.centralwidgetTrain)
        self.verticalLayoutWidgetTable.setGeometry(QtCore.QRect(10, 500, 300, 200))
        self.verticalLayoutWidgetTable.setObjectName("verticalLayoutWidget_3")

        self.layoutWorkWithTable = QtWidgets.QGridLayout(self.verticalLayoutWidgetTable)

        # TABLE: Checkboxes
        self.list_checkboxes = QtWidgets.QListWidget(self.centralwidgetTrain)
        self.list_checkboxes.setDragDropMode(self.list_checkboxes.InternalMove)
        self.list_checkboxes.setGeometry(10, 500, 200, 150)
        # self.layoutWorkWithTable.addWidget(self.centralwidget)
        # self.layoutWotkWithTable.addWidget(self.pushButton_3, 1, 3)
        # self.layoutWotkWithTable.addWidget(self.button4, 0, 1)
        # self.layoutWotkWithTable.addWidget(self.button5, 1, 0)

        # TABLE: Button update table
        self.buttonUpdateTable = QPushButton(self.centralwidgetTrain)
        self.buttonUpdateTable.setGeometry(230, 500, 200, 30)
        self.buttonUpdateTable.setText("Обновить таблицу")
        self.buttonUpdateTable.clicked.connect(self.update_table_checkboxes)
        self.buttonUpdateTable.setStyleSheet(Constants.UPDATE_TABLE_BUTTON_STYLE)

        # TABLE: Button select all checkboxes
        self.buttonSelectAll = QPushButton(self.centralwidgetTrain)
        self.buttonSelectAll.setGeometry(230, 540, 200, 30)
        self.buttonSelectAll.setText("Выбрать все элементы")
        self.buttonSelectAll.clicked.connect(lambda: self.selectAll(True))
        self.buttonSelectAll.setStyleSheet(Constants.SELECTALL_BUTTON)
        # fontSelectButton = QFont()
        # fontSelectButton.setBold(True)
        # self.buttonSelectAll.setFont(fontSelectButton)

        # TABLE: Button unselect all checkboxes
        self.buttonUnselectAll = QPushButton(self.centralwidgetTrain)
        self.buttonUnselectAll.setGeometry(230, 580, 200, 30)
        self.buttonUnselectAll.setText("Отменить выбор")
        self.buttonUnselectAll.clicked.connect(lambda: self.selectAll(False))
        self.buttonUnselectAll.setStyleSheet(Constants.SELECTALL_BUTTON)

        # TABLE: Button delete NaN values
        self.buttonDeleteNan = QPushButton(self.centralwidgetTrain)
        self.buttonDeleteNan.setGeometry(450, 500, 250, 30)
        self.buttonDeleteNan.setText("Работа с Nan значениями")
        self.buttonDeleteNan.setStyleSheet(Constants.DELETE_NAN)
        self.buttonDeleteNan.clicked.connect(lambda: self.OpenDeleteNanWindow())

        # TABLE: Combobox target variable
        self.probaWidget = QtWidgets.QWidget(self.centralwidgetTrain)
        self.probaWidget.setGeometry(755, 10, 550, 230)
        self.probaWidget.setStyleSheet("""QWidget {background-color:lightgrey; border: 1px solid black}""")

        self.labelCombobox = QtWidgets.QLabel(self.centralwidgetTrain)
        self.labelCombobox.setText("<b>Выбрать целевую переменную</b>")
        self.labelCombobox.setGeometry(770, 10, 250, 50)
        self.comboboxSelectTargetVariable = QComboBox(self.centralwidgetTrain)
        self.comboboxSelectTargetVariable.setGeometry(1050, 10, 150, 50)

        # TABLE: Plot distribution target variable
        self.buttonDistributionTargetVariable = QPushButton(self.centralwidgetTrain)
        self.buttonDistributionTargetVariable.setGeometry(1030, 185, 270, 30)
        self.buttonDistributionTargetVariable.setText("Распределение целевой переменной")
        self.buttonDistributionTargetVariable.setStyleSheet(Constants.DELETE_NAN)
        self.buttonDistributionTargetVariable.clicked.connect(self.buttonPlotDistributionTargetVariable)

        # TABLE: Split size
        self.labelSplitSize = QLabel(self.centralwidgetTrain)
        self.labelSplitSize.setText("<b>Соотношение разбиения выборки</b>")
        self.labelSplitSize.setGeometry(770, 60, 250, 50)
        self.comboboxSplitSize = QComboBox(self.centralwidgetTrain)
        self.comboboxSplitSize.setGeometry(1050, 60, 150, 50)
        self.comboboxSplitSize.addItems(["70:30", "75:25", "80:20", "85:15", "90:10"])
        self.comboboxSplitSize.setCurrentIndex(2)

        # TABLE: Button save prepared dataset
        self.buttonSavePreparedDataset = QPushButton(self.centralwidgetTrain)
        self.buttonSavePreparedDataset.setGeometry(760, 120, 260, 40)
        self.buttonSavePreparedDataset.setText("Сохранить подготовленный датасет")
        self.buttonSavePreparedDataset.clicked.connect(self.SavePreparedDataset)

        # TABLE: Open prepared dataset
        self.buttonOpenPreparedDataset = QPushButton(self.centralwidgetTrain)
        self.buttonOpenPreparedDataset.setText("Открыть подготовленный датасет")
        self.buttonOpenPreparedDataset.setGeometry(760, 180, 260, 40)
        self.buttonOpenPreparedDataset.clicked.connect(self.loadPreparedData)

        # FE: Button open feature engine window
        self.buttonFeatureEngine = QPushButton(self.centralwidgetTrain)
        self.buttonFeatureEngine.setGeometry(230, 620, 200, 30)
        self.buttonFeatureEngine.setText("Feature Engine")
        # self.buttonUnselectAll.clicked.connect(lambda: self.selectAll(False))
        self.buttonFeatureEngine.setStyleSheet(Constants.SELECTALL_BUTTON)
        self.buttonFeatureEngine.clicked.connect(lambda: self.OpenFeatureEngineWindow())

        # MODEL: Model settings
        self.border_train = QtWidgets.QWidget(self.centralwidgetTrain)
        self.border_train.setGeometry(755, 250, 550, 270)
        self.border_train.setStyleSheet("""QWidget {background-color:lightgrey; border: 1px solid black}""")

        self.setting_model_params = QLabel(self.centralwidgetTrain)
        self.setting_model_params.setText("Настройка параметров модели")
        self.setting_model_params.setGeometry(760, 250, 350, 50)
        self.setting_model_params.setFont(QFont("Times", 16, QtGui.QFont.Bold))

        # MODEL: Batch size
        self.batch_size_label = QLabel(self.centralwidgetTrain)
        self.batch_size_label.setText("Размер батча")
        self.batch_size_label.setGeometry(760, 300, 350, 50)

        self.batch_size_lineEdit = QLineEdit(self.centralwidgetTrain)
        self.batch_size_lineEdit.setGeometry(880, 315, 50, 20)
        self.batch_size_lineEdit.setStyleSheet("""QLineEdit {border: 1px solid black;}""")
        self.batch_size_lineEdit.setText("64")
        self.batch_size_lineEdit.setAlignment(QtCore.Qt.AlignCenter)

        # MODEL: Epochs
        self.epochs_label = QLabel(self.centralwidgetTrain)
        self.epochs_label.setText("Кол-во эпох")
        self.epochs_label.setGeometry(1050, 300, 300, 50)

        self.epochs_lineEdit = QLineEdit(self.centralwidgetTrain)
        self.epochs_lineEdit.setGeometry(1150, 315, 50, 20)
        self.epochs_lineEdit.setStyleSheet("""QLineEdit {border: 1px solid black;}""")
        self.epochs_lineEdit.setText("100")
        self.epochs_lineEdit.setAlignment(QtCore.Qt.AlignCenter)

        # MODEL: Optimizator
        self.optimizator_label = QLabel(self.centralwidgetTrain)
        self.optimizator_label.setText("Оптимизатор")
        self.optimizator_label.setGeometry(760, 360, 200, 50)

        self.optimizator_combobox = QComboBox(self.centralwidgetTrain)
        self.optimizator_combobox.setGeometry(875, 365, 100, 50)
        self.optimizator_combobox.addItems(["Adam", "SGD"])
        self.optimizator_combobox.setCurrentIndex(0)

        # MODEL: Learning rate
        self.learning_rate_label = QLabel(self.centralwidgetTrain)
        self.learning_rate_label.setText("Коэффициент обучения")
        self.learning_rate_label.setGeometry(1050, 360, 200, 50)

        self.learning_rate_lineEdit = QLineEdit(self.centralwidgetTrain)
        self.learning_rate_lineEdit.setGeometry(1220, 375, 70, 20)
        self.learning_rate_lineEdit.setStyleSheet("""QLineEdit {border: 1px solid black;}""")
        self.learning_rate_lineEdit.setText("0.0001")
        self.learning_rate_lineEdit.setAlignment(QtCore.Qt.AlignCenter)

        # MODEL: Criterion
        self.criterion_NN = QLabel(self.centralwidgetTrain)
        self.criterion_NN.setText("Критерий NN")
        self.criterion_NN.setGeometry(1050, 420, 200, 50)

        self.criterion_NN_combobox = QComboBox(self.centralwidgetTrain)
        self.criterion_NN_combobox.setGeometry(1150, 435, 150, 20)
        self.criterion_NN_combobox.addItems(["CrossEntropy", "BinaryCrossEntropy"])
        self.criterion_NN_combobox.setStyleSheet("""QLineEdit {border: 1px solid black;}""")

        # MODEL: Select model
        self.model_label = QLabel(self.centralwidgetTrain)
        self.model_label.setText("Архитектура модели")
        self.model_label.setGeometry(760, 420, 200, 50)

        self.combobox_models = QComboBox(self.centralwidgetTrain)
        self.combobox_models.setGeometry(900, 423, 100, 50)
        self.combobox_models.addItems(["NN", "AE_NN"])
        self.combobox_models.setCurrentIndex(1)

        # MODEL: Select device
        self.device_label = QLabel(self.centralwidgetTrain)
        self.device_label.setText("Вычислитель")
        self.device_label.setGeometry(1050, 480, 200, 30)

        self.combobox_devices = QComboBox(self.centralwidgetTrain)
        self.combobox_devices.setGeometry(1150, 473, 100, 50)
        self.combobox_devices.addItems(["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
        self.combobox_devices.setCurrentIndex(0)

        # MODEL: Run model
        self.button_runModel = QPushButton(self.centralwidgetTrain)
        self.button_runModel.setGeometry(760, 480, 200, 30)
        self.button_runModel.setText("Запустить обучение")
        self.button_runModel.setStyleSheet(Constants.UPDATE_TABLE_BUTTON_STYLE)
        self.button_runModel.clicked.connect(self.runModel)

        # Инференс
        self.border_inference = QtWidgets.QWidget(self.centralwidgetTrain)
        self.border_inference.setGeometry(755, 530, 550, 150)
        self.border_inference.setStyleSheet("""QWidget {background-color:lightgrey; border: 1px solid black}""")

        self.module_test_wells = QLabel(self.centralwidgetTrain)
        self.module_test_wells.setText("Модуль тестирования данных")
        self.module_test_wells.setGeometry(760, 520, 350, 50)
        self.module_test_wells.setFont(QFont("Times", 16, QtGui.QFont.Bold))

        # Test wells
        self.button_test_data = QPushButton(self.centralwidgetTrain)
        self.button_test_data.setGeometry(760, 580, 250, 30)
        self.button_test_data.setText("Протестировать данные")
        self.button_test_data.setStyleSheet(Constants.SELECTALL_BUTTON)
        self.button_test_data.clicked.connect(lambda: self.OpenTestDataWindow())

        # INFERECNE:
        # self.button_edit_test_data = QPushButton(self.centralwidgetTrain)
        # self.button_edit_test_data.setGeometry(450, 540, 250, 30)
        # self.button_edit_test_data.setText("Использовать конфиг подготовки")
        # self.button_edit_test_data.setStyleSheet(Constants.DELETE_NAN)
        # self.button_edit_test_data.clicked.connect()
        # self.comboboxSelectTargetVariable.addItems(self.data)
        # self.comboboxSelectTargetVariable.currentIndexChanged['QString'].connect(self.updateSeparatorTable)
        # self.buttonEditTargetVariable = QPushButton(self.centralwidget)
        # self.buttonEditTargetVariable.setGeometry(650, 30, 250, 30)
        # self.buttonEditTargetVariable.setText("")

        # MENU: Menu settings
        self.actionSettings = QtWidgets.QAction(MainWindow)
        iconSettings = QtGui.QIcon()
        iconSettings.addPixmap(QtGui.QPixmap("assets/settings.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSettings.setIcon(iconSettings)
        self.actionSettings.setObjectName("actionSettings")
        self.actionSettings.setFont(QFont("", 14))

        self.menuOpen.addAction(self.actionOpen)
        self.menuOpen.addAction(self.actionSaveFile)
        self.menuEdit.addAction(self.actionSettings)
        self.menubar.addAction(self.menuOpen.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # # INFERECNE
        #
        # self. = QLabel(self.centralwidgetTrain)


    # def updateColumnsTable(self):
    #     if ch.isChecked():
    #         self.data = self.data[self.data.columns]
    #         self.model = TableModel(self.data[self.data.columns])
    #         self.table.setModel(self.model)
    #         #self.labelA.setText("Selected.")
    #     else:
    #         print(ch.text)
    #         cols = self.data[set(self.data.columns) - set(ch.text)]
    #         self.data = self.data[cols]
    #         self.model = TableModel(self.data[cols])
    #         self.table.setModel(self.model)

    # def showDistributionNanValues(self):
    #
    #     plotDistributionNanValues(self.data)
    #
    #     file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Nan.html"))
    #     self.web.load(QtCore.QUrl.fromLocalFile(file_path))
    #     self.web.show()

    def on_selection_changed(self, selected, deselected):
        for item in selected.indexes():
            print(item.row(), item.column())
            print(self.data.iloc[item.row():item.row() + 1, item.column():item.column() + 1])
            print(type(self.data.iloc[item.row():item.row() + 1, item.column():item.column() + 1]))

    # MENU: Function for open file menu
    def openFileDialog(self):
        self.filename, _ = QtWidgets.QFileDialog.getOpenFileName(filter="*.csv")
        print(self.filename)
        try:
            self.data = pd.read_csv(self.filename, sep=self.commobox_separators.currentText())
        except:
            self.data = pd.DataFrame({"empty": [""]})
        self.model_table = TableModel(self.data)
        self.table.setModel(self.model_table)
        self.table.selectionModel().selectionChanged.connect(self.on_selection_changed)
        self.num_rows.setText("<b>Количество строк:</b> " + str(len(self.data)) if self.data is not None else "0")
        self.num_columns.setText(
            "<b>Количество столбцов: </b>" + str(len(self.data.columns)) if self.data is not None else "0")
        self.data_copy = self.data.copy()
        if self.data is not None:
            self.list_checkboxes.clear()
            self.comboboxSelectTargetVariable.addItems(self.data.columns)
            for col in self.data.columns:
                item = QtWidgets.QListWidgetItem(col)
                item.setCheckState(Qt.Checked)
                self.list_checkboxes.addItem(item)
        else:
            self.list_checkboxes.clear()
            item = QtWidgets.QListWidgetItem("None")
            item.setCheckState(Qt.Checked)
            self.list_checkboxes.addItem(item)

    # TABLE: Function preprocessing for button
    def preProcess(self):
        self.pbar_process.setVisible(True)
        self.thread = Process(self.data)
        # self.thread.change_value.connect(self.setProgressVal)
        self.thread.start()
        self.data = self.thread.data
        if self.data is not None:
            self.model_table = TableModel(self.data)
            self.table.setModel(self.model_table)
            self.table.selectionModel().selectionChanged.connect(self.on_selection_changed)
        else:
            QtWidgets.QMessageBox.about(MainWindow, "Error", "Нет данных для предобработки!")

    def setProgressVal(self, val):
        # self.button.setEnabled(False)
        # self.pbar.setAlignment(Qt.AlignCenter)
        self.pbar_process.setValue(val)
        # if ceil(val) >= 100:
        #     QtWidgets.QMessageBox.about(MainWindow, "Success", "Предобработка данных завершилась успешно!")
        #     self.pbar_process.setVisible(False)

    # TABLE: Function for update separator button
    def updateSeparatorTable(self):
        try:
            self.data = pd.read_csv(self.filename, sep=self.commobox_separators.currentText())
        except:
            self.data = pd.DataFrame({"empty": [""]})
            QtWidgets.QMessageBox.about(MainWindow, "Error", "Файл прочтен некорректно. Попробуйте изменить сепаратор!")
        self.model_table = TableModel(self.data)
        self.table.setModel(self.model_table)
        self.table.selectionModel().selectionChanged.connect(self.on_selection_changed)
        self.num_rows.setText("<b>Количество строк:</b> " + str(len(self.data)) if self.data is not None else "0")
        self.num_columns.setText(
            "<b>Количество столбцов: </b>" + str(len(self.data.columns)) if self.data is not None else "0")

        if self.data is not None:
            self.list_checkboxes.clear()
            self.data_copy = self.data.copy()
            self.comboboxSelectTargetVariable.clear()
            self.comboboxSelectTargetVariable.addItems(self.data.columns)
            for col in self.data.columns:
                item = QtWidgets.QListWidgetItem(col)
                item.setCheckState(Qt.Checked)
                self.list_checkboxes.addItem(item)
                # self.layoutWorkWithTable.addWidget(self.list_checkboxes, 0, 0)
        else:
            self.list_checkboxes.clear()
            item = QtWidgets.QListWidgetItem("None")
            item.setCheckState(Qt.Checked)
            self.list_checkboxes.addItem(item)
            # self.layoutWorkWithTable.addWidget(self.list_checkboxes, 0, 0)

    def update_table_checkboxes(self):
        # true_cols = []
        # false_cols = []
        self.true_cols.clear()
        self.false_cols.clear()
        for i in range(self.list_checkboxes.count()):
            if self.list_checkboxes.item(i).checkState():
                self.true_cols.append(self.list_checkboxes.item(i).text())
            else:
                self.false_cols.append(self.list_checkboxes.item(i).text())
        if self.data is not None:
            self.data = self.data_copy[self.true_cols]
            self.model_table = TableModel(self.data)
            self.table.setModel(self.model_table)
            self.table.selectionModel().selectionChanged.connect(self.on_selection_changed)
            self.num_rows.setText("<b>Количество строк:</b> " + str(len(self.data)) if self.data is not None else "0")
            self.num_columns.setText(
                "<b>Количество столбцов: </b>" + str(len(self.data.columns)) if self.data is not None else "0")
            self.comboboxSelectTargetVariable.clear()
            self.comboboxSelectTargetVariable.addItems(self.data.columns)
        else:
            QtWidgets.QMessageBox.about(MainWindow, "ERROR", "Ошибка!")
        print("True cols: ", self.true_cols)
        print("False cols: ", self.false_cols)

    # TABLE: Function select all function for button
    def selectAll(self, flag: bool):
        for i in range(self.list_checkboxes.count()):
            self.list_checkboxes.item(i).setCheckState(Qt.Checked if flag is True else not Qt.Checked)

    def _update_table(self):
        pass

    # def update_list_checkboxes(self):
    #     #self.list_checkboxes.clear()
    #     new_cols = list(set(self.data_copy.columns) - set(self.data.columns))
    #     all_cols = self.data_copy.columns + new_cols
    #     for col in all_cols:
    #         item = QtWidgets.QListWidgetItem(col)
    #         item.setCheckState(Qt.Checked)
    #         self.list_checkboxes.addItem(item)

    # FE: Function to open feature engine window
    def OpenFeatureEngineWindow(self):  # <===
        # self.engineWindow.data = self.data
        self.engineWindow = FeatureEngineWindow(self.data)
        self.engineWindow.submitClicked.connect(self.confirmChanges)
        self.engineWindow.displayInfo(self.data)

    def OpenTestDataWindow(self):
        # if self.data is not None:
        self.testWindow = TestDataWindow(self.data.copy() if self.data is not None else self.data)
        # self.testWindow.submitClicked.connect(self.confirmChanges)
        self.testWindow.displayInfo()
        # else:
        #     QtWidgets.QMessageBox.about(MainWindow, "ERROR", "Не загружены данные для тестирования! Загрузите данные, чтобы открыть меню")

    # TABLE: Function for deleting NaN values
    def OpenDeleteNanWindow(self):
        self.deleteNanWindow = DeleteNanWindow(self.data)
        self.deleteNanWindow.submitClicked.connect(self.confirmChanges)
        self.deleteNanWindow.displayInfo()

    # TABLE: Function to confirm changes
    def confirmChanges(self, data):  # <-- This is the main window's slot
        self.data = data
        self.model_table = TableModel(self.data)
        self.table.setModel(self.model_table)

        self.num_rows.setText("<b>Количество строк:</b> " + str(len(self.data)) if self.data is not None else "0")
        self.num_columns.setText("<b>Количество столбцов: </b>" + str(len(self.data.columns)) if self.data \
                                                                                                 is not None else "0")
        self.comboboxSelectTargetVariable.clear()
        self.comboboxSelectTargetVariable.addItems(self.data.columns)
        # self.update_list_checkboxes()

    # TABLE: Function to plot distribution target variable
    def buttonPlotDistributionTargetVariable(self):
        text_y = self.comboboxSelectTargetVariable.currentText()
        y = self.data[text_y]
        Plots.plotDistributedTargetVariable(y)
        # file_path = Path("plots", "TargetVariable.html")
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "plots/TargetVariable.html"))
        # self.web.load(QtCore.QUrl.fromLocalFile(file_path))
        # self.web.show()

    # TABLE: Function to save prepared dataset
    def SavePreparedDataset(self):
        if self.data is not None:
            if not self.data.isnull().values.any():
                dialog = QtWidgets.QFileDialog(None, caption='Data Log File Dir')
                dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
                dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
                dialog.setDirectory(str(Path().cwd() / Path("projects")))
                dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
                dialog.setLabelText(QtWidgets.QFileDialog.Accept, "Save")

                if dialog.exec_() == QtWidgets.QFileDialog.Accepted:
                    project_dir = dialog.selectedFiles()[-1]
                    Path(project_dir).mkdir(exist_ok=True)
                    self.project_path = project_dir
                    # if self.data is not None:
                    #     if not self.data.isnull().values.any():
                    _, test_size = self.comboboxSplitSize.currentText().split(":")
                    y = self.data[self.comboboxSelectTargetVariable.currentText()]
                    x = self.data.drop([self.comboboxSelectTargetVariable.currentText()], axis=1)
                    self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
                        x, y, shuffle=True, test_size=int(test_size) / 100,
                        stratify=self.data[self.comboboxSelectTargetVariable.currentText()], random_state=12)

                    self.x_train.to_csv(Path(project_dir, "X_train.csv"), index=False)
                    self.x_valid.to_csv(Path(project_dir, "X_valid.csv"), index=False)
                    self.y_train.to_csv(Path(project_dir, "y_train.csv"), index=False)
                    self.y_valid.to_csv(Path(project_dir, "y_valid.csv"), index=False)

                    dict_yaml = {}
                    mul_list = []
                    pow_2_list = []
                    pow_3_list = []

                    for col in self.x_train.columns:
                        if "*" in col:
                            mul_list.append(col.split("*"))
                        elif "^2" in col:
                            pow_2_list.append(col.split("^")[0])
                        elif "^3" in col:
                            pow_3_list.append(col.split("^")[0])
                    dict_yaml["*"] = mul_list
                    dict_yaml["^2"] = pow_2_list
                    dict_yaml["^3"] = pow_3_list
                    dict_yaml["Number features"] = len(self.x_train.columns)
                    dict_yaml["Target variable"] = self.comboboxSelectTargetVariable.currentText()
                    with open(Path(project_dir, 'config_processing_data.yml'), 'w') as outfile:
                        yaml.dump(dict_yaml, outfile, default_flow_style=False)
            else:
                QtWidgets.QMessageBox.about(MainWindow, "Error",
                                            "Данные содержат NaN значения. Удалите их перед сохранением проекта")

    # TABLE: Function to load prepared dataset
    def loadPreparedData(self):
        self.project_path = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select Folder')
        all_files = sorted(list(Path(self.project_path).glob("*.csv")))
        for file in all_files:
            if file.name.startswith("X_train"):
                self.x_train = pd.read_csv(file)
            elif file.name.startswith("X_valid"):
                self.x_valid = pd.read_csv(file)
            elif file.name.startswith("y_train"):
                self.y_train = pd.read_csv(file)
            elif file.name.startswith("y_valid"):
                self.y_valid = pd.read_csv(file)

        if (self.x_train is not None) and \
                (self.x_valid is not None) and \
                (self.y_train is not None) and \
                (self.y_valid is not None):
            X = pd.concat([self.x_train, self.x_valid], axis=0).reset_index(drop=True)
            y = pd.concat([self.y_train, self.y_valid], axis=0).reset_index(drop=True)
            self.data = pd.concat([X, y], axis=1)
            self.model_table = TableModel(self.data)
            self.table.setModel(self.model_table)
            self.table.selectionModel().selectionChanged.connect(self.on_selection_changed)
            self.num_rows.setText("<b>Количество строк:</b> " + str(len(self.data)) if self.data is not None else "0")
            self.num_columns.setText(
                "<b>Количество столбцов: </b>" + str(len(self.data.columns)) if self.data is not None else "0")
            self.data_copy = self.data.copy()
            if self.data is not None:
                self.list_checkboxes.clear()
                self.comboboxSelectTargetVariable.clear()
                self.comboboxSelectTargetVariable.addItems(self.data.columns)
                for col in self.data.columns:
                    item = QtWidgets.QListWidgetItem(col)
                    item.setCheckState(Qt.Checked)
                    self.list_checkboxes.addItem(item)
            else:
                self.list_checkboxes.clear()
                item = QtWidgets.QListWidgetItem("None")
                item.setCheckState(Qt.Checked)
                self.list_checkboxes.addItem(item)
        else:
            QtWidgets.QMessageBox.about(MainWindow, "Error",
                                        "Ошибка при загрузке подготволенного датасета. Проверьте его наличие!")
            # print(file)

    # MODEL: Function to run model
    def runModel(self):
        # train_dataset = CustomDataset(torch.FloatTensor(self.x_train.values),
        #                               torch.FloatTensor(self.y_train.values))
        # val_dataset = CustomDataset(torch.FloatTensor(self.x_valid.values),
        #                             torch.FloatTensor(self.y_valid.values))

        train_dataset = CustomDataset(torch.from_numpy(self.x_train.values).float(),
                                          torch.from_numpy(self.y_train.values).long())
        val_dataset = CustomDataset(torch.from_numpy(self.x_valid.values).float(),
                                        torch.from_numpy(self.y_valid.values).long())

        device = self.combobox_devices.currentText()
        # if torch.cuda.is_available():
        #     device = "cuda"
        # else:
        #     device = "cpu"

        # writer = SummaryWriter()
        # criterion_NN = BCEWithLogitsLoss()
        if self.criterion_NN.text() == "CrossEntropy":
            criterion_NN = CrossEntropyLoss()
        elif self.criterion_NN.text() == "BinaryCrossEntropy":
            criterion_NN = BCEWithLogitsLoss()
        else:
            criterion_NN = CrossEntropyLoss()

        if self.combobox_models.currentText() == "AE_NN":
            model_AE = AutoEncoder(in_features=self.x_train.shape[1]).to(device)
            model_NN = NN(in_features=model_AE.encoder.bottleneck_linear3.out_features,
                          num_classes=len(np.unique(self.y_valid.values)))
            criterion_AE = torch.nn.MSELoss()

            if self.optimizator_combobox.currentText() == "Adam":
                optimizer_NN = Adam(model_NN.parameters(), lr=eval(self.learning_rate_lineEdit.text()))
            elif self.optimizator_combobox.currentText() == "SGD":
                optimizer_NN = SGD(model_NN.parameters(), lr=eval(self.learning_rate_lineEdit.text()))
            optimizer_AE = Adam(model_AE.parameters(), lr=0.001)

            # if not self.project_path:
            #     time_now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            #     self.project_path = str(Path("projects", time_now))

            training_params = {"model_AE": model_AE,
                               "model_NN": model_NN,
                               "train_dataset": train_dataset,
                               "val_dataset": val_dataset,
                               "batch_size": int(self.batch_size_lineEdit.text()),
                               "epochs_AE": int(self.epochs_lineEdit.text()),
                               "epochs_NN": int(self.epochs_lineEdit.text()),
                               "device": device,
                               "criterion_AE": criterion_AE,
                               "criterion_NN": criterion_NN,
                               "optimizer_AE": optimizer_AE,
                               "optimizer_NN": optimizer_NN,
                               "out_dir": self.project_path,
                               "scheduler": None}

            print(training_params)

            self.thread = TrainModel(train_and_validation_autoencoder, training_params)
            self.thread.start()

        elif self.combobox_models.currentText() == "NN":
            model_NN = NN(in_features=self.x_train.shape[1], num_classes=4).to(device)

            if self.optimizator_combobox.currentText() == "Adam":
                optimizer_NN = Adam(model_NN.parameters(), lr=eval(self.learning_rate_lineEdit.text()))
            elif self.optimizator_combobox.currentText() == "SGD":
                optimizer_NN = SGD(model_NN.parameters(), lr=eval(self.learning_rate_lineEdit.text()))

            training_params = {"model": model_NN,
                               "train_dataset": train_dataset,
                               "val_dataset": val_dataset,
                               "batch_size": int(self.batch_size_lineEdit.text()),
                               "epochs": int(self.epochs_lineEdit.text()),
                               "device": device,
                               "criterion": criterion_NN,
                               "optimizer": optimizer_NN,
                               "out_dir": self.project_path,
                               "scheduler": None}

            self.thread = TrainModel(train_and_validation, training_params)
            self.thread.start()

    # INFERENCE: Button to open config.yaml
    # def loadConfigFile(self):
    #     config_name, _ = QtWidgets.QFileDialog.getOpenFileName(filter="*.csv")
    #     pass

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

        # MENU: Menu
        self.menuOpen.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))

        # MENU: Actions for menu
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionSaveFile.setText(_translate("MainWindow", "Save"))
        self.actionSettings.setText(_translate("MainWindow", "Settings"))

        # MENU: Buttons
        self.actionOpen.triggered.connect(self.openFileDialog)
        # self.commobox_separators.currentIndexChanged['QString'].connect(self.updateSeparatorTable)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
