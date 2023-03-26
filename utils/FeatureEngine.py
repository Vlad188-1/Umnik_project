from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
from PyQt5.QtCore import Qt


class FeatureEngineWindow(QtWidgets.QMainWindow):

    submitClicked = QtCore.pyqtSignal(pd.DataFrame)

    def __init__(self, data):
        super().__init__()

        #self.data = pd.DataFrame({"a": [1, 2, 3], "b": [10, 100, 1000], "c": [0.3, 0.5, 0.8]})
        self.features = ['Возведение в степень 2',
                         'Возведение в степень 3',
                         'Дифференцирование',
                         'Перемножение признаков',
                         "One hot кодирование"]
        self.data = data
        self.squared_features = dict()
        self.cubic_features = dict()
        self.diff_features = dict()
        self.multiply_features = dict()

        self.setWindowTitle("Feature Engine Windows")
        self.resize(800, 400)

        self.centralwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.setCentralWidget(self.centralwidget)

        self.combobox = QtWidgets.QComboBox(self.centralwidget)
        self.combobox.setGeometry(10, 25, 200, 50)
        self.combobox.addItems(self.features)

        self.list_checkboxes = QtWidgets.QListWidget(self.centralwidget)
        self.list_checkboxes.setDragDropMode(self.list_checkboxes.InternalMove)
        self.list_checkboxes.setGeometry(10, 80, 220, 250)

        if self.data is not None:
            for col in self.data.columns:
                item = QtWidgets.QListWidgetItem(col)
                item.setCheckState(not Qt.Checked)
                self.list_checkboxes.addItem(item)
        # else:
        #     QtWidgets.QMessageBox.about(self, "Error", "Пустая таблица")

        self.add_feature = QtWidgets.QPushButton(self.centralwidget)
        self.add_feature.setGeometry(245, 280, 150, 50)
        self.add_feature.setText("Добавить")
        self.add_feature.clicked.connect(self.addFeatureEngine)

        self.remove_created_feature = QtWidgets.QPushButton(self.centralwidget)
        self.remove_created_feature.setGeometry(400, 280, 150, 50)
        self.remove_created_feature.setText("Удалить запись")
        self.remove_created_feature.clicked.connect(self.deleteCreatedFeature)

        self.run_feature_engine = QtWidgets.QPushButton(self.centralwidget)
        self.run_feature_engine.setGeometry(555, 280, 200, 50)
        self.run_feature_engine.setText("Применить изменения")
        self.run_feature_engine.clicked.connect(self.runFeatureEngine)

        self.list_added_items = QtWidgets.QListWidget(self.centralwidget)
        self.list_added_items.setGeometry(250, 80, self.width() - 300, 170)

    def addFeatureEngine(self):
        current_text_action = self.combobox.currentText()

        if current_text_action == self.features[3]:
            count_checked = 0
            notes_for_adding = []
            for i in range(self.list_checkboxes.count()):
                if self.list_checkboxes.item(i).checkState():
                    count_checked += 1
                    notes_for_adding.append(self.list_checkboxes.item(i).text())
            if count_checked < 2:
                QtWidgets.QMessageBox.about(self, "Error",
                                            "Дополните хотя бы еще один признак для перемножения")
            elif count_checked > 4:
                QtWidgets.QMessageBox.about(self, "Error",
                                            "Максимальное количество признаков для перемножения - 4")
            else:
                self.list_added_items.addItem(f"{current_text_action}: " + " ".join(notes_for_adding))
                self.multiply_features[f"{current_text_action}: " + " ".join(notes_for_adding)] = notes_for_adding

        elif current_text_action == self.features[2]:
            for i in range(self.list_checkboxes.count()):
                if self.list_checkboxes.item(i).checkState():
                    self.list_added_items.addItem(f"{current_text_action}: {self.list_checkboxes.item(i).text()}")
                    self.diff_features[f"{current_text_action}: {self.list_checkboxes.item(i).text()}"] = self.list_checkboxes.item(i).text()
        elif current_text_action == self.features[1]:
            for i in range(self.list_checkboxes.count()):
                if self.list_checkboxes.item(i).checkState():
                    self.list_added_items.addItem(f"{current_text_action}: {self.list_checkboxes.item(i).text()}")
                    self.cubic_features[f"{current_text_action}: {self.list_checkboxes.item(i).text()}"] = self.list_checkboxes.item(i).text()
        elif current_text_action == self.features[0]:
            for i in range(self.list_checkboxes.count()):
                if self.list_checkboxes.item(i).checkState():
                    self.list_added_items.addItem(f"{current_text_action}: {self.list_checkboxes.item(i).text()}")
                    self.squared_features[f"{current_text_action}: {self.list_checkboxes.item(i).text()}"] = self.list_checkboxes.item(i).text()

        for i in range(self.list_checkboxes.count()):
            self.list_checkboxes.item(i).setCheckState(not Qt.Checked)

    def deleteCreatedFeature(self):
        listItems = self.list_added_items.selectedItems()
        if not listItems: return
        for item in listItems:
            self.list_added_items.takeItem(self.list_added_items.row(item))
            #if item.text().startswith(self.features[0]):
            if item.text() in self.squared_features.keys():
                del self.squared_features[item.text()]
            elif item.text() in self.cubic_features.keys():
                del self.cubic_features[item.text()]
                # self.squared_features.remove(item.text())
                #print("Removed: ", item.text().split()[-1])
            #elif item.text().startswith(self.features[1]):
                # self.cubic_features.remove(item.text())
                #print("Removed: ", item.text().split()[-1])
            #elif item.text().startswith(self.features[2]):
                # self.diff_features.remove(item.text())
                #print("Removed: ", item.text().split()[-1])
            #elif item.text().startswith(self.features[3]):
                # self.multiply_features.remove(item.text())
                #print("Removed: ", item.text().split()[2:])

    def runFeatureEngine(self):
        if self.data is not None:
            if len(self.squared_features) > 0:
                all_columns = [item for item in self.squared_features.values()]
                for name_column in all_columns:
                    self.data[name_column + "^2"] = self.data[name_column]**2
            if len(self.cubic_features) > 0:
                all_columns = [item for item in self.cubic_features.values()]
                for name_column in all_columns:
                    self.data[name_column + "^3"] = self.data[name_column]**3
            if len(self.multiply_features) > 0:
                customized_features = [item for item in self.multiply_features.values()]
                for feature in customized_features:
                    if len(feature) == 2:
                        self.data["*".join(feature)] = self.data[feature[0]] * self.data[feature[1]]
                    elif len(feature) == 3:
                        self.data["*".join(feature)] = self.data[feature[0]] * self.data[feature[1]] * self.data[feature[2]]
                    elif len(feature) == 4:
                        self.data["*".join(feature)] = self.data[feature[0]] * self.data[feature[1]] * self.data[feature[2]] \
                                                       * self.data[feature[3]]
            self.submitClicked.emit(self.data)
            QtWidgets.QMessageBox.about(self, "INFO", "Изменения применены!")
        else:
            QtWidgets.QMessageBox.about(self, "ERROR", "Ошибка!")

    def displayInfo(self, data):
        self.data = data
        self.show()
