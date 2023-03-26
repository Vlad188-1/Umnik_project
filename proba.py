import sys
from PyQt5 import QtCore, QtWidgets

class Main_window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Main_window, self).__init__(parent)
        self.setGeometry(50, 50, 1100, 750)
        self.setWindowTitle("Programm")

        self.tabWidget = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabWidget)

        self.tab_v1 = QtWidgets.QWidget()
        self.tabWidget.addTab(self.tab_v1, "Tab 1")

        self.tab_v2 = QtWidgets.QWidget()
        self.tabWidget.addTab(self.tab_v2, "Tab 2")

def main():
    app = QtWidgets.QApplication(sys.argv)
    ex = Main_window()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()