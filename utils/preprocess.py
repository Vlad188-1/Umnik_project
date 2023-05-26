import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal
from math import ceil
from tqdm import tqdm


class Process(QThread):
    change_value = pyqtSignal(int)

    def __init__(self, data: pd.DataFrame):
        super(QThread, self).__init__()
        self.data = data

    def run(self):
        cnt = 0
        if self.data is not None:
            # Replace decimal comma to dot and convert it
            for i in tqdm(self.data.columns, desc='Read files...'):
                if i == 'lith':
                    continue
                else:
                    self.data[i] = self.data[i].replace(',', '.', regex=True).astype(float)
                # cnt += ceil(100 / len(self.data.columns))
                # self.change_value.emit(cnt)
        else:
            self.data = None
