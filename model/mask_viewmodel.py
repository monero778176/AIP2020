from PyQt6.QtWidgets import QTableView, QApplication, QMainWindow
from PyQt6.QtCore import QAbstractTableModel, Qt

# 1. 創建一個繼承自 QAbstractTableModel 的類別，作為我們的資料模型
class maskTableModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    # 2. 定義 rowCount 和 columnCount 方法，告訴 QTableView 表格有多大
    def rowCount(self, parent):
        return len(self._data)

    def columnCount(self, parent):
        return len(self._data[0]) if self._data else 0

    # 3. 定義 data 方法，告訴 QTableView 每個儲存格要顯示什麼
    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            return str(self._data[index.row()][index.column()])
        return None

# 假設這是你的資料
# data = [
#     ['A', 'B', 'C'],
#     ['D', 'E', 'F'],
#     # ... 假設有成千上萬行資料
# ]

# app = QApplication([])
# window = QMainWindow()

# # 4. 創建我們的 Model 實例
# model = MyTableModel(data)

# # 5. 創建 QTableView 並設定它的 Model
# table_view = QTableView()
# table_view.setModel(model)

# window.setCentralWidget(table_view)
# window.show()
# app.exec()