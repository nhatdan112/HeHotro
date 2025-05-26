import os
import matplotlib.pyplot as plt
from datetime import datetime

# Thư mục lưu biểu đồ
THU_MUC_BIEU_DO = os.path.join('static', 'charts')
if not os.path.exists(THU_MUC_BIEU_DO):
    os.makedirs(THU_MUC_BIEU_DO)

