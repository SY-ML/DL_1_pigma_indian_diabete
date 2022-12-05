"""
AutoViz Installation:
pip install -q AutoViz
"""

from autoviz.AutoViz_Class import AutoViz_Class
import pandas as pd



# Load data
df_train = pd.read_csv("pima-indians-diabetes3.csv")
num_row, num_col = df_train.shape

# Run AutoViz
AV = AutoViz_Class()
AV.AutoViz(filename='', dfte=df_train, depVar='diabetes',
           verbose=2, max_rows_analyzed=num_row,
           max_cols_analyzed=num_col)

