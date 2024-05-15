# SOMを用いてVRを計算するプログラム
# 杉井さんのコードをクラス化

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import somoclu

class VRCalculator():
    
    def __init__(self):
        self.NORM_DIVISOR = 3   # 2値化ストロークを何mmで区切るか
        self.SOM_ROWS = 100     # SOMのマップの行数
        self.SOM_COLS = 100     # SOMのマップの列数
        self.som = None         # SOMのモデル
        self.df_train = None    # 学習用ドローイングのdf
        self.df_predict = None  # 推論用ドローイングのdf
    
        return
    
    # 学習用のデータをセットする
    def set_train_data(self, df_train):
        self.df_train = df_train
        return
    
    # 推論用のデータをセットする
    def set_predict_data(self, df_predict):
        self.df_predict = df_predict
        return
    
    # SOMを学習させる                
    def train(self, initial_codebook):
        df_train = self.df_train.copy()
        stroke_bin_vector = [df[f'px_{self.NORM_DIVISOR}mm_grid_flat'] for df in df_train]
        
        random_codebook = np.random.randint(0, 2, (100, 100, 13860))
        random_codebook = random_codebook.astype(np.float32)
        self.som = somoclu.Somoclu(
            n_columns=self.SOM_COLS, 
            n_rows=self.SOM_ROWS, 
            initialcodebook = random_codebook, 
            compactsupport=False
            )
        return
    
    # SOMに推論させる
    def predict(self):
        
        return
    
    
        

def main():
    
    return

if __name__ == '__main__':
    main()

