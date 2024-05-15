### 2023/12/4
### ファイル生成, データ加工に使う関数

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox 

class ViewFunction():    
    def __init__(self):
        self.DF_HTML_PATH = r'../temp/dataframe.html'
    
    # データフレームからhtmlファイルを生成する
    def create_html(self, df):
        html = df.to_html()
        text = open(self.DF_HTML_PATH, 'w')
        text.write(html)
        text.close()
    
    # 成果物を描画する
    def draw_all_stroke(self, xs, ys, rotate_type):
        # グラフ用パラメータを定義する
        WIDTH = 420
        HEIGHT = 297
        SCALE = 0.02
        STROKE_WIDTH = 0.5
        STROKE_COLOR = 'black'
        FACE_COLOR = 'white'
        plt.rcParams['font.family'] = 'Times New Roman'
        # ===== コメントアウトで軸を消去する ===== #
        # plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, 
        #                 bottom=False, left=False, right=False, top=False)
        # ===================================== #
        
        
        # 回転方向によって用紙の向きを変える
        if rotate_type == 0 or rotate_type ==180:
            FIG_SIZE = (WIDTH*SCALE, HEIGHT*SCALE)
        else:
            FIG_SIZE = (HEIGHT*SCALE, WIDTH*SCALE)
        
        fig, ax = plt.subplots(facecolor=FACE_COLOR, figsize=(FIG_SIZE[0],FIG_SIZE[1]))
        
        if rotate_type == 0 or rotate_type ==180:
            ax.set_xlim([0, WIDTH])    # x方向の描画範囲を指定
            ax.set_ylim([0, HEIGHT])    # y方向の描画範囲を指定
        else:
            ax.set_xlim([0, HEIGHT])    # x方向の描画範囲を指定
            ax.set_ylim([0, WIDTH])    # y方向の描画範囲を指定
            
        # ストロークを描画する
        for x, y in zip(xs, ys):
            ax.plot(x, y, color=STROKE_COLOR, linewidth=STROKE_WIDTH)
        
        ax.set_facecolor(FACE_COLOR)
        
        # ===== y軸の向きをデータ収集系に揃える（必須!!!） ===== #
        ax.invert_yaxis()
        # =================================================== #
        
        plt.show()
        return
       
    # 散布図にストローク画像を出力する
    def plot_img_scatter(self, df, title, ax0_colname, ax1_colname):
        color = 'blue'
        plt.rcParams['font.family'] = 'MS Gothic'
        FIG_SIZE = (10, 10)
        fig, ax = plt.subplots(figsize=(FIG_SIZE[0],FIG_SIZE[1]))
        ax.set_xlabel(ax0_colname)
        ax.set_ylabel(ax1_colname)
        ax.set_title(title)
        # ax.set_xticks(np.linspace(-420, 420, 5))
        # ax.set_yticks(np.linspace(-297, 297, 5))
        ax.grid(True, alpha=0.5)
        for drawing_id, stroke_id, shape_int, axis0, axis1 in zip(
            df['drawing_id'], df['stroke_id'], df['shape_int'], df[ax0_colname], df[ax1_colname]
            ):
            image_path = f'../data/20240115_strokeimage_abcs_drawingid/{drawing_id}_{stroke_id}.jpg'
            image = plt.imread(image_path)
            zoom = 0.03
            oi = OffsetImage(image, zoom=zoom)
            ab = AnnotationBbox(oi, (axis0, axis1), xycoords='data', frameon=False)
            artists = []
            artists.append(ax.add_artist(ab))
            ax.scatter(axis0, axis1, color=color, marker='o', alpha=1)
        plt.show()
        return
    
    def draw_stroke(self, x, y, rotate_type):
        # グラフ用パラメータを定義する
        CANVAS_COLOR = 'white'
        STROKE_COLOR = 'black'
        STROKE_WIDTH = 3
        SCALE = 0.01
        STROKE_SCALE = 1
        plt.rcParams['font.family'] = 'MS Gothic'
        
        # 回転方向によって用紙の向きを変える
        if rotate_type == 0 or rotate_type == 180:
            X_MIN = 0
            X_MAX = 420
            Y_MIN = 0
            Y_MAX = 297
        else:
            X_MIN = 0
            X_MAX = 297
            Y_MIN = 0
            Y_MAX = 420
        
        fig, ax = plt.subplots(figsize=((X_MAX - X_MIN)*SCALE, (Y_MAX - Y_MIN)*SCALE), facecolor='white')
        ax.set_facecolor(CANVAS_COLOR)
        
        # # 座標の始点を用紙の中心(X_MAX/2, Y_MAX/2)にする
        # x_plot = [(c - x[0])*STROKE_SCALE + (X_MAX-X_MIN)/2 for c in x]
        # y_plot = [(c - y[0])*STROKE_SCALE + (Y_MAX-Y_MIN)/2 for c in y]
        
        # 座標の外包矩形が接するように拡大する
        x_slide_ex = []
        y_slide_ex = []
        x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_slide = [float(c) - x_min for c in x]
        y_slide = [float(c) - y_min for c in y]
        
        # ストローク座標が1点しかない場合は，拡大はせず原点にスライドするだけ
        if x_range == 0 and y_range == 0:
            x_slide_ex = x_slide
            y_slide_ex = y_slide
        # ありえないが，一応，ストロークが寸分違わず垂直の場合を考慮
        elif x_range == 0:
            y_ratio = (Y_MAX - Y_MIN) / y_range
            x_slide_ex = [float(x) * y_ratio for x in x_slide]
            y_slide_ex = [float(y) * y_ratio for y in y_slide]
        # ありえないが，一応，ストロークが寸分違わず水平の場合を考慮
        elif y_range == 0 :
            x_ratio = (X_MAX - X_MIN) / x_range
            x_slide_ex = [float(x) * x_ratio for x in x_slide]
            y_slide_ex = [float(y) * x_ratio for y in y_slide]
        else :
            x_ratio = (X_MAX - X_MIN) / x_range
            y_ratio = (Y_MAX - Y_MIN) / y_range
            if x_ratio <= y_ratio :
                x_slide_ex = [float(x) * x_ratio for x in x_slide]
                y_slide_ex = [float(y) * x_ratio for y in y_slide]
            elif x_ratio > y_ratio :
                x_slide_ex = [float(x) * y_ratio for x in x_slide]
                y_slide_ex = [float(y) * y_ratio for y in y_slide]
            else :
                pass
            
        x_plot = x_slide_ex
        y_plot = y_slide_ex
        
        
        # 軸の大きさ
        MARGIN_MM = 5
        ax.set_xlim(X_MIN-MARGIN_MM, X_MAX+MARGIN_MM)
        ax.set_ylim(Y_MIN-MARGIN_MM, Y_MAX+MARGIN_MM)
        
        ax.plot(x_plot, y_plot, color=STROKE_COLOR, linewidth=STROKE_WIDTH)
        ax.scatter(x_plot, y_plot, color=STROKE_COLOR, marker='o')
        # ax.scatter(x_plot[:1], y_plot[:1], color='orange', s=200)
        
        ax.invert_yaxis()
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, 
                        bottom=False, left=False, right=False, top=False)
        # plt.savefig(f'{save_folder}/{drawing_id}_{stroke_id}.png')
        plt.show()
        plt.close()
        return

def main():
    return

if __name__ == '__main__':
    main()