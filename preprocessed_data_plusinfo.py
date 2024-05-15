### 2023/11/04
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from preprocessed_data import PreprocessedData

class PreprocessedDataPlusInfo():
    # 初期化
    def __init__(self, url, rotate_type):
        self.DRAWING_INFO_PATH = '../data/all_drawing_info.xlsx'
        self.URL = url
        self.ROTATE_TYPE = rotate_type
        self.CANVAS_HEIGHT = 297
        self.CANVAS_WIDTH = 420
        self.MAX_POINT = 943
        self.NORM_DIVISOR = 3
        self.columns = [ 'time', 'alpha', 'max_points', 'ptx_anoto', 'pty_anoto', 'shape_str',
                    'pressure_style', 'len_mm_acc', 'len_anoto', 'pressure_avg', 'shape_int']
        self.BIN_HEIGHT = 420
        self.BIN_WIDTH = 420
        
        self.data_org = PreprocessedData(self.URL, self.ROTATE_TYPE)
        self.df_addinfo = self.add_drawing_info()
        
    # 描画時の情報を追加
    def add_drawing_info(self):
        df_addinfo = self.data_org.df_drop_point.copy()
        df_drawing_info = pd.read_excel(self.DRAWING_INFO_PATH)
        for i, url in enumerate(df_drawing_info['url']):
           if url == self.URL:
               drawer_index = i
        
        df_drawer = df_drawing_info[drawer_index:drawer_index+1]
        
        len_df_addinfo = len(df_addinfo)
        
        drawing_id = df_drawing_info['drawing_id'][drawer_index]
        year = df_drawing_info['year'][drawer_index]
        month = df_drawing_info['month'][drawer_index]
        day = df_drawing_info['day'][drawer_index]
        motif = df_drawing_info['motif'][drawer_index]
        times = df_drawing_info['times'][drawer_index]
        name = df_drawing_info['name'][drawer_index]
        
        df_addinfo['drawing_id'] = [drawing_id for i in range(len_df_addinfo)]
        df_addinfo['year'] = [year for i in range(len_df_addinfo)]
        df_addinfo['month'] = [month for i in range(len_df_addinfo)]
        df_addinfo['day'] = [day for i in range(len_df_addinfo)]
        df_addinfo['motif'] = [motif for i in range(len_df_addinfo)]
        df_addinfo['times'] = [times for i in range(len_df_addinfo)]
        df_addinfo['name'] = [name for i in range(len_df_addinfo)]
        
        return df_addinfo
    
        # ストロークを描画
    def draw_stroke(self, stroke_cnt):
        df = self.df_addinfo.copy()[stroke_cnt:stroke_cnt+1]
        SCALE = 0.02
        plt.rcParams['font.family'] = 'Times New Roman'
        STROKE_COLOR = 'black'
        FACE_COLOR = 'white'
        STROKE_WIDTH = 0.5
        
        if self.ROTATE_TYPE == 0 or self.ROTATE_TYPE ==180:
            FIG_SIZE = (420*SCALE, 297*SCALE)
        else:
            FIG_SIZE = (297*SCALE, 420*SCALE)
        
        fig, ax = plt.subplots(facecolor=FACE_COLOR, figsize=(FIG_SIZE[0],FIG_SIZE[1]))
        
        # コメントアウトで軸を消去
        # plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, 
        #                 bottom=False, left=False, right=False, top=False)
        
        if self.ROTATE_TYPE == 0 or self.ROTATE_TYPE ==180:
            ax.set_xlim([0, 420])    # x方向の描画範囲を指定
            ax.set_ylim([0, 297])    # y方向の描画範囲を指定
        else:
            ax.set_xlim([0, 297])    # x方向の描画範囲を指定
            ax.set_ylim([0, 420])    # y方向の描画範囲を指定
        for x_list, y_list in zip(df['ptx_mm'], df['pty_mm']):
            ax.plot(x_list, y_list, color=STROKE_COLOR, linewidth=STROKE_WIDTH)
        
        ax.set_facecolor(FACE_COLOR)
        ax.invert_yaxis()
        plt.show()
        
    # 成果物を描画
    def draw_allstroke(self, file_name):
        df = self.df_addinfo
        SCALE = 0.02
        plt.rcParams['font.family'] = 'Times New Roman'
        STROKE_COLOR = 'black'
        FACE_COLOR = 'white'
        STROKE_WIDTH = 0.2
        
        if self.ROTATE_TYPE == 0 or self.ROTATE_TYPE ==180:
            FIG_SIZE = (420*SCALE, 297*SCALE)
        else:
            FIG_SIZE = (297*SCALE, 420*SCALE)
        
        fig, ax = plt.subplots(facecolor=FACE_COLOR, figsize=(FIG_SIZE[0],FIG_SIZE[1]))
        
        ### コメントアウトで軸を消去 ###
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, 
                        bottom=False, left=False, right=False, top=False)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ##############################
        
        if self.ROTATE_TYPE == 0 or self.ROTATE_TYPE ==180:
            ax.set_xlim([0, 420])    # x方向の描画範囲を指定
            ax.set_ylim([0, 297])    # y方向の描画範囲を指定
        else:
            ax.set_xlim([0, 297])    # x方向の描画範囲を指定
            ax.set_ylim([0, 420])    # y方向の描画範囲を指定
        for x_list, y_list in zip(df['ptx_mm'], df['pty_mm']):
            ax.plot(x_list, y_list, color=STROKE_COLOR, linewidth=STROKE_WIDTH)
        
        ax.set_facecolor(FACE_COLOR)
        ax.invert_yaxis()
        
        # 保存用
        # plt.savefig(file_name)
        
        plt.show()
        
    # 2値化されたストロークを描画
    def draw_binstroke(self, stroke_cnt):
        df = self.df_addinfo.copy()[stroke_cnt:stroke_cnt+1]
        SCALE = 0.05
        plt.rcParams['font.family'] = 'Times New Roman'
        STROKE_COLOR = '#B9FF00'
        FACE_COLOR = '#2C0251'
        FIG_SIZE = (self.BIN_WIDTH / self.NORM_DIVISOR*SCALE, self.BIN_HEIGHT / self.NORM_DIVISOR*SCALE)
        fig, ax = plt.subplots(facecolor=FACE_COLOR, figsize=(FIG_SIZE[0],FIG_SIZE[1]))
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, 
                        bottom=False, left=False, right=False, top=False)
        ax.set_xlim([0, self.BIN_WIDTH / self.NORM_DIVISOR])    # x方向の描画範囲を指定
        ax.set_ylim([0, self.BIN_HEIGHT / self.NORM_DIVISOR])    # y方向の描画範囲を指定
        ax.set_facecolor(FACE_COLOR)
        ax.invert_yaxis()
        
        for px_grid in df[f'{int(self.BIN_WIDTH / self.NORM_DIVISOR)}*{int(self.BIN_HEIGHT / self.NORM_DIVISOR)}_img']:
            for x in range(int(self.BIN_HEIGHT / self.NORM_DIVISOR)):
                for y in range(int(self.BIN_HEIGHT / self.NORM_DIVISOR)):
                    if px_grid[y, x] == 1:
                        r = patches.Rectangle(xy=(x, y), width=1, height=1, fc=STROKE_COLOR, ec=STROKE_COLOR, fill=True)
                        ax.add_patch(r)
        plt.show() 

def main():
    return

if __name__ == '__main__':
    main()