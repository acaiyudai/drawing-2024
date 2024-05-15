# P3:杉井さんの先行研究と同様の入力データを作成する
# 開始:2023/07/28 => 終了:2023/08/05

import pandas as pd
import numpy as np
import requests
import io
import math
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches

class PreprocessedData():
    def __init__(self, url, rotate_type):
        self.CANVAS_HEIGHT = 297
        self.CANVAS_WIDTH = 420
        self.MAX_POINT = 943
        self.NORM_DIVISOR = 3
        self.columns = [ 'time', 'alpha', 'max_points', 'ptx_anoto', 'pty_anoto', 'shape_str',
                    'pressure_style', 'len_mm_acc', 'len_anoto', 'pressure_avg', 'shape_int']
        self.BIN_HEIGHT = 420
        self.BIN_WIDTH = 420
        
        self.df_org = None
        self.df_coord = None
        self.df_unify = None
        self.df_lerp = None
        self.df_slide_expand = None
        self.df_norm_grid = None        
        self.df_dropcol = None
        self.df_drop_point = None
        self.url = url
        self.rotate_type = rotate_type
        
        # urlからデータを作成
        self.get_df(self.url)
        self.conv_str_to_coord()
        
        # 誤り
        #self.unify_coord(self.rotate_type)
        # 正しい
        self.rotate_coord()
        
        # 2024/01/16 重複する座標を削除 #
        self.remove_overlap_coord()
        ################################
        
        self.lerp_stroke()
        self.slide_expand_stroke()
        
        self.binarize_stroke()
        
        self.drop_col()
        self.drop_point()
        
        # 2024/02/21 コサイン類似度を算出 #
        self.add_cos_sim()
        self.add_cossim_org()
        #################################
        
    # URLからデータフレームを取得 & ストローク長の列を追加
    def get_df(self, url):
        content = requests.get(url, auth=('19t2003a', 'ireneRED77')).content
        df = pd.read_table(io.StringIO(content.decode('utf-8')), header = None)
        df.columns = self.columns
        st_len_mm = [l*0.3 for l in df['len_anoto']]
        df['len_mm'] = st_len_mm
        # ストローク固有のidを付与
        df_reindex = df.reset_index()
        df_reindex = df_reindex.rename(columns={'index': 'stroke_id'})
        
        self.df_org = df_reindex.copy()
    
    # 文字列座標データをリストに変換 & anotoから普通座標に変換
    def conv_str_to_coord(self):
        df = self.df_org.copy()
        x_list = []
        y_list = []
        for str_x, str_y in zip(df['ptx_anoto'], df['pty_anoto']):
            x_list.append([float(x)*0.3 for x in str_x.split(',')])
            y_list.append([float(y)*0.3 for y in str_y.split(',')])
        df['ptx_mm'] = x_list
        df['pty_mm'] = y_list
        self.df_coord = df
        
    # 杉井さん加工 1:半時計周りに回転
    # | cosΘ -sinΘ | | x |
    # | sinΘ  cosΘ | | y |
    # x' = xcosΘ - ysinΘ
    # y' = xsinΘ + ycosΘ
    def rotate_coord(self):
        df = self.df_coord.copy()
        x_unify = []
        y_unify = []
        if self.rotate_type == 0:
            pass
        elif self.rotate_type == 180:
            rotate_deg = np.deg2rad(self.rotate_type)
            sin_deg = np.sin(rotate_deg)
            cos_deg = np.cos(rotate_deg)
            for x_list, y_list in zip(df['ptx_mm'], df['pty_mm']):
                
                # 座標の中心を原点に持ってくる
                x_center = [float(x) - self.CANVAS_WIDTH / 2 for x in x_list]
                y_center = [float(y) - self.CANVAS_HEIGHT / 2 for y in y_list]
                
                # 座標の中心を原点に持ってくる
                x_rotate = [(x * cos_deg) - (y * sin_deg) 
                            for x, y in zip(x_center, y_center)]
                y_rotate = [(x * sin_deg) + (y * cos_deg) 
                            for x, y in zip(x_center, y_center)]
            
                # 座標の中心を戻して格納する
                x_unify.append([float(x) + self.CANVAS_WIDTH / 2 for x in x_rotate])
                y_unify.append([float(y) + self.CANVAS_HEIGHT / 2 for y in y_rotate])
            df['ptx_mm'] = x_unify 
            df['pty_mm'] = y_unify
        else:
            rotate_deg = np.deg2rad(self.rotate_type)
            sin_deg = np.sin(rotate_deg)
            cos_deg = np.cos(rotate_deg)
            for x_list, y_list in zip(df['ptx_mm'], df['pty_mm']):
                
                # 座標の中心を原点に持ってくる
                x_center = [float(x) - self.CANVAS_WIDTH / 2 for x in x_list]
                y_center = [float(y) - self.CANVAS_HEIGHT / 2 for y in y_list]
                
                # 座標の中心を原点に持ってくる
                x_rotate = [(x * cos_deg) - (y * sin_deg) 
                            for x, y in zip(x_center, y_center)]
                y_rotate = [(x * sin_deg) + (y * cos_deg) 
                            for x, y in zip(x_center, y_center)]
            
                # 座標の中心を戻して格納する
                x_unify.append([float(x) + self.CANVAS_HEIGHT / 2 for x in x_rotate])
                y_unify.append([float(y) + self.CANVAS_WIDTH / 2 for y in y_rotate])
            df['ptx_mm'] = x_unify 
            df['pty_mm'] = y_unify
        self.df_unify = df
    
    # 杉井さん加工 2:ストロークをMAX_POINTの数だけ線形補完し，補完後の座標を別の列に保存
    def lerp_stroke(self):
        df = self.df_unify.copy()
        x_lerp_col = []       # 線形補完後の座標を入れるリスト(要素1つ1つが座標リスト)
        y_lerp_col = []
        for st_type, len_anoto, x_org, y_org in zip(df['shape_int'], df['len_anoto'], df['ptx_mm'], df['pty_mm']):
            len_all = len_anoto * 0.3
            max_len = 0
            check_pt = 0
            x_lerp = []
            y_lerp = []
            
            # 座標数が補完数943と同じ場合はそのままにする
            if self.MAX_POINT ==  len(x_org):
                x_lerp = copy.copy(x_org)
                y_lerp = copy.copy(y_org)
            # 線種が「点」の場合は最後の座標でかさ増し？
            elif st_type == 0:
                for x in x_org:
                    x_lerp.append(x)
                for y in y_org:
                    y_lerp.append(y)
                while len(x_lerp) < self.MAX_POINT:
                    x_lerp.append(x_org[-1])
                    y_lerp.append(y_org[-1])
            # その他の場合
            else:
                for i in range(len(x_org) - 1):
                    lerp_pt = self.MAX_POINT - len(x_org) # 補完する座標数
                    len_2pt = math.sqrt(((x_org[i + 1] - x_org[i]) ** 2) + ((y_org[i + 1] - y_org[i]) ** 2)) # 隣り合う座標との距離
                    pt_cnt = math.floor(lerp_pt * (len_2pt / len_all))    #座標間の距離の和と、各座標間の距離の差から比率を計算し、何個補完するかを決める。
                    if max_len < len_2pt :
                        max_len = len_2pt
                        check_pt = len(x_lerp)
                        check_cnt = pt_cnt
                    pt_x = np.linspace(x_org[i], x_org[i + 1], pt_cnt + 2)
                    pt_y = np.linspace(y_org[i], y_org[i + 1], pt_cnt + 2)
                    ptlist_x = pt_x.tolist()              #一次元配列から標準リストに変換
                    ptlist_y = pt_y.tolist()
                    ptlist_x.pop(-1)                          #データを結合する際に重なるデータを削除する
                    ptlist_y.pop(-1)
                    x_lerp = x_lerp + ptlist_x
                    y_lerp = y_lerp + ptlist_y
                    
                    
                x_lerp.append(x_org[-1])
                y_lerp.append(y_org[-1])
                
                # データ数の調整
                if len(x_lerp) < self.MAX_POINT:
                    sa2 = self.MAX_POINT - len(x_lerp)
                    sa3 = sa2 + check_cnt
                    pt_x = np.linspace(x_lerp[check_pt], x_lerp[check_pt + check_cnt + 1], sa3 + 2)
                    pt_y = np.linspace(y_lerp[check_pt], y_lerp[check_pt + check_cnt + 1], sa3 + 2)
                    ptlist_x = pt_x.tolist()
                    ptlist_y = pt_y.tolist()
                    for i in range(check_cnt + 2) :
                        x_lerp.pop(check_pt)
                        y_lerp.pop(check_pt)
                    for i in reversed(ptlist_x) :
                        x_lerp.insert(check_pt, i)
                    for i in reversed(ptlist_y) :
                        y_lerp.insert(check_pt, i)          
                elif len(x_lerp) > self.MAX_POINT:
                    sa2 = len(x_lerp) - self.MAX_POINT
                    sa3 = check_cnt - sa2
                    pt_x = np.linspace(x_lerp[check_pt], x_lerp[check_pt + check_cnt + 1], sa3 + 2)
                    pt_y = np.linspace(y_lerp[check_pt], y_lerp[check_pt + check_cnt + 1], sa3 + 2)
                    ptlist_x = pt_x.tolist()
                    ptlist_y = pt_y.tolist()
                    for i in range(check_cnt + 2) :
                        x_lerp.pop(check_pt)
                        y_lerp.pop(check_pt)
                    for i in reversed(ptlist_x) :
                        x_lerp.insert(check_pt, i)
                    for i in reversed(ptlist_y) :
                        y_lerp.insert(check_pt, i)
                else:
                    pass
    
            x_lerp_col.append(x_lerp)
            y_lerp_col.append(y_lerp)
        df['ptx_mm_lerp'] = x_lerp_col
        df['pty_mm_lerp'] = y_lerp_col  
        self.df_lerp = df
            
    # 杉井さん加工 3:ストロークを平行移動 & 拡大
    def slide_expand_stroke(self):
        df = self.df_lerp.copy()
        x_slide_ex_col = []
        y_slide_ex_col = []
        is_point = []
        
        for st_type, x_lerp, y_lerp in zip(df['shape_int'], df['ptx_mm_lerp'], df['pty_mm_lerp']):
            x_slide_ex = []
            y_slide_ex = []
            x_min, x_max, y_min, y_max = min(x_lerp), max(x_lerp), min(y_lerp), max(y_lerp)
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_slide = [float(x) - x_min for x in x_lerp]
            y_slide = [float(y) - y_min for y in y_lerp]
            
            # ストローク座標が1点しかない場合は，拡大はせず原点にスライドするだけ
            if x_range == 0 and y_range == 0:
                x_slide_ex = x_slide
                y_slide_ex = y_slide
            # ありえないが，一応，ストロークが寸分違わず垂直の場合を考慮
            elif x_range == 0:
                y_ratio = self.BIN_HEIGHT / y_range
                x_slide_ex = [float(x) * y_ratio for x in x_slide]
                y_slide_ex = [float(y) * y_ratio for y in y_slide]
            # ありえないが，一応，ストロークが寸分違わず水平の場合を考慮
            elif y_range == 0 :
                x_ratio = self.BIN_WIDTH / x_range
                x_slide_ex = [float(x) * x_ratio for x in x_slide]
                y_slide_ex = [float(y) * x_ratio for y in y_slide]
            else :
                x_ratio = self.BIN_WIDTH / x_range
                y_ratio = self.BIN_HEIGHT / y_range
                if x_ratio <= y_ratio :
                    x_slide_ex = [float(x) * x_ratio for x in x_slide]
                    y_slide_ex = [float(y) * x_ratio for y in y_slide]
                elif x_ratio > y_ratio :
                    x_slide_ex = [float(x) * y_ratio for x in x_slide]
                    y_slide_ex = [float(y) * y_ratio for y in y_slide]
                else :
                    pass
            
            # --- ここから点の検知 --- #
            if x_range <= 3 and y_range <= 3:    # 線種が点の場合
                is_point.append(1)
            elif st_type == 0:
                is_point.append(1)
            else:
                is_point.append(0)
            # --- ここまで点の検知 --- #
                
            x_slide_ex_col.append(x_slide_ex)
            y_slide_ex_col.append(y_slide_ex)

        df['ptx_mm_slide_expand'] = x_slide_ex_col
        df['pty_mm_slide_expand'] = y_slide_ex_col
        # df['is_point'] = is_point
        self.df_slide_expand = df
        
    # 杉井さん加工 4:ストロークを2値化
    def binarize_stroke(self):
        df = self.df_slide_expand
        canvas_x_norm = int(self.BIN_WIDTH / self.NORM_DIVISOR)
        canvas_y_norm = int(self.BIN_HEIGHT / self.NORM_DIVISOR)
        canvas_vec_col = []
        canvas_vec_flat_col = []
        for st_type, x_org, y_org in zip(df['shape_int'], df['ptx_mm_slide_expand'], df['pty_mm_slide_expand']):
            canvas_norm = np.zeros((canvas_y_norm, canvas_x_norm))  # キャンバスを0埋め
            x_norm = [float(x) / self.NORM_DIVISOR for x in x_org]  # 標準化したい数「3」で割る
            y_norm = [float(y) / self.NORM_DIVISOR for y in y_org]
            for x, y in zip(x_norm, y_norm):
                px_x = math.floor(x)
                px_y = math.floor(y)
                if px_x == int(self.BIN_WIDTH / self.NORM_DIVISOR):
                    px_x = int(self.BIN_WIDTH / self.NORM_DIVISOR) - 1
                else:
                    pass
                if px_y == int(self.BIN_HEIGHT / self.NORM_DIVISOR):
                    px_y = int(self.BIN_HEIGHT / self.NORM_DIVISOR) - 1
                else:
                    pass
                canvas_norm[px_y, px_x] = 1
            
            # --- 線種が点の場合は左上の1点のみを点にする(浅井が追加した処理) ---#
            if st_type == 0:
                canvas_norm = np.zeros((canvas_y_norm, canvas_x_norm))
                canvas_norm[0, 0] = 1
            # --------------------------------------------#
            
            canvas_vec_col.append(canvas_norm)
            canvas_vec_flat = canvas_norm.flatten()
            canvas_vec_flat_col.append(canvas_vec_flat)
            
        df[f'{int(self.BIN_WIDTH / self.NORM_DIVISOR)}*{int(self.BIN_HEIGHT / self.NORM_DIVISOR)}_img'] = canvas_vec_col
        df[f'{int(self.BIN_WIDTH / self.NORM_DIVISOR)}*{int(self.BIN_WIDTH / self.NORM_DIVISOR)}_img_vector'] = canvas_vec_flat_col
        self.df_norm_grid = df
        
    # 使わない列を間引く
    def drop_col(self):
        df = self.df_slide_expand.copy()
        df = df.drop('alpha', axis=1)
        df = df.drop('max_points', axis=1)
        df = df.drop('ptx_anoto', axis=1)
        df = df.drop('pty_anoto', axis=1)
        df = df.drop('pressure_style', axis=1)
        df = df.drop('len_anoto', axis=1)
        df = df.drop('len_mm_acc', axis=1)
        df = df.drop('ptx_mm_lerp', axis=1)
        df = df.drop('pty_mm_lerp', axis=1)
        # df = df.drop('ptx_mm_slide_expand', axis=1)
        # df = df.drop('pty_mm_slide_expand', axis=1)
        self.df_drop_col = df
    
    # 点を除外
    def drop_point(self):
        df = self.df_drop_col.copy()
        df_drop_point = df.query('shape_int != 0')
        self.df_drop_point = df_drop_point
    
    # 2値化されたストロークを描画
    def draw_binstroke(self, stroke_cnt):
        df = self.df_norm_grid.copy()[stroke_cnt:stroke_cnt+1]
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

    # ストロークを描画
    def draw_stroke(self, stroke_cnt):
        df = self.df_lerp.copy()[stroke_cnt:stroke_cnt+1]
        SCALE = 0.02
        plt.rcParams['font.family'] = 'Times New Roman'
        STROKE_COLOR = 'black'
        FACE_COLOR = 'white'
        STROKE_WIDTH = 0.5
        
        if self.rotate_type == 0 or self.rotate_type ==180:
            FIG_SIZE = (420*SCALE, 297*SCALE)
        else:
            FIG_SIZE = (297*SCALE, 420*SCALE)
        
        fig, ax = plt.subplots(facecolor=FACE_COLOR, figsize=(FIG_SIZE[0],FIG_SIZE[1]))
        
        # コメントアウトで軸を消去
        # plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, 
        #                 bottom=False, left=False, right=False, top=False)
        
        if self.rotate_type == 0 or self.rotate_type ==180:
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
    def draw_allstroke(self):
        df = self.df_lerp
        SCALE = 0.02
        plt.rcParams['font.family'] = 'Times New Roman'
        STROKE_COLOR = 'black'
        FACE_COLOR = 'white'
        STROKE_WIDTH = 0.5
        
        if self.rotate_type == 0 or self.rotate_type ==180:
            FIG_SIZE = (420*SCALE, 297*SCALE)
        else:
            FIG_SIZE = (297*SCALE, 420*SCALE)
        
        fig, ax = plt.subplots(facecolor=FACE_COLOR, figsize=(FIG_SIZE[0],FIG_SIZE[1]))
        
        # コメントアウトで軸を消去
        # plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, 
        #                 bottom=False, left=False, right=False, top=False)
        
        if self.rotate_type == 0 or self.rotate_type ==180:
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
        
        
    # 重複している座標を排除
    def remove_overlap_coord(self):
        df = self.df_unify.copy()
        
        org_ptxs = df['ptx_mm']
        org_ptys = df['pty_mm']
        remove_indexs = []
        for ptxs, ptys in zip(org_ptxs, org_ptys):
            remove_index = []
            for i in range(1, len(ptxs)):
                if ptxs[i] == ptxs[i-1] and ptys[i] == ptys[i-1]:
                    remove_index.append(i)
                else:
                    pass
            remove_indexs.append(remove_index)
        
        df['remove_index'] = remove_indexs
        
        remove_overlap_ptxs = []
        remove_overlap_ptys = []
        for ptxs, ptys, remove_index in zip(org_ptxs, org_ptys, df['remove_index']):
            remove_overlap_ptx = [ ptxs[i] for i in range(len(ptxs)) if i not in remove_index]
            remove_overlap_pty = [ ptys[i] for i in range(len(ptys)) if i not in remove_index]
            remove_overlap_ptxs.append(remove_overlap_ptx)
            remove_overlap_ptys.append(remove_overlap_pty)    
        
        df['ptx_mm'] = remove_overlap_ptxs
        df['pty_mm'] = remove_overlap_ptys
        
        self.df_unify = df
        
    # コサイン類似度を求める
    
    def add_cos_sim(self):
        def calc_cos_sim(v1, v2):
            if (np.linalg.norm(v1) * np.linalg.norm(v2)) == 0:
                print('------------ exception -------------')
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        ptxs_list = self.df_drop_point['ptx_mm_slide_expand']
        ptys_list = self.df_drop_point['pty_mm_slide_expand']
        vectors_list = []
        cos_sims_list = []

        for ptxs, ptys in zip(ptxs_list, ptys_list):
            vectors = [np.array([ptxs[i] - ptxs[i-1], ptys[i] - ptys[i-1]])  for i in range(1, len(ptxs))]
            vectors_list.append(vectors)
            cos_sims = [calc_cos_sim(vectors[i], vectors[i-1]) for i in range(1, len(vectors))]
            cos_sims_list.append(cos_sims)

        self.df_drop_point['cos_sim'] = cos_sims_list
        
    def add_cossim_org(self):
        def calc_cos_sim(v1, v2):
            if (np.linalg.norm(v1) * np.linalg.norm(v2)) == 0:
                print('------------ exception -------------')
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        ptxs_list = self.df_drop_point['ptx_mm']
        ptys_list = self.df_drop_point['pty_mm']
        vectors_list = []
        cos_sims_list = []

        for ptxs, ptys in zip(ptxs_list, ptys_list):
            vectors = [np.array([ptxs[i] - ptxs[i-1], ptys[i] - ptys[i-1]])  for i in range(1, len(ptxs))]
            vectors_list.append(vectors)
            cos_sims = [calc_cos_sim(vectors[i], vectors[i-1]) for i in range(1, len(vectors))]
            cos_sims_list.append(cos_sims)

        self.df_drop_point['cos_sim_org'] = cos_sims_list
        
def main():
    return

if __name__ == '__main__':
    main()