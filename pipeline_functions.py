import numpy as np
import pandas as pd
import math
# 有効桁数統一用 #
import sigfig
from decimal import *
################

# 使い方
# - PipelineFunctionsクラスに前処理関数を追記していく
# 原則
# - ライブラリのようにクラス内の関数だけを引っ張り出して使用する
# (例:df = pd.read_csv('data.tsv', delimiter='\t', index_col=0)
#     pf = PipelineFunctions()
#     df_rotate = pf.add_rotated_coord(df, rotate_type)
# )
# - クラス内変数は設定しない
# - 前処理関数内で定数を設定したいときは，関数内変数を使用する
# - 解析にあたっての必須処理には必須処理と明記する

class PipelineFunctions():
    def __init__(self):
        return
    
    # ===== 必須処理 ここから ===== #
    # tsvファイルの読み込み(tsvファイルへのパス)
    def read_tsv(self, path):
        return pd.read_csv(path, delimiter='\t', index_col=0)
    
    # tsvファイルに文字列で格納された座標をリストに変換する(対象df)
    def conv_str_to_coord(self, df):
        x = [[float(val) for val in str_x.replace('[', '').replace(']', '').split(',')] for str_x in df['ptx_mm_org']]
        y = [[float(val) for val in str_y.replace('[', '').replace(']', '').split(',')] for str_y in df['pty_mm_org']]
        df['ptx_mm_org'] = x
        df['pty_mm_org'] = y
        return df
    
    # 連続して重複する座標を削除する(対象df)
    def remove_overlap_coord(self, df):
        anoto_x = [[float(val) for val in str_x.split(',')] for str_x in df['ptx_anoto']]
        anoto_y = [[float(val) for val in str_y.split(',')] for str_y in df['pty_anoto']]
        
        remove_indexes = []
        for x, y in zip(anoto_x, anoto_y):
            remove_index = []
            for i in range(1, len(x)):
                if x[i] == x[i-1] and y[i] == y[i-1]:
                    remove_index.append(i)
                else:
                    pass
            remove_indexes.append(remove_index)
        
        df['remove_index'] = remove_indexes
        
        org_x = df['ptx_mm_org']
        org_y = df['pty_mm_org']
        rm_overlap_xs = []
        rm_overlap_ys = []
        for x, y, remove_index in zip(org_x, org_y, df['remove_index']):
            rm_overlap_x = [ x[i] for i in range(len(x)) if i not in remove_index]
            rm_overlap_y = [ y[i] for i in range(len(y)) if i not in remove_index]
            rm_overlap_xs.append(rm_overlap_x)
            rm_overlap_ys.append(rm_overlap_y)    
        
        df['ptx_mm_rmoverlap'] = rm_overlap_xs
        df['pty_mm_rmoverlap'] = rm_overlap_ys
        
        # df['ptx_mm_rmoverlap'] = df['ptx_mm_org']
        # df['pty_mm_rmoverlap'] = df['pty_mm_org']
        
        return df
    
    # 座標を半時計周りに回転した座標の列を追加する(対象df, 回転角)
    def rotate_coord(self, df, rotate_type):
        
        # ----------------------------------- #
        # | cosΘ -sinΘ | | x |
        # | sinΘ  cosΘ | | y |
        # x' = xcosΘ - ysinΘ
        # y' = xsinΘ + ycosΘ
        # ----------------------------------- #
        
        CANVAS_WIDTH = 420
        CANVAS_HEIGHT = 297
        unified_x = []
        unified_y = []
        if rotate_type == 0:
            df['rotated_x'] = df['ptx_mm_rmoverlap']
            df['rotated_y'] = df['pty_mm_rmoverlap']
            pass
        elif rotate_type == 180:
            rotate_deg = np.deg2rad(rotate_type)
            sin_deg = np.sin(rotate_deg)
            cos_deg = np.cos(rotate_deg)
            for org_x, org_y in zip(df['ptx_mm_rmoverlap'], df['pty_mm_rmoverlap']):
                # 座標の中心を原点に持ってくる
                centered_x = [float(x) - CANVAS_WIDTH / 2 for x in org_x]
                centered_y = [float(y) - CANVAS_HEIGHT / 2 for y in org_y]
                # 回転させる
                rotated_x = [(x * cos_deg) - (y * sin_deg) 
                            for x, y in zip(centered_x, centered_y)]
                rotated_y = [(x * sin_deg) + (y * cos_deg) 
                            for x, y in zip(centered_x, centered_y)]
                # 座標の中心を戻して格納する
                unified_x.append([float(x) + CANVAS_WIDTH / 2 for x in rotated_x])
                unified_y.append([float(y) + CANVAS_HEIGHT / 2 for y in rotated_y])
            df['rotated_x'] = unified_x 
            df['rotated_y'] = unified_y
        else:
            rotate_deg = np.deg2rad(rotate_type)
            sin_deg = np.sin(rotate_deg)
            cos_deg = np.cos(rotate_deg)
            for org_x, org_y in zip(df['ptx_mm_rmoverlap'], df['pty_mm_rmoverlap']):
                
                # 座標の中心を原点に持ってくる
                centered_x = [float(x) - CANVAS_WIDTH / 2 for x in org_x]
                centered_y = [float(y) - CANVAS_HEIGHT / 2 for y in org_y]
                # 回転させる
                rotated_x = [(x * cos_deg) - (y * sin_deg) 
                            for x, y in zip(centered_x, centered_y)]
                rotated_y = [(x * sin_deg) + (y * cos_deg) 
                            for x, y in zip(centered_x, centered_y)]
                # 座標の中心を戻して格納する
                unified_x.append([float(x) + CANVAS_HEIGHT / 2 for x in rotated_x])
                unified_y.append([float(y) + CANVAS_WIDTH / 2 for y in rotated_y])
            df['rotated_x'] = unified_x
            df['rotated_y'] = unified_y
            
        return df
    
    # ストロークの大きさをそろえた後の座標の列を追加(対象df)
    def unify_stroke_size(self, df):
        UNI_SIZE_WIDTH = 420
        UNI_SIZE_HEIGHT = 420
        unified_x = []
        unified_y = []
        
        for org_x, org_y in zip(df['rotated_x'], df['rotated_y']):
            slided_expanded_x = []
            slided_expanded_y = []
            x_min, x_max, y_min, y_max = min(org_x), max(org_x), min(org_y), max(org_y)
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_slide = [float(x) - x_min for x in org_x]
            y_slide = [float(y) - y_min for y in org_y]
            
            # ストローク座標が1点しかない場合は，拡大はせず原点にスライドするだけ
            if x_range == 0 and y_range == 0:
                slided_expanded_x = x_slide
                slided_expanded_y = y_slide
            # ありえないが，一応，ストロークが寸分違わず垂直の場合を考慮
            elif x_range == 0:
                y_ratio = UNI_SIZE_HEIGHT / y_range
                slided_expanded_x = [float(x) * y_ratio for x in x_slide]
                slided_expanded_y = [float(y) * y_ratio for y in y_slide]
            # ありえないが，一応，ストロークが寸分違わず水平の場合を考慮
            elif y_range == 0 :
                x_ratio = UNI_SIZE_WIDTH / x_range
                slided_expanded_x = [float(x) * x_ratio for x in x_slide]
                slided_expanded_y = [float(y) * x_ratio for y in y_slide]
            else :
                x_ratio = UNI_SIZE_WIDTH / x_range
                y_ratio = UNI_SIZE_HEIGHT / y_range
                if x_ratio <= y_ratio :
                    slided_expanded_x = [float(x) * x_ratio for x in x_slide]
                    slided_expanded_y = [float(y) * x_ratio for y in y_slide]
                elif x_ratio > y_ratio :
                    slided_expanded_x = [float(x) * y_ratio for x in x_slide]
                    slided_expanded_y = [float(y) * y_ratio for y in y_slide]
                else :
                    pass
                
            unified_x.append(slided_expanded_x)
            unified_y.append(slided_expanded_y)

        df['size_unified_x'] = unified_x
        df['size_unified_y'] = unified_y
        
        return df
    
    # 必須処理をまとめて行う
    def get_normalized_data(self, path, rotate_type):
        org_df = self.read_tsv(path)
        conv_str_df = self.conv_str_to_coord(org_df)
        rm_overlap_df = self.remove_overlap_coord(conv_str_df)
        rotated_df = self.rotate_coord(rm_overlap_df, rotate_type)
        size_unified_df = self.unify_stroke_size(rotated_df)
        return size_unified_df
    
    # 有効桁数を統一する(有効桁数を統一したいリスト)
    def unify_sigdig(self, l, sigdig):
        return [round(val, sigdig) for val in l]
    # ===== 必須処理 ここまで ===== #
    
    # ===== 特徴量抽出処理 ここから ===== #
    
    # ストローク外包矩形が3mm × 3mm以下の線を「点」として取り除く
    def remove_3x3area(self, df):
        def is_in_3x3area(x, y):
            x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)
            x_range = x_max - x_min
            y_range = y_max - y_min
            _is_in_3x3area = 1 if x_range <= 3 and y_range <=3 else 0
            return _is_in_3x3area
        
        df['is_in_3x3area'] = [is_in_3x3area(x, y) for x, y in zip(df['rotated_x'], df['rotated_y'])]
        return df.query('is_in_3x3area == 0')
    
    # 既存の線種判定で「点」ラベルが付いたストロークを取り除く
    def remove_point(df):
        return df.query('shape_int != 0')
    
    # 座標をオーバーサンプリングする(x, y, サンプリング座標数)
    # 隣合う座標どうしの線分が長いところの中点から補間する
    def get_oversampled_coord(self, x, y, sample_size):
        sample_rest_size = sample_size - len(x)
        sampled_x = x.copy()
        sampled_y = y.copy()
        
        for i in range(sample_rest_size):
            max_len = 0
            max_len_index = 0 # max_len_index の手前に補間座標を追加する
            for j in range(1, len(sampled_x)):
                segment_len = math.sqrt((sampled_x[j] - sampled_x[j-1])**2 + (sampled_y[j] - sampled_y[j-1])**2)
                max_len_index = j if max_len < segment_len else max_len_index
                max_len = segment_len if max_len < segment_len else max_len
            
            center_x = round((sampled_x[max_len_index] + sampled_x[max_len_index-1]) / 2, 4)
            center_y = round((sampled_y[max_len_index] + sampled_y[max_len_index-1]) / 2, 4)
            sampled_x.insert(max_len_index, center_x)
            sampled_y.insert(max_len_index, center_y)
        return sampled_x, sampled_y
    
    # 座標をダウンサンプリングする(x, y, サンプリング後の座標数)
    # 座標の順番が等間隔になるように抽出する
    def get_downsampled_coord_by_coordnum(self, x, y, sample_size):
        # リサンプリングする要素のインデックスを取得する
        org_size = len(x)
        sample_index = []
        space  = (org_size - 1) // (sample_size - 1)       # リサンプリングの間隔
        rest_size = (org_size - 1) % (sample_size - 1)
        index = 0
        for i in range(sample_size):
            sample_index.append(index)
            index += space
            if rest_size > 0:
                index += 1
                rest_size -= 1
                
        # インデックスをもとにリストから要素を抽出
        sampled_x = [val for i, val in enumerate(x) if i in sample_index]
        sampled_y = [val for i, val in enumerate(y) if i in sample_index]

        return sampled_x, sampled_y
    
    # 隣合う座標どうしのコサイン類似度が高い線分の組み合わせを見つけ
    # その線分を連結させている座標を優先的に取り除く
    # 一斉に取り除くアルゴリズム
    def get_downsampled_coord_by_cossim(self, x, y, sample_size):
        # コサイン類似度を計算する
        def calc_cos_sim(v1, v2):
            if (np.linalg.norm(v1) * np.linalg.norm(v2)) == 0:
                print('------------ exception -------------')
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        # 取り除く座標の数
        cos_sims = []
        for j in range(2, len(x)):
            first_vec = np.array([x[j-1] - x[j-2], y[j-1] - y[j-2]])
            second_vec = np.array([x[j] - x[j-1], y[j] - y[j-1]])
            cos_sims.append(calc_cos_sim(first_vec, second_vec))
        
        # コサイン類似度の上位のみを除外する
        is_cossim_higher = [False for i in range(len(cos_sims))]
        
        # 始点と終点の数だけ，サンプリング座標数を減らす
        for i in range(sample_size-2):
            # 昇順に並べ替えたリスト
            sorted_cossims = sorted(cos_sims, reverse=False)
            # i番目に小さい値のインデックス
            i_min = sorted_cossims[i]
            extract_index = cos_sims.index(i_min)
            is_cossim_higher[extract_index] = True

        sampled_x = [x[i+1] for i in range(len(cos_sims)) if is_cossim_higher[i]]
        sampled_y = [y[i+1] for i in range(len(cos_sims)) if is_cossim_higher[i]]
        sampled_x.insert(0, x[0])
        sampled_x.append(x[-1])
        sampled_y.insert(0, y[0])
        sampled_y.append(y[-1])
        
        return sampled_x, sampled_y
    
    # 隣合う座標どうしのコサイン類似度が高い線分の組み合わせを見つけ
    # その線分を連結させている座標を優先的に取り除く
    # 一つずつ取り除くアルゴリズム
    def get_downsampled_coord_by_cossim_sequentially(self, x, y, sample_size):
        # コサイン類似度を計算する
        def calc_cos_sim(v1, v2):
            if (np.linalg.norm(v1) * np.linalg.norm(v2)) == 0:
                print('------------ exception -------------')
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        # 取り除く座標の数
        remove_size = len(x) - sample_size
        sampled_x = x.copy()
        sampled_y = y.copy()
        
        for i in range(remove_size):
            cos_sims = []
            for j in range(2, len(sampled_x)):
                first_vec = np.array([sampled_x[j-1] - sampled_x[j-2], sampled_y[j-1] - sampled_y[j-2]])
                second_vec = np.array([sampled_x[j] - sampled_x[j-1], sampled_y[j] - sampled_y[j-1]])
                cos_sims.append(calc_cos_sim(first_vec, second_vec))
            # コサイン類似度の最大値のインデックスを取得する
            # コサイン類似度の最大値が複数ある場合は最初に見つかった最大値のインデックスを取得する
            remove_index = cos_sims.index(max(cos_sims)) + 1
            sampled_x.pop(remove_index)
            sampled_y.pop(remove_index)
            
        return sampled_x, sampled_y
    
    # 座標をダウン/オーバーサンプリングする(x, y, サンプリング後の座標数)
    # 座標の長さが等間隔になるように抽出する
    def get_sampled_coord_vector_by_length(self, x, y, sample_size):
        rest_size = sample_size - 2
        space_size = sample_size - 1
        # ストロークの座標間のすべての線分の長さを格納したリスト
        segment_st_length = [math.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2) for i in range(1, len(x))]
        # ストロークの長さ
        st_length = sum(segment_st_length)
        # ストロークの長さを間隔の数で割った長さ
        space_length = st_length / space_size
        space_lengths = [space_length * i for i in range(1, rest_size+1)]
        
        # 抽出する座標を計算する
        current_len_acc = 0
        previous_len_acc = 0
        sampled_xs = []
        sampled_ys = []
        
        # 抽出した座標の勾配ベクトル
        unit_vec_xs = []
        unit_vec_ys = []
        
        # 検算用 #
        # lerp_xs = x.copy()
        # lerp_ys = y.copy()
        # is_org = [True for i in range(len(x))]
        #########
        
        sample_cnt = 0
        for sl in space_lengths:
            current_len_acc = 0
            previous_len_acc = 0
            for i in range(1, len(x)):
                current_len = math.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
                current_len_acc += current_len
                
                if sl <= current_len_acc:
                    sl_carry_over = sl - previous_len_acc
                    sampled_x = x[i-1] + (sl_carry_over * (x[i] - x[i-1])) / current_len
                    sampled_y = y[i-1] + (sl_carry_over * (y[i] - y[i-1])) / current_len
                    sampled_xs.append(sampled_x)
                    sampled_ys.append(sampled_y)
                    
                    # 直前のオリジナル座標から抽出した座標までの単位ベクトルを保存
                    sampled_vec_x = sampled_x - x[i-1]
                    sampled_vec_y = sampled_y - y[i-1]
                    norm = math.sqrt((sampled_x - x[i-1])**2 + (sampled_y - y[i-1])**2)
                    
                    unit_vec_xs.append(sampled_vec_x / norm)
                    unit_vec_ys.append(sampled_vec_y / norm)
                    
                    # 検算用
                    # lerp_xs.insert(i+sample_cnt, sampled_x)
                    # lerp_ys.insert(i+sample_cnt, sampled_y)
                    # is_org.insert(i+sample_cnt, False)
                    #########
                    
                    break
                else:
                    pass
                previous_len_acc += current_len
            # 検算用
            # sample_cnt += 1
            #########
        
        # 元のリストの最初と最後の要素を追加する
        sampled_xs.insert(0, x[0])
        sampled_xs.append(x[-1])
        sampled_ys.insert(0, y[0])
        sampled_ys.append(y[-1])
        
        return  sampled_xs, sampled_ys, unit_vec_xs, unit_vec_ys
    
    # コサイン類似度を計算する(対象リスト)
    def get_cossim(self, xs, ys):
        def calc_cos_sim(v1, v2):
            if (np.linalg.norm(v1) * np.linalg.norm(v2)) == 0:
                print('------------ exception -------------')
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        vectors_list = []
        cos_sims_list = []

        for x, y in zip(xs, ys):
            vectors = [np.array([x[i] - x[i-1], y[i] - y[i-1]]) for i in range(1, len(x))]
            vectors_list.append(vectors)
            cos_sims = [calc_cos_sim(vectors[i], vectors[i-1]) for i in range(1, len(vectors))]
            cos_sims_list.append(cos_sims)
            
        return cos_sims_list
    
    def get_minmax_scaled_data(self, data):
        return [(val - min(data)) / (max(data) - min(data)) for val in data]
    # ===== 特徴量抽出処理 ここまで ===== #
    
def main():    
    return

if __name__ == '__main__':
    main()