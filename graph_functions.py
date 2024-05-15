### 2022/12/4
### グラフの描画に使う関数
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

# ストロークを実際に描画する関数(用紙の中心を(0, 0)とする)
def draw_stroke(df):
    # 用紙のサイズの大きさの変更
    SCALE = 0.1
    
    plt.rcParams['font.family'] = 'Times New Roman'
    STROKE_COLOR = 'black'
    FACE_COLOR = 'white'
    STROKE_WIDTH = 0.5
    FIG_SIZE = (420*SCALE, 297*SCALE)
    fig, ax = plt.subplots(facecolor=FACE_COLOR, figsize=(FIG_SIZE[0],FIG_SIZE[1]))
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, 
                    bottom=False, left=False, right=False, top=False)
    ax.set_xlim([-210, 210])    # x方向の描画範囲を指定
    ax.set_ylim([-148.5, 148.5])    # y方向の描画範囲を指定
    ax.set_facecolor(FACE_COLOR)
    
    for x_list, y_list in zip(df['ptx'], df['pty']):
        ax.plot(x_list, y_list, color=STROKE_COLOR, linewidth=STROKE_WIDTH)
        
    plt.show()
    plt.savefig('../temp/view_algo_fintime_stroke/')
    
    return

# 2つのリストから散布図を作成する
def show_scatter(ser1, ser2, title=''):
    plt.rcParams['font.family'] = 'MS Gothic'
    color = 'purple'
    FIG_SIZE = (10, 10)
    
    fig, ax = plt.subplots(figsize=(FIG_SIZE[0],FIG_SIZE[1]))
    ax.set_title(title)
    ax.set_xlabel('ser1')
    ax.set_ylabel('ser2')
    # ax.set_xticks(np.linspace(-420, 420, 5))
    # ax.set_yticks(np.linspace(-297, 297, 5))
    ax.grid(True, alpha=0.5)
    
    ax.scatter(ser1, ser2, color=color, marker='o', alpha=1)
    ax.plot([i for i in range(0, 801)], [ i*44.1+ 9500 for i in range(0, 801)])
    plt.show()
    return

def show_connect(df):
    plt.rcParams['font.family'] = 'MS Gothic'
    COLOR = 'purple'
    STROKE_WIDTH = 1
    SCALE = 0.02
    FIG_SIZE = (420*SCALE, 297*SCALE)
    
    fig, ax = plt.subplots(figsize=(FIG_SIZE[0],FIG_SIZE[1]))
    ax.set_xlim([-210, 210])    # x方向の描画範囲を指定
    ax.set_ylim([-148.5, 148.5])    # y方向の描画範囲を指定
    
    x = [ p[0] for p in df['pre_init_vector'] ]
    y = [ p[1] for p in df['pre_init_vector'] ]
    ax.plot(x, y, color=COLOR, linewidth=STROKE_WIDTH, marker='.', ls='-', markersize=5)
    plt.show()
    return

# 極座標グラフを表示する
def show_polar(df):
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_rmax(260)
    ax.set_rticks([0, 32.5, 65, 97.5, 130, 162.5, 195, 227.5, 260])
    ax.grid(True)
    title = ''
    ax.set_title(title)
    ax.plot(df['polar_degree'], df['polar_radius'])
    plt.show()
    return

# 箱ひげ図を作成する
def boxplot_ser(ser, label, ymax):
    title = ''
    plt.rcParams['font.family'] = 'MS Gothic'
    FIG_WIDTH = 15
    FIG_HEIGHT = 30
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor='white')
    ax.set_ylim(0, ymax)
    ax.grid(True)
    ax.set_xticklabels(label)

    ax.set_title(title)
    
    ax.boxplot(ser)
    # ax.boxplot(ser, whis=(0, 100))
    
    # ax.violinplot(ser)
    
    plt.show()
    return

# ストロークを背景に薄く描画して, endまでストローク始点を描画する関数
def stroke_and_initpt(df, end, savefolder):
    plt.rcParams['font.family'] = 'Times New Roman'
    STROKE_COLOR = 'black'
    INITPT_COLOR = 'orange'
    STROKE_ALPHA = 0.1
    INITPT_ALPHA = 0.8
    FACE_COLOR = 'white'
    STROKE_WIDTH = 0.3
    SCALE = 0.02
    FIG_SIZE = (420*SCALE, 297*SCALE)
    
    fig, ax = plt.subplots(facecolor=FACE_COLOR, figsize=(FIG_SIZE[0],FIG_SIZE[1]))
    ax.set_xlim([-210, 210])    # x方向の描画範囲を指定
    ax.set_ylim([-148.5, 148.5])    # y方向の描画範囲を指定
    ax.set_facecolor(FACE_COLOR)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, 
                    bottom=False, left=False, right=False, top=False)
    
    # 背景のストローク
    for x_list, y_list in zip(df['ptx'], df['pty']):
        ax.plot(x_list, y_list, color=STROKE_COLOR, linewidth=STROKE_WIDTH, alpha=STROKE_ALPHA)
    
    # ストローク始点
    initpt_x = df['init_ptx'][0:end+1]
    initpt_y = df['init_pty'][0:end+1]
    ax.scatter(initpt_x, initpt_y, color=INITPT_COLOR, marker='o', s=30, alpha=INITPT_ALPHA)
    # ax.plot(initpt_x, initpt_y, color=INITPT_COLOR, alpha=INITPT_ALPHA, linewidth=1, marker='.', markersize=10)
    plt.savefig(savefolder + f'/frame_{end}')
    plt.show()
    return

def connect_two_scatter(x1, x2, title=''):
    plt.rcParams['font.family'] = 'MS Gothic'
    COLOR_SCATTER = 'purple'
    COLOR_CONNECT = 'gray'
    LINEWIDTH = 1
    ALPHA = 1
    FIG_SIZE = (10, 10)
    Y_MAX = 700
    
    fig, ax = plt.subplots(figsize=(FIG_SIZE[0],FIG_SIZE[1]))
    ax.set_ylim(-5, Y_MAX)
    ax.set_title(title)
    # ax.set_xlabel('x1')
    # ax.set_ylabel('x2')
    # ax.set_xticks([0, 1], ['x1', 'x2'],  fontsize=18)
    ax.set_xticks([0, 1], fontsize=18)
    
    
    for x in x1:
        ax.scatter(0, x, color=COLOR_SCATTER, marker='o', s=30, alpha=ALPHA)
    for x in x2:
        ax.scatter(1, x, color=COLOR_SCATTER, marker='o', s=30, alpha=ALPHA)
    
    for s, e in zip(x1, x2):
        ax.plot(s, e, linewidth=LINEWIDTH, color=COLOR_CONNECT)
    
    plt.show()
    return


def main():
    return

if __name__ == '__main__':
    main()