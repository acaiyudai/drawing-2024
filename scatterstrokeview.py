### 2022/12/19
### tkiterにグラフを埋め込む

import tkinter as tk
import matplotlib.pyplot as plt
import pandas as pd

class Application(tk.Frame):
    
    SCALE = 1
    GRAPH_WIDTH = 500 * SCALE
    GRAPH_hEIGHT = 400 * SCALE
    ART_WIDTH = 500 * SCALE
    ART_HEIGHT = 400 * SCALE
    CTRL_WIDTH = 400
    CTRL_HEIGHT = 400
    
    
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title('Graph Viewer')
        self.master.geometry('800x600')
        self.create_widgets()
        
    def create_widgets(self):
        
        
        return

def main():
    root = tk.Tk()
    app = Application(master=root)
    root.mainloop()
    return

if __name__ == '__main__':
    main()

