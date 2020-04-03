#! /usr/bin/env python3
import os,sys,pickle
import tkinter as tk
from tkinter import ttk
import pandas as pd

#Root class; handles frame switching in gui
class GUI_Start(tk.Tk):
    def __init__(self,startPage,*args):
        self.root = tk.Tk.__init__(self)
        self._frame = None
        self.homedirectory = os.getcwd()
        if self.homedirectory[-1] != '/':
            self.homedirectory+='/'
        self.switch_frame(startPage,*args)

    def switch_frame(self, frame_class,*args):
        """Destroys current frame and replaces it with a new one."""
        new_frame = frame_class(self,*args)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()


class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

class InputDatasetSelectionPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        frame = ScrollableFrame(self)

        for i in range(50):
            ttk.Label(frame.scrollable_frame, text="Sample scrolling label").pack()

        frame.pack()

df = pickle.load(open('../output/all-WT.pkl','rb'))
print(pd.unique(df.index.get_level_values('Data')))
df = df.rename({'OT1-CAR_CytokineOnly_2':'OT1_CAR_CytokineOnly_2'})
print(pd.unique(df.index.get_level_values('Data')))
