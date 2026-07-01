from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import customtkinter as ctk

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MPL=True
except ImportError:
    HAS_MPL=False
    
class LossCanvas(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")
        self._data: Dict[str, Dict[str, List]]={}
        if not HAS_MPL:
            ctk.CTkLabel(self, text="matplotlib not available", text_color="gray").pack(expand=True)
            self._canvas=None
            return
        bg="#1A1A1A"
        self._fig, self._axes=plt.subplots(2, 2, figsize=(8, 5.5), facecolor=bg)
        self._ax_loss, self._ax_acc, self._ax_f1, self._ax_gap=self._axes.flatten()
        self._setup_axes()
        self._mpl_canvas=FigureCanvasTkAgg(self._fig, master=self)
        self._mpl_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _setup_axes(self):
        bg="#1A1A1A"
        titles=["Loss", "Accuracy", "F1-Score", "Overfitting Gap"]
        for ax, title in zip(self._axes.flatten(), titles):
            ax.set_facecolor(bg)
            ax.tick_params(colors="gray", labelsize=8)
            ax.spines[:].set_color("#444")
            ax.set_title(title, color="gray", fontsize=10)
        self._fig.tight_layout(pad=2.0)
        
    def add_point(self, stage: str, epoch: int, t_loss: float, v_loss: float, t_acc: float, v_acc: float, t_f1: float, v_f1: float) -> None:
        if not HAS_MPL:
            return
        d=self._data.setdefault(stage, {"epoch": [], "t_loss": [], "v_loss": [], "t_acc": [], "v_acc": [], "t_f1": [], "v_f1": []})
        d["epoch"].append(epoch)
        d["t_loss"].append(t_loss)
        d["v_loss"].append(v_loss)
        d["t_acc"].append(t_acc*100 if t_acc<=1.0 else t_acc)
        d["v_acc"].append(v_acc*100 if v_acc<=1.0 else v_acc)
        d["t_f1"].append(t_f1)
        d["v_f1"].append(v_f1)
        self._redraw()
        
    def _redraw(self):
        colors=["#378ADD", "#1D9E75", "#EF9F27", "#E24B4A", "#7F77DD", "#D85A30", "#009688"]
        for ax in self._axes.flatten():
            ax.cla()
        self._setup_axes()
        for i, (stage, d) in enumerate(self._data.items()):
            c=colors[i%len(colors)]
            ep=d["epoch"]
            if not ep:
                continue
            #Loss Graph
            self._ax_loss.plot(ep, d["t_loss"], color=c, lw=1.5, linestyle=":", alpha=0.7, label=f"{stage} (Train)")
            self._ax_loss.plot(ep, d["v_loss"], color=c, lw=1.5, label=f"{stage} (Val)")
            #Acc Graph
            self._ax_acc.plot(ep, d["t_acc"], color=c, lw=1.5, linestyle=":", alpha=0.7, label=f"{stage} (Train)")
            self._ax_acc.plot(ep, d["v_acc"], color=c, lw=1.5, label=f"{stage} (Val)")    
            #F1 Graph
            self._ax_f1.plot(ep, d["t_f1"], color=c, lw=1.5, linestyle=":", alpha=0.7, label=f"{stage} (Train)")
            self._ax_f1.plot(ep, d["v_f1"], color=c, lw=1.5, label=f"{stage} (Val)")
            #Gap Overfitting
            gap=[tf1-vf1 for tf1, vf1 in zip(d["t_f1"], d["v_f1"])]
            self._ax_gap.plot(ep, gap, color=c, lw=1.5, label=f"{stage} Gap")
            self._ax_gap.axhline(0, color="gray", lw=1, linestyle="--", alpha=0.5)
            
        for ax in self._axes.flatten():
            if ax.has_data():
                ax.legend(fontsize=7, facecolor="#2A2A2A", labelcolor="gray", loc="best")
                
        if self._mpl_canvas:
            self._mpl_canvas.draw_idle()
    
    def clear(self) -> None:
        self._data.clear()
        if HAS_MPL:
            self._redraw()