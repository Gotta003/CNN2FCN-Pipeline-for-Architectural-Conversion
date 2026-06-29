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
        self._fig, (self._ax_loss, self._ax_acc)=plt.subplots(1, 2, figsize=(7, 2.8), facecolor=bg)
        for ax in (self._ax_loss, self._ax_acc):
            ax.set_facecolor(bg)
            ax.tick_params(colors="gray", labelsize=8)
            ax.spines[:].set_color("#444")
        self._ax_loss.set_title("Training loss", color="gray", fontsize=9)
        self._ax_acc.set_title("Val accuracy (%)", color="gray", fontsize=9)
        self._fig.tight_layout(pad=1.5)
        self._mpl_canvas=FigureCanvasTkAgg(self._fig, master=self)
        self._mpl_canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def add_point(self, stage: str, epoch: int, loss: float, val_acc: float) -> None:
        if not HAS_MPL:
            return
        d=self._data.setdefault(stage, {"epoch": [], "loss": [], "val": []})
        d["epoch"].append(epoch)
        d["loss"].append(loss)
        d["val"].append(val_acc*100)
        self._redraw()
        
    def _redraw(self):
        colors=["#378ADD", "#1D9E75", "#EF9F27", "#E24B4A", "#7F77DD", "#D85A30", "#009688"]
        self._ax_loss.cla()
        self._ax_acc.cla()
        self._ax_loss.set_title("Training loss", color="gray", fontsize=9)
        self._ax_acc.set_title("Val accuracy (%)", color="gray", fontsize=9)
        bg="#1A1A1A"
        for ax in (self._ax_loss, self._ax_acc):
            ax.set_facecolor(bg)
            ax.tick_params(colors="gray", labelsize=8)
            ax.spines[:].set_colors("#444")
        for i, (stage, d) in enumerate(self._data.items()):
            c=colors[i%len(colors)]
            if d["loss"]:
                self._ax_loss.plot(d["epoch"], d["loss"], color=c, lw=1.5, label=stage)
            if d["val"]:
                self._ax_acc.plot(d["epoch"], d["val"], color=c, lw=1.5, label=stage)
                
        if any(d["loss"] for d in self._data.values()):
            self._ax_loss.legend(fontsize=7, facecolor="#2A2A2A", labelcolor="gray")
        if any(d["val"] for d in self._data.values()):
            self._ax_acc.legend(fontsize=7, facecolor="#2A2A2A", labelcolor="gray")
        self._fig.tight_layout(pad=1.5)
        if self._mpl_canvas:
            self._mpl_canvas.draw_idle()
    
    def clear(self) -> None:
        self._data.clear()
        if HAS_MPL:
            self._redraw()