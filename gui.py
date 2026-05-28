from __future__ import annotations
import argparse
import sys
import tkinter
import customtkinter
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description="GUI Pipeline")
    p.add_argument("--root", default=None, help="Project root directory (default: directory containing gui.py")
    args=p.parse_args()
    root=Path(args.root).resolve() if args.root else Path(__file__).parent.resolve()
    sys.path.insert(0, str(root))
    from gui.app import App
    app=App(project_root=root)
    app.mainloop()

if __name__=="__main__":
    main()