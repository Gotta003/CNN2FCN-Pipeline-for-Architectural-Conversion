import sys

checks={
    "tkinter": lambda: str(__import__("tkinter").TkVersion),
    "pandas": lambda: __import__("pandas").__version__,
    "numpy": lambda: __import__("numpy").__version__,
    "ttkbootstrap": lambda: __import__("ttkbootstrap").__version__,
    "customtkinter": lambda: __import__("customtkinter").__version__,
    "netron": lambda: __import__("netron").__version__,
    "tkinterweb": lambda: __import__("tkinterweb").__version__,
}

GREEN="\033[0;32m"
RED="\033[0;31m"
RESET="\033[0m"
failed=[]

for pkg, ver in checks.items():
    try:
        print(f"{GREEN}{pkg:<20} {ver}{RESET}")
    except Exception as e:
        print(f"{RED}{pkg:<20} not found{RESET}")
        failed.append(pkg)

if failed:
    print(f"\n{RED}The following packages are missing: {', '.join(failed)}{RESET}")
    sys.exit(1)
else:
    print(f"\n{GREEN}All packages are installed!{RESET}")