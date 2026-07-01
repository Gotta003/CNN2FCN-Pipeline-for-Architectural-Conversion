import sys

checks={
    "tkinter": lambda: str(__import__("tkinter").TkVersion),
    "pandas": lambda: __import__("pandas").__version__,
    "numpy": lambda: __import__("numpy").__version__,
    "customtkinter": lambda: __import__("customtkinter").__version__,
    "netron": lambda: __import__("netron").__version__,
    "yaml": lambda: __import__("yaml").__version__,
    "pillow": lambda: __import__("PIL").__version__,
    "ipykernel": lambda: __import__("ipykernel").__version__,
    "torch": lambda: __import__("torch").__version__,
    "scikit-learn": lambda: __import__("sklearn").__version__,
    "matplotlib": lambda: __import__("matplotlib").__version__,
    "requests": lambda: __import__("requests").__version__,
    "soundfile": lambda: __import__("soundfile").__version__,
    "tqdm": lambda: __import__("tqdm").__version__,
    "cupy": lambda: __import__("cupy").__version__,
}

GREEN="\033[0;32m"
RED="\033[0;31m"
RESET="\033[0m"
failed=[]

for pkg, ver in checks.items():
    try:
        ver()
        print(f"{GREEN}{pkg:<20} {ver}{RESET}")
    except Exception as e:
        print(f"{RED}{pkg:<20} not found{RESET}")
        failed.append(pkg)

if failed:
    print(f"\n{RED}The following packages are missing: {', '.join(failed)}{RESET}")
    sys.exit(1)
else:
    print(f"\n{GREEN}All packages are installed!{RESET}")