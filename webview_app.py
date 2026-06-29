import os
import sys
os.environ["PYWEBVIEW_GUI"]="qt"
os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"
sys.argv.append("--disable-web-security")
sys.argv.append("--allow-running-insecure-content")

try:
    import webview
except ImportError:
    print("Error: pywebview not installed")
    sys.exit(1)
    
def main():
    if len(sys.argv)<3:
        sys.exit(1)
    url=sys.argv[1]
    title=sys.argv[2]
    webview.create_window(
        title=title,
        url=url,
        width=1000,
        height=750,
        background_color="#1A1A1A",
        easy_drag=False
    )
    webview.start()
    
if __name__=="__main__":
    main()