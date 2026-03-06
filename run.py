"""
NeuraPilot — Launcher
=====================
Run from inside the neurapilot/ folder:

    cd /path/to/neurapilot
    streamlit run run.py
"""
import sys, os

HERE  = os.path.dirname(os.path.abspath(__file__))
INNER = os.path.join(HERE, "neurapilot")

for p in [HERE, INNER]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(INNER)
exec(open(os.path.join(INNER, "ui_streamlit.py")).read())
