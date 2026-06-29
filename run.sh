#!/usr/bin/env bash
chmod +x setup.sh
./setup.sh
source .tiny_env/bin/activate

python3 src/icons/generate_icons.py
python3 gui.py