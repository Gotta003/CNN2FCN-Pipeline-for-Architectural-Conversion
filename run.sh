#!/usr/bin/env bash
chmod +x setup.sh
./setup.sh
source .tiny_env/bin/activate

python3 gui.py