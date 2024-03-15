#!/bin/bash
echo "Init Stable Diffusion Environment"
sudo cog run python /src/init_env.py
echo "Install Requirements"
sudo cog run pip install --no-cache-dir -r requirements.txt
echo "Download Models"
command2
