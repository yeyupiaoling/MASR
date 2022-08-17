#!/bin/bash
apt update
apt install -y libsndfile-dev ffmpeg
pip install -r requirements.txt
