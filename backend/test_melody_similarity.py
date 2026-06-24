"""
Melody similarity analysis script.

Runs the full pipeline on any two audio files and logs:
  - latency for each step
  - every pairwise window comparison, sorted by similarity
  - detected matches above threshold
"""

import os
import sys
import time
import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))

# INPUT_AUDIO  = os.path.join(_HERE, "..", "public", "data", "dragonBoyLofi.wav")
# INPUT_AUDIO  = os.path.join(_HERE, "..", "public", "data", "mono_poly.mp3")
INPUT_AUDIO = os.path.join(_HERE, "..", "public", "data", "piano1.wav")
OUTPUT_AUDIO = os.path.join(_HERE, "..", "public", "data", "piano2.wav")

WINDOW_LEN_MS = 1000.0
HOP_LEN_MS    = 500.0
SIM_THRESHOLD = 0.5