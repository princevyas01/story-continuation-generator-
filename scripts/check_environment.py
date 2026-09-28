#!/usr/bin/env python3
"""Environment Verification Script for Story Continuation LSTM.

Checks and displays runtime versions, hardware configuration, and directory status.
"""

import os
import sys
import platform
import psutil

def check_env():
    print("=" * 60)
    print("ENVIRONMENT & HARDWARE VERIFICATION")
    print("=" * 60)

    # Python version
    print(f"Python Version:       {sys.version.split()[0]} ({platform.architecture()[0]})")
    print(f"Python Executable:    {sys.executable}")
    print(f"Operating System:     {platform.system()} {platform.release()} ({platform.version()})")

    # NumPy
    try:
        import numpy as np
        print(f"NumPy Version:        {np.__version__}")
    except ImportError as e:
        print(f"NumPy Version:        NOT INSTALLED ({e})")

    # TensorFlow & Keras
    try:
        import tensorflow as tf
        print(f"TensorFlow Version:   {tf.__version__}")
        gpu_devices = tf.config.list_physical_devices('GPU')
        cpu_devices = tf.config.list_physical_devices('CPU')
        print(f"TensorFlow CPU Devs:  {len(cpu_devices)} device(s) detected: {[d.name for d in cpu_devices]}")
        print(f"TensorFlow GPU Devs:  {len(gpu_devices)} device(s) detected: {[d.name for d in gpu_devices]}")
    except ImportError as e:
        print(f"TensorFlow Version:   NOT INSTALLED ({e})")
        print("TensorFlow GPU Devs:  N/A")

    try:
        import keras
        print(f"Keras Version:        {keras.__version__}")
    except ImportError:
        print("Keras Version:        Exposed via TensorFlow internal or not standalone")

    # Streamlit
    try:
        import streamlit as st
        print(f"Streamlit Version:    {st.__version__}")
    except ImportError as e:
        print(f"Streamlit Version:    NOT INSTALLED ({e})")

    # PyYAML
    try:
        import yaml
        print(f"PyYAML Version:       {yaml.__version__}")
    except ImportError as e:
        print(f"PyYAML Version:       NOT INSTALLED ({e})")

    # Hardware Info (CPU & RAM)
    cpu_count_phys = psutil.cpu_count(logical=False)
    cpu_count_log = psutil.cpu_count(logical=True)
    mem = psutil.virtual_memory()
    total_ram_gb = mem.total / (1024 ** 3)
    avail_ram_gb = mem.available / (1024 ** 3)

    print(f"CPU Physical Cores:   {cpu_count_phys}")
    print(f"CPU Logical Threads:  {cpu_count_log}")
    print(f"Total Visible RAM:    {total_ram_gb:.2f} GB")
    print(f"Available RAM:        {avail_ram_gb:.2f} GB")

    # Project Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("\nProject Paths:")
    print(f"  Base Directory:     {base_dir}")
    print(f"  Configs:            {os.path.join(base_dir, 'configs')}")
    print(f"  Data Directory:     {os.path.join(base_dir, 'data')}")
    print(f"  Models Directory:   {os.path.join(base_dir, 'models')}")
    print(f"  Artifacts:          {os.path.join(base_dir, 'artifacts')}")
    print(f"  Tests:              {os.path.join(base_dir, 'tests')}")
    print("=" * 60)

if __name__ == "__main__":
    check_env()
