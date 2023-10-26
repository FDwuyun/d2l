import hashlib
import os
import tarfile
import zipfile
import requests

script_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, "..", "..", 'data')

os.makedirs(cache_dir, exist_ok=True)
print(cache_dir)
print(os.getcwd())