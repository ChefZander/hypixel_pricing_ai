import sqlite3
import base64
import gzip
import io
import nbtlib
import numpy as np
import zlib
import os
from tqdm import tqdm

DB_NAME = "skyblock_auctions.db"
OUTPUT_FILE = "training_data.npz"
VECTOR_SIZE = 16384 

def decode_nbt(b64_string):
    try:
        data = base64.b64decode(b64_string)
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
            return nbtlib.File.parse(io.BytesIO(f.read()))
    except:
        return None

def flatten_nbt(nbt_data, prefix=""):
    items = []
    if isinstance(nbt_data, (nbtlib.Compound, dict)):
        for key, value in nbt_data.items():
            items.extend(flatten_nbt(value, prefix=f"{prefix}.{key}"))
    elif isinstance(nbt_data, (nbtlib.List, list)):
        for i, value in enumerate(nbt_data):
            items.extend(flatten_nbt(value, prefix=f"{prefix}[{i}]"))
    else:
        items.append(f"{prefix}:{nbt_data}")
    return items

def process_to_vector(item_bytes, vector_size):
    vec = np.zeros(vector_size, dtype=np.float32)
    nbt = decode_nbt(item_bytes)
    if nbt:
        tags = flatten_nbt(nbt)
        for t in tags:
            idx = zlib.adler32(t.encode('utf-8')) % vector_size
            vec[idx] += 1.0
    return np.log1p(vec)

conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM auctions WHERE price > 0")
total = cursor.fetchone()[0]

x_mmap = np.memmap('x_temp.dat', dtype='float32', mode='w+', shape=(total, VECTOR_SIZE))
y_mmap = np.memmap('y_temp.dat', dtype='float32', mode='w+', shape=(total,))

cursor.execute("SELECT item_bytes, price FROM auctions WHERE price > 0")

print(f"Processing {total} auctions to disk...")
for i, (item_bytes, price) in enumerate(tqdm(cursor, total=total)):
    x_mmap[i] = process_to_vector(item_bytes, VECTOR_SIZE)
    y_mmap[i] = np.log10(price)

print("Compressing and saving final file...")
np.savez_compressed(OUTPUT_FILE, x=x_mmap, y=y_mmap)

conn.close()
del x_mmap
del y_mmap
os.remove('x_temp.dat')
os.remove('y_temp.dat')

print(f"Done! Saved to {OUTPUT_FILE}")