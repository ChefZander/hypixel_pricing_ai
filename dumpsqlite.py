import sqlite3
import base64
import gzip
import io
import nbtlib
import numpy as np
import zlib
from tqdm import tqdm

DB_NAME = "skyblock_auctions.db"
OUTPUT_FILE = "training_data.npz"
VECTOR_SIZE = 1024 * 2 * 2 * 2 * 2

def decode_nbt(b64_string):
    data = base64.b64decode(b64_string)
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
        uncompressed_data = f.read()
        return nbtlib.File.parse(io.BytesIO(uncompressed_data))

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
    else: print("NBT is None")
    
    return np.log1p(vec)

conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM auctions")
total = cursor.fetchone()[0]

cursor.execute("SELECT item_bytes, price FROM auctions")

inputs = []
outputs = []

print(f"Processing {total} auctions...")
for item_bytes, price in tqdm(cursor, total=total):
    if price <= 0: continue
    
    vector = process_to_vector(item_bytes, VECTOR_SIZE)

    log_price = np.log10(price)
    
    inputs.append(vector)
    outputs.append(log_price)
    
conn.close()

x_data = np.array(inputs, dtype=np.float32)
y_data = np.array(outputs, dtype=np.float32)

np.savez_compressed(OUTPUT_FILE, x=x_data, y=y_data)
print(f"Saved {len(x_data)} samples to {OUTPUT_FILE}")