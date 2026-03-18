import torch
import torch.nn as nn
import numpy as np
import base64
import gzip
import io
import nbtlib
import zlib

VECTOR_SIZE = 16384 

class ResBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.BatchNorm1d(size)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.net(x))

class SkyblockPriceNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.initial = nn.Sequential(nn.Linear(input_size, 1024), nn.ReLU())
        self.res1 = ResBlock(1024)
        self.res2 = ResBlock(1024)
        self.res3 = ResBlock(1024) 
        self.output = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.ReLU(), 
            nn.Linear(256, 1)
        )
    def forward(self, x):
        x = self.initial(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.output(x)

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
    else: print("NBT is None?")
    return np.log1p(vec)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    checkpoint = torch.load("skyblock_model_v2.pth", map_location=device)
    
    y_mean = checkpoint['y_mean']
    y_std = checkpoint['y_std']
    v_size = checkpoint.get('vector_size', 16384)
    
    model = SkyblockPriceNet(input_size=v_size).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f">>> Model v2 loaded.")
    print(f">>> Norm Stats: Mean={y_mean:.2f}, Std={y_std:.2f}")
except Exception as e:
    print(f"Could not load model: {e}")
    exit()

print("\n" + "="*50)
print("God Potion Test Run")
vector = process_to_vector("H4sIAAAAAAAA/11Vy47jxhVlt8dJd8NBsjCy8EoBZkd0RFHUgwFmQfFN8SVSFCVujCKr+JCKjxZJSdQ/5A+y702+or8lf5Bt4JjqGWNsEygU6t5T55xbBdZ9IohH4i57Igji7p64z+DdP++Ib/myLZq7J+KbBiSPxAdURClx+74hHpUMIgmDpO6XPz0RT+6hxdg6F+j4QNyrkPjIjGOaiSL4PKUj+nk8BtRzGLLgeRTGs3hMhVOE0B3xl7SrsgvCdlm1GDQI9mpP9rGs0LHJUP1IPDTo0rRHVL9beyAe3SwpwC1y/29b1UYvZL6xzma+qZkriiM5oDl3UbNd6kSV4QiLNBXo/QYWbTYUT+yCvroelmVF8USZdaMSULbjSo0usQuDrBmKLbzDxN7jzuuKiWdPR6IhWTLA4my+wCLrDqsuak4HJW5WtQ7mKgaeNFlGkkA7E1xX8ZlejenQMYM4Ya/GOtDnulbvC06WlGraolM5ky0RztPdCuSJGpCiOTv5YKNjyp0tlXlS2eqBHw5NQ9gWmwCWJXLckf6SqA4TZYvwjBXuYBqOiIx2s+MmcYzUoRoEzGTu1pD1J7MxAqTpOXnFQl/vqikb8RkFC5711yxd5Hx1dYbkpjgwO2/OXbqAx0M6VHw7Jm3veKz2+bW4HLK5BIFR8K7OK7a9d+FVi0NuokkcSxbIMJj0eg4FxjoYJUyCJchtaKuZMlkAaicWotLMFhSUVtCQLsnFUY4O2YyFLrDPXRWpQqa88PIUq90YUseTteJmQZUflhBUlzXnSCdreapNfNymEsZprg+ngJ45L7lZbGLn4NhWftK1y47dTw6GWM12PlispoyYVb7mS7yfckNf3HnkiqSuvjrXHEVuwtwaJnywthOS5Mm0XPhzj5fOLr0HLwlzWlDnqT53ZjGfv6xVft2wWszMSnJLKmObBNOmu1i8mPmLBWu/tLP9JChDlR3NloljF6ShdtJwtrw0cMYIiyNNb6UkuLoxbptNKFDoXC1HyacH4tsNwC26+w86l4nKaxTwRzgaO2m45TJVKBNjvbtYQjIyr2o/PMbszkuV57JI0U5BjuvAwwc146Yqr1LB3qODdUJbwoG2ZI8KfPG8y829QRu0eXWweV2NLQHuLV6t+YxL1GLRhXRQhfLG2vW6n3k0YeOZ8loQR2r2RYvGRZhLFNxq2Ms3F+jjLvBX7/6goo0C9zMOyhsGKpsu2Brvuc819TrY1F2P+W3sht9q9RfNEirO2crmp19xtKGP28A3u50fUHo+wZBnqWCbvvvQr7uzsVfpQHD2hr+6GnKQBb6Gd7SErbUxCWRxYggcbQgmNmXz0Odx0J+jtdZSw9cyg/Yuhr+b7HxjYuz74YuU6YuZzn31F/obauc7KZTF33u/3VUT0g4OeTWxstsZXarQVZe/5ONV+XVWqGW8+vSpf6ueiD/CrK4w6PqXUy+P6KEPfkd8fHud8WVRtzkaNGlWD6qyycpi0JSDI4pQdkIDUBDDHgXqujw2OSqaQRn3sDprbtm3V/hlC+r//qipe+jfeuq/9rPQHsEt9Y8eBWgmHVA58f3gtiBHdDq4IVAMWtwQP/w6GnI4SlHeDXR0Qrjn+nuPXKdoAL/w3QzIJRzY78L1IALFIESDugHRAcGb+g9vr8jJkrR5jnAWHW7lRJ+rvGX/9PYavb1i1xZ5ldMfiA8myNF79Ctrf2B/Fi/NEXBNc8zCtkH1w60REd/JlvCjba1Vy/yR7ve2bR/8SDMIRRE9eqbGIfvMUEzfXibR+DmkAAsoGMUxPf5APDZZjnqbedU3j3/976f//5cg7ok/CCAHCerbGPEzy6iqvfgGAAA=", v_size)
    
if np.any(vector):
    tensor_in = torch.from_numpy(vector).unsqueeze(0).to(device)
    with torch.no_grad():
        z_score = model(tensor_in).item()
        
        log_price = (z_score * y_std) + y_mean
        
        price = 10 ** log_price
        
    print(f"\nPREDICTED VALUE: {price:,.0f} Coins")
else:
    print("Failed to process item bytes.")


while True:
    print("\n" + "="*50)
    item_bytes = input("Paste item_bytes (or 'q'): ").strip()
    if item_bytes.lower() == 'q': break
    
    vector = process_to_vector(item_bytes, v_size)
    
    if np.any(vector):
        tensor_in = torch.from_numpy(vector).unsqueeze(0).to(device)
        with torch.no_grad():
            z_score = model(tensor_in).item()
            
            log_price = (z_score * y_std) + y_mean
            
            price = 10 ** log_price
            
        print(f"\nPREDICTED VALUE: {price:,.0f} Coins")
    else:
        print("Failed to process item bytes.")