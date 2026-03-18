import asyncio
import aiohttp
import sqlite3
import time

DB_NAME = "skyblock_auctions.db"
API_URL = "https://api.hypixel.net/v2/skyblock/auctions_ended"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    conn.execute('''CREATE TABLE IF NOT EXISTS auctions (id TEXT PRIMARY KEY, price INTEGER, item_bytes TEXT, timestamp INTEGER)''')
    conn.commit()
    return conn

async def fetch_ended(session):
    try:
        async with session.get(API_URL, timeout=15) as resp:
            return await resp.json() if resp.status == 200 else None
    except Exception:
        return None

async def run_sync():
    conn = init_db()
    last_api_update = 0
    
    async with aiohttp.ClientSession() as session:
        print("Scraper active.")
        
        while True:
            data = await fetch_ended(session)
            if not data or not data.get('success'):
                await asyncio.sleep(10)
                continue
            
            current_api_update = data.get('lastUpdated', 0)
            if current_api_update <= last_api_update:
                await asyncio.sleep(15)
                continue
            
            last_api_update = current_api_update
            ended_auctions = data.get('auctions', [])
            
            new_count = 0
            for auc in ended_auctions:
                try:
                    conn.execute("INSERT OR IGNORE INTO auctions VALUES (?,?,?,?)", 
                               (auc.get('auction_id'), auc.get('price'), auc.get("item_bytes"), auc.get("timestamp")))
                    new_count += 1
                except:
                    continue
            
            conn.commit()
            print(f"[{time.strftime('%H:%M:%S')}] Logged {new_count} sales to training data.")
            
            await asyncio.sleep(40)

try:
    asyncio.run(run_sync())
except KeyboardInterrupt:
    print("Stopped.")