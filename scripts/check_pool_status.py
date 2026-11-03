import concurrent.futures
import sys
from pathlib import Path

repo_dir = Path(r"C:\Users\Wagur1\prism-vcm")
sys.path.insert(0, str(repo_dir))

from scripts.dispatch_pool_5k import discover_pool_accounts, get_account_for_shard, check_shard_status

accounts = discover_pool_accounts(Path(r"C:\Users\Wagur1\.kaggle_pool"))

def check_one(i):
    acc = get_account_for_shard(i, accounts)
    res = check_shard_status(i, acc)
    return i, acc["username"], res["status"]

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
    results = sorted(ex.map(check_one, range(19)))

print(f"{'Shard':<8} | {'Account':<20} | Status")
print("-" * 55)
for i, user, status in results:
    clean_status = status.replace('KernelWorkerStatus.', '').replace('has status', '').strip(' "\'')
    print(f"Shard {i:02d}   | {user:<20} | {clean_status}")
