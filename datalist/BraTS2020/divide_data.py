import os
import random

random.seed(42)

base_dir = os.path.dirname(os.path.abspath(__file__))

# Read all subject IDs
with open(os.path.join(base_dir, 'train_all.list'), 'r') as f:
    all_ids = [line.strip() for line in f if line.strip()]

random.shuffle(all_ids)

n = len(all_ids)
train_split = int(n * 0.7)
val_split = int(n * 0.8)

train = sorted(all_ids[:train_split])
val = sorted(all_ids[train_split:val_split])
test = sorted(all_ids[val_split:])

for name, ids in [('train', train), ('val', val), ('test', test)]:
    with open(os.path.join(base_dir, f'{name}.list'), 'w') as f:
        for sid in ids:
            f.write(sid + '\n')
    print(f'{name}: {len(ids)} samples')

print(f'Total: {n}')
