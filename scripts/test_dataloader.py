from src import data, tokenizer, dataloader, utils
import yaml
import time

with open("configs/test.yaml") as f:
    config = yaml.safe_load(f)

utils.set_seed(config["seed"])

print("Loading dataset...")
ds_raw = data.get_dataset("pretrain", config)

print("Loading tokenizer...")
token = tokenizer.get_tokenizer(ds_raw, config)

print("Preparing dataset...")
ds = data.prepare_dataset("pretrain", ds_raw, token, config)

print("Creating dataloaders...")
dls = dataloader.create_dataloaders("pretrain", ds, config)

train_dl = dls["train"]
print(f"num_workers={train_dl.num_workers}, prefetch_factor={train_dl.prefetch_factor}, batch_size={train_dl.batch_size}")

num_batches = 10
print(f"\nTiming {num_batches} batches...")
start = time.time()
for i, batch in enumerate(train_dl):
    elapsed = time.time() - start
    print(f"  batch {i}: shape={batch.shape}, time={elapsed:.3f}s")
    if i >= num_batches - 1:
        break

total = time.time() - start
print(f"\nTotal: {total:.3f}s for {num_batches} batches ({total/num_batches:.3f}s/batch)")
