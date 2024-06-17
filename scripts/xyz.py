from datasets import load_dataset
import polars as pl


print("DF1")
data = load_dataset("nampdn-ai/xyz")
df_1 = pl.DataFrame(data['train'].to_dict('records'))

print("DF2")
data = load_dataset("nampdn-ai/xyz-all")
df_2 = pl.DataFrame(data['train'].to_dict('records'))

print("ALL DF")
all_df = pl.concat([df_1, df_2])

print("SHUFFLE")
all_df = all_df.sample(n=all_df.height, shuffle=True)

print("WRITE")
all_df.write_parquet("/work/samba-pretrain/data/xyz.parquet")

