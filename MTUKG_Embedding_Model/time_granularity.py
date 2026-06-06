import json

idx = json.load(open("HUSK_Embedding_Model/embedding_quarter/NYC_region_temporal_index.json"))
print("time_granularity:", idx.get("time_granularity"))
print("shape:", idx.get("shape"))

time_axis = idx.get("time_axis", [])
print("num days:", len(time_axis))
print("first 10:")
for r in time_axis[:10]:
    print(r)

print("source_granularity counts:")
cnt = {}
for r in time_axis:
    sg = r.get("source_granularity", "unknown")
    cnt[sg] = cnt.get(sg, 0) + 1
print(cnt)