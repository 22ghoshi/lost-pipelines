from ultralytics import YOLO
import csv

model_path = "models/front_yolo_v3.pt"   # change to your .pt path
tree_name  = "front_yolo_v3_labels"
out_csv    = f"{model_path}-labels.csv"

m = YOLO(model_path)
names = m.names  # dict[int,str]

with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    # Write header matching the required format
    w.writerow(["idx", "name", "external_id", "parent_leaf_id"])
    
    # Write root node (idx=1, no parent)
    w.writerow([1, tree_name, "", ""])
    
    # Write child nodes (starting from idx=2)
    # Each child has the root (idx=1) as parent
    for i, k in enumerate(sorted(names), start=2):
        class_name = names[k]
        external_id = str(k)
        parent_leaf_id = 1  # Root node ID
        w.writerow([i, class_name, external_id, parent_leaf_id])

print("Wrote:", out_csv)