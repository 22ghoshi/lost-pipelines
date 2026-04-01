from ultralytics import YOLO
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model-path", type=str, help="path to model (e.g. models/best.pt)", required=True)
parser.add_argument("-t", "--tree-name", type=str, help="the name to be given to the label tree", required=True)
parser.add_argument("-o", "--out-path", type=str, help="the path to put the output csv in", required=True)

parsed_args, _ = parser.parse_known_args()
model_path = parsed_args.model_path    # change to your .pt path
tree_name  = parsed_args.tree_name
out_csv    = parsed_args.out_path

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