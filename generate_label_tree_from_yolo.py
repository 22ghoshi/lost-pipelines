from ultralytics import YOLO
import csv

model_path = "models/best.pt"   # change to your .pt path
tree_name  = "yolo-labels"
out_csv    = f"{model_path}-labels.csv"

m = YOLO(model_path)
names = m.names  # dict[int,str]

with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name","parent","external_id"])
    w.writerow([tree_name,"","root"])
    for k in sorted(names):
        w.writerow([names[k], tree_name, str(k)])

print("Wrote:", out_csv)