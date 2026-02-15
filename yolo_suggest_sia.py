from lost.pyapi import script
import os
from PIL import Image

# You need ultralytics installed in the environment this script runs in
from ultralytics import YOLO

ENVS = ["lost"]
ARGUMENTS = {
    "model_path": {"value": "models/best.pt", "help": "Path to YOLO .pt (relative to pipeline project)."},
    "conf": {"value": 0.25, "help": "Confidence threshold."},
    "single_image": {"value": "-", "help": "If not '-', only process this basename (e.g. img001.jpg)."},
    "recursive": {"value": "false", "help": "Walk recursively if datasource is a directory."}
}

class LostScript(script.Script):
    def _xyxy_to_rel_xywh(self, xyxy, w, h):
        x1, y1, x2, y2 = xyxy
        # clamp to image bounds
        x1 = max(0.0, min(float(x1), w))
        x2 = max(0.0, min(float(x2), w))
        y1 = max(0.0, min(float(y1), h))
        y2 = max(0.0, min(float(y2), h))

        xc = ((x1 + x2) / 2.0) / w
        yc = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        return [xc, yc, bw, bh]

    def _iter_images(self, fs, path):
        # datasource may be a single file
        if fs.isfile(path):
            yield path
            return

        # otherwise a directory
        if self.get_arg("recursive"):
            for root, dirs, files in fs.walk(path):
                for f in files:
                    yield os.path.join(root, f)
        else:
            for p in fs.ls(path):
                yield p

    def main(self):
        # Resolve the model path inside the pipeline project
        model_rel = self.get_arg("model_path")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_abs = model_rel if os.path.isabs(model_rel) else os.path.join(script_dir, model_rel)
        model = YOLO(model_abs)

        # Create a label tree matching YOLO names (optional, but enables anno_labels)
        # If tree exists already, this will return it; otherwise create_root will.
        tree = self.get_label_tree("yolo-labels")
        if tree is None:
            tree = self.create_label_tree("yolo-labels")
            # root exists as the tree root; add children as class labels
            # Ultralytics model.names is dict[int,str]
            df = tree.to_df()
            root_id = int(df.loc[df["is_root"] == True, "idx"].iloc[0])
            for cls_id, cls_name in model.names.items():
                tree.create_child(root_id, cls_name, external_id=str(cls_id))

        # Build mapping from YOLO class id -> LOST label_leaf_id
        # (we look up children by name)
        df = tree.to_df()
        name_to_leaf_id = {
            str(row["name"]).lower(): int(row["idx"])
            for _, row in df.iterrows()
            if not bool(row.get("is_root", False))  # skip root if present
        }

        only_basename = self.get_arg("single_image")
        conf = float(self.get_arg("conf"))

        for ds in self.inp.datasources:
            fs = ds.get_fs()
            base_path = ds.path

            for img_path in self._iter_images(fs, base_path):
                ext = os.path.splitext(img_path)[1].lower()
                if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    continue
                if only_basename != "-" and os.path.basename(img_path) != only_basename:
                    continue

                # Read image size (works for local filesystem paths; for non-local fs youd need fs.open)
                # LOST rawFile is usually local, so this is fine.
                with Image.open(img_path) as im:
                    w, h = im.size

                results = model.predict(source=img_path, conf=conf, verbose=False)

                # One image -> one Results object
                r = results[0]

                annos = []
                anno_types = []
                anno_labels = []

                # r.boxes.xyxy and r.boxes.cls are standard in Ultralytics Results
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().tolist()
                    cls_ids = r.boxes.cls.int().cpu().tolist()

                    for box_xyxy, cls_id in zip(xyxy, cls_ids):
                        annos.append(self._xyxy_to_rel_xywh(box_xyxy, w, h))
                        anno_types.append("bbox")

                        cls_name = model.names.get(int(cls_id), str(cls_id))
                        leaf_id = name_to_leaf_id.get(cls_name)
                        anno_labels.append([int(leaf_id)] if leaf_id is not None else [])

                kwargs = dict(img=img_path, fs=fs)


                if annos:
                    kwargs.update(annos=annos, anno_types=["bbox"] * len(annos), anno_labels=anno_labels)

                self.outp.request_annos(**kwargs)

                self.logger.info(f"Requested {len(annos)} YOLO proposals for {img_path}")

if __name__ == "__main__":
    my_script = LostScript()
