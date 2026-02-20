from lost.pyapi import script
import os
from PIL import Image

# You need ultralytics installed in the environment this script runs in
from ultralytics import YOLO

ENVS = ["lost"]
ARGUMENTS = {
    "model_path": {"value": "models/best.pt", "help": "Path to YOLO .pt (relative to pipeline project)."},
    "conf": {"value": 0.25, "help": "Confidence threshold."},
    "recursive": {"value": "true", "help": "Walk recursively if datasource is a directory."}
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
        self.logger.info(f"_iter_images called with path: {path}")
        self.logger.info(f"fs.isfile({path}): {fs.isfile(path)}")
        
        # datasource may be a single file
        if fs.isfile(path):
            yield path
            return

        # otherwise a directory
        recursive_arg = self.get_arg("recursive")
        self.logger.info(f"recursive argument: {recursive_arg} (type: {type(recursive_arg)})")
        
        recursive = str(recursive_arg).lower() == "true"
        self.logger.info(f"recursive evaluated to: {recursive}")
        
        if recursive:
            self.logger.info(f"Walking directory recursively: {path}")
            for root, dirs, files in fs.walk(path):
                self.logger.info(f"Walking root: {root}, found {len(files)} files")
                for f in files:
                    full_path = os.path.join(root, f)
                    self.logger.debug(f"Yielding: {full_path}")
                    yield full_path
        else:
            self.logger.info(f"Listing directory (non-recursive): {path}")
            for p in fs.ls(path):
                self.logger.debug(f"Yielding: {p}")
                yield p

    def main(self):
        self.logger.info("=== YOLO SCRIPT STARTED ===")
        
        # Resolve the model path inside the pipeline project
        model_rel = self.get_arg("model_path")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_abs = model_rel if os.path.isabs(model_rel) else os.path.join(script_dir, model_rel)
        
        self.logger.info(f"Loading model from: {model_abs}")
        model = YOLO(model_abs)
        self.logger.info(f"Model classes: {model.names}")

        # Create a label tree matching YOLO names
        tree = self.get_label_tree("yolo-labels-2")
        if tree is None:
            self.logger.info("Creating new label tree")
            tree = self.create_label_tree("yolo-labels")
            df = tree.to_df()
            root_id = int(df.loc[df["is_root"] == True, "idx"].iloc[0])
            self.logger.info(f"Root ID: {root_id}")
            
            for cls_id, cls_name in model.names.items():
                tree.create_child(root_id, cls_name, external_id=str(cls_id))
                self.logger.info(f"Created label: {cls_name} (id={cls_id})")
        else:
            self.logger.info("Using existing label tree")

        # Build mapping from YOLO class id -> LOST label_leaf_id
        df = tree.to_df()
        self.logger.info(f"Label tree dataframe:\n{df}")
        
        name_to_leaf_id = {
            str(row["name"]).lower(): int(row["idx"])
            for _, row in df.iterrows()
            if not bool(row.get("is_root", False))
        }
        self.logger.info(f"Name to leaf_id mapping: {name_to_leaf_id}")

        conf = float(self.get_arg("conf"))
        
        self.logger.info(f"Processing with conf={conf}")

        for ds in self.inp.datasources:
            fs = ds.get_fs()
            base_path = ds.path
            self.logger.info(f"Processing datasource: {base_path}")

            for img_path in self._iter_images(fs, base_path):
                self.logger.debug(f"Found file: {img_path}")
                ext = os.path.splitext(img_path)[1].lower()
                if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    self.logger.debug(f"Skipping {img_path} - wrong extension")
                    continue

                self.logger.info(f"Processing image: {img_path}")
                
                with Image.open(img_path) as im:
                    w, h = im.size

                results = model.predict(source=img_path, conf=conf, verbose=False)
                r = results[0]

                annos = []
                anno_types = []
                anno_labels = []

                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().tolist()
                    cls_ids = r.boxes.cls.int().cpu().tolist()

                    for box_xyxy, cls_id in zip(xyxy, cls_ids):
                        annos.append(self._xyxy_to_rel_xywh(box_xyxy, w, h))
                        anno_types.append("bbox")

                        cls_name = model.names.get(int(cls_id), str(cls_id))
                        cls_name_lower = str(cls_name).lower()
                        leaf_id = name_to_leaf_id.get(cls_name_lower)
                        
                        if leaf_id is None:
                            self.logger.warning(f"Class '{cls_name}' not found in label tree! Available: {list(name_to_leaf_id.keys())}")
                            # Don't add labels if not found - let frontend handle unlabeled boxes
                            anno_labels.append([])
                        else:
                            anno_labels.append([int(leaf_id)])
                            self.logger.debug(f"Box: class={cls_name} -> leaf_id={leaf_id}")

                kwargs = dict(img=img_path, fs=fs)

                if annos:
                    kwargs.update(
                        annos=annos, 
                        anno_types=["bbox"] * len(annos), 
                        anno_labels=anno_labels
                    )
                    self.logger.info(f"Requesting {len(annos)} boxes with labels: {anno_labels}")
                else:
                    self.logger.info(f"No detections for {img_path}")

                self.outp.request_annos(**kwargs)

        self.logger.info("=== YOLO SCRIPT COMPLETED ===")

if __name__ == "__main__":
    my_script = LostScript()
