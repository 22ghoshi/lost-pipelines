from lost.pyapi import script
import os
import tempfile
import cv2
import numpy as np

from custom_setup.custom_arch import load_checkpoint
from custom_setup.custom_infer import infer_image

ENVS = ["lost"]
ARGUMENTS = {
    "model_path": {"value": "models/low_label_1783020702_phase2_yolo11n_autolab_best.pth", "help": "Path to autolabeler .pth (relative to pipeline project)."},
    "conf": {"value": 0.45, "help": "Confidence threshold."},
    "iou_thres": {"value": 0.45, "help": "IoU threshold for NMS."},
    "recursive": {"value": "true", "help": "Walk recursively if datasource is a directory."},
    "label_tree": {"value": "2025_comp", "help": "name of label tree to use"}
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
        self.logger.info("=== AUTOLABELER INFERENCE SCRIPT STARTED ===")
        
        # Resolve the model path inside the pipeline project
        model_rel = self.get_arg("model_path")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_abs = model_rel if os.path.isabs(model_rel) else os.path.join(script_dir, model_rel)
        
        self.logger.info(f"Loading model from: {model_abs}")
        model, class_names = load_checkpoint(model_abs, device="cpu")
        self.logger.info(f"Model classes: {class_names}")

        # Create a label tree matching YOLO names
        tree = self.get_label_tree(self.get_arg("label_tree"))
        if tree is None:
            self.logger.info("Creating new label tree")
            tree = self.create_label_tree("yolo-labels")
            df = tree.to_df()
            root_id = int(df.loc[df["is_root"] == True, "idx"].iloc[0])
            self.logger.info(f"Root ID: {root_id}")
            
            for cls_id, cls_name in enumerate(class_names):
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
                
                # Open image from S3 using the filesystem object
                try:
                    with fs.open(img_path, 'rb') as f:
                        img_bytes = f.read()
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        h, w = img.shape[:2]
                        if img is None:
                            self.logger.error(f"Failed to decode image: {img_path}")
                            continue
                        h, w = img.shape[:2]
                        self.logger.debug(f"Image size: {w}x{h}, channels: {img.shape[2] if len(img.shape) > 2 else 1}")
                        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                            cv2.imwrite(tmp.name, img)
                            tmp_path = tmp.name
                            
                        try:
                            # Run inference
                            results = infer_image(model, img, "cpu", conf_thres=conf, iou_thres=0.45, img_size=640)

                            annos = []
                            anno_types = []
                            anno_labels = []

                            if results.shape[0] > 0:
                                xyxy = results[:, :4].cpu().tolist()
                                cls_ids = results[:, 5].int().cpu().tolist()

                                for box_xyxy, cls_id in zip(xyxy, cls_ids):
                                    annos.append(self._xyxy_to_rel_xywh(box_xyxy, w, h))
                                    anno_types.append("bbox")

                                    cls_name = class_names[int(cls_id)] if 0 <= int(cls_id) < len(class_names) else str(cls_id)
                                    cls_name_lower = str(cls_name).lower()
                                    leaf_id = name_to_leaf_id.get(cls_name_lower)
                                    
                                    if leaf_id is None:
                                        self.logger.warning(f"Class '{cls_name}' not found in label tree! Available: {list(name_to_leaf_id.keys())}")
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
                            
                        finally:
                            # Clean up temp file
                            import os as os_module
                            if os_module.path.exists(tmp_path):
                                os_module.unlink(tmp_path)
                except Exception as e:
                    self.logger.error(f"Error processing {img_path}: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    continue
 
        self.logger.info("=== AUTOLABELER INFERENCE SCRIPT COMPLETED ===")
 
if __name__ == "__main__":
    my_script = LostScript()
