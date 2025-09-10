import os, glob, random
import cv2
import numpy as np
from pathlib import Path

# ----------------- Config -----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# Output
OUT_IMG_DIR = Path("generated/images")
OUT_LBL_DIR = Path("generated/labels")
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

# Dataset
BACKGROUND_DIR = Path("background_images")
FRUITS_ROOT = Path("fruit_images")  # expects subfolders per class, e.g., fruit_images/potato/*.png

# Canvas size
TARGET_W, TARGET_H = 640, 640

# Object count per image
MIN_OBJECTS, MAX_OBJECTS = 2, 2

# Scale/rotation augmentation
MIN_SCALE, MAX_SCALE = 0.3, 1.4
MAX_ROTATE_DEG = 25

# Dataset size knobs
VARIATIONS_PER_BACKGROUND_RANDOM = 5
PAIRS = [("potato", "orange"), ("orange", "pumpkin"), ("potato", "pumpkin")]
VARIATIONS_PER_PAIR_PER_BACKGROUND = 3

# Placement constraints
MAX_PLACEMENT_TRIES = 120
MAX_IOU_ALLOWED = 0.2     # for non-overlap scenes

# ---- Overlap controls ----
OVERLAP_SCENE_PROB = 0.2    # 50% of scenes will enforce at least one overlapping pair
OVERLAP_IOU_RANGE = (0.25, 0.60)  # target IoU for the overlapped pair
ALLOW_MULTI_OVERLAPS = True # allow more than one overlap in an "overlap" scene

# ------------------------------------------

def load_assets():
    class_to_imgs = {}
    class_names = []
    for class_dir in sorted([p for p in FRUITS_ROOT.iterdir() if p.is_dir()]):
        imgs = sorted(glob.glob(str(class_dir / "*")))
        if imgs:
            cls = class_dir.name
            class_names.append(cls)
            class_to_imgs[cls] = imgs

    backgrounds = sorted(glob.glob(str(BACKGROUND_DIR / "*")))
    if not backgrounds:
        raise RuntimeError("No backgrounds found in background_images/")

    if not class_names:
        raise RuntimeError("No fruit classes found. Expected subfolders under fruit_images/")

    class_to_id = {c: i for i, c in enumerate(class_names)}
    return class_to_imgs, class_to_id, backgrounds, class_names

def read_background(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read background: {path}")
    return cv2.resize(img, (TARGET_W, TARGET_H))

def read_fruit_with_mask(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read fruit: {path}")

    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)[1]
    else:
        bgr = img
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]

    x, y, w, h = cv2.boundingRect(mask)
    bgr = bgr[y:y+h, x:x+w]
    mask = mask[y:y+h, x:x+w]
    return bgr, mask

def augment_scale_rotate(img, mask, scale, angle_deg):
    h, w = img.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    center = (new_w // 2, new_h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = np.abs(M[0, 0]); sin = np.abs(M[0, 1])
    nW = int((new_h * sin) + (new_w * cos))
    nH = int((new_h * cos) + (new_w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    img_rot = cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    mask_rot = cv2.warpAffine(mask, M, (nW, nH), flags=cv2.INTER_NEAREST, borderValue=0)

    if np.count_nonzero(mask_rot) == 0:
        return None, None
    x, y, w, h = cv2.boundingRect(mask_rot)
    return img_rot[y:y+h, x:x+w], mask_rot[y:y+h, x:x+w]

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    return interArea / float(areaA + areaB - interArea + 1e-9)

def place_object(canvas_w, canvas_h, obj_w, obj_h, existing_boxes, max_iou=MAX_IOU_ALLOWED):
    """Place object trying to avoid overlap (IoU <= max_iou vs. every existing box)."""
    for _ in range(MAX_PLACEMENT_TRIES):
        x = np.random.randint(-obj_w//3, canvas_w - (2*obj_w)//3)
        y = np.random.randint(-obj_h//3, canvas_h - (2*obj_h)//3)
        cand = (max(0, x), max(0, y),
                min(obj_w, canvas_w - max(0, x)),
                min(obj_h, canvas_h - max(0, y)))
        if cand[2] <= 1 or cand[3] <= 1:
            continue
        ok = True
        for b in existing_boxes:
            if iou(cand, b) > max_iou:
                ok = False
                break
        if ok:
            return x, y
    return None, None

def place_object_with_overlap(canvas_w, canvas_h, obj_w, obj_h, existing_boxes,
                              target_range=OVERLAP_IOU_RANGE, allow_multi=ALLOW_MULTI_OVERLAPS):
    """
    Place object so that it overlaps with at least one existing box in target IoU range.
    For other boxes (if any), allow overlap if allow_multi else keep overlap light.
    """
    if not existing_boxes:
        # If no existing boxes, fall back to free placement
        return place_object(canvas_w, canvas_h, obj_w, obj_h, existing_boxes, max_iou=1.0)

    tmin, tmax = target_range
    for _ in range(MAX_PLACEMENT_TRIES):
        # Bias sampling near a random anchor to increase odds of achieving target IoU
        anchor = existing_boxes[np.random.randint(0, len(existing_boxes))]
        # Sample around anchor with some jitter
        ax, ay, aw, ah = anchor
        x = np.random.randint(ax - obj_w + aw//2, ax + aw)
        y = np.random.randint(ay - obj_h + ah//2, ay + ah)

        # Clip box to canvas
        cand = (max(0, x), max(0, y),
                min(obj_w, canvas_w - max(0, x)),
                min(obj_h, canvas_h - max(0, y)))
        if cand[2] <= 1 or cand[3] <= 1:
            continue

        # Check IoU to any box meets target range
        meets = False
        for b in existing_boxes:
            ov = iou(cand, b)
            if tmin <= ov <= tmax:
                meets = True
                break
        if not meets:
            continue

        # For other boxes, limit extreme pileups unless allow_multi
        if not allow_multi:
            clean = True
            for b in existing_boxes:
                ov = iou(cand, b)
                if ov > tmax + 0.05:  # small slack
                    clean = False
                    break
            if not clean:
                continue

        return x, y

    # If we couldn't hit the target IoU, relax: place freely (may or may not overlap)
    return place_object(canvas_w, canvas_h, obj_w, obj_h, existing_boxes, max_iou=1.0)

def composite_one(background_bgr, objects, force_overlap=False):
    """
    objects: list of (class_id, fruit_path)
    force_overlap: ensure at least one overlapping pair is created if True
    Returns: composited image, YOLO labels [(cls, cx, cy, w, h) normalized]
    """
    canvas = background_bgr.copy()
    H, W = canvas.shape[:2]
    placed_boxes = []
    yolo_labels = []
    overlapped_pair_done = False

    # Randomize draw order to vary occlusion
    order = list(range(len(objects)))
    random.shuffle(order)

    for idx, obj_i in enumerate(order):
        cls_id, fruit_path = objects[obj_i]
        base_bgr, base_mask = read_fruit_with_mask(fruit_path)

        scale = np.random.uniform(MIN_SCALE, MAX_SCALE)
        angle = np.random.uniform(-MAX_ROTATE_DEG, MAX_ROTATE_DEG)
        aug_bgr, aug_mask = augment_scale_rotate(base_bgr, base_mask, scale, angle)
        if aug_bgr is None:
            continue

        fh, fw = aug_bgr.shape[:2]

        # Decide placement strategy
        use_overlap = False
        if force_overlap:
            if (not overlapped_pair_done and len(placed_boxes) >= 1) or \
               (ALLOW_MULTI_OVERLAPS and random.random() < 0.3 and len(placed_boxes) >= 1):
                use_overlap = True

        if use_overlap:
            x, y = place_object_with_overlap(W, H, fw, fh, placed_boxes,
                                             target_range=OVERLAP_IOU_RANGE,
                                             allow_multi=ALLOW_MULTI_OVERLAPS)
        else:
            # free placement (non-overlap-biased)
            x, y = place_object(W, H, fw, fh, placed_boxes, max_iou=MAX_IOU_ALLOWED)

        if x is None:
            # couldn't place this one; skip
            continue

        # Clip to canvas
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x + fw); y1 = min(H, y + fh)
        if x1 <= x0 or y1 <= y0:
            continue

        ox0 = x0 - x
        oy0 = y0 - y
        ox1 = ox0 + (x1 - x0)
        oy1 = oy0 + (y1 - y0)

        obj_roi = aug_bgr[oy0:oy1, ox0:ox1]
        msk_roi = aug_mask[oy0:oy1, ox0:ox1]

        # composite
        bg_roi = canvas[y0:y1, x0:x1]
        inv = cv2.bitwise_not(msk_roi)
        bg_part = cv2.bitwise_and(bg_roi, bg_roi, mask=inv)
        fg_part = cv2.bitwise_and(obj_roi, obj_roi, mask=msk_roi)
        canvas[y0:y1, x0:x1] = cv2.add(bg_part, fg_part)

        nz = cv2.findNonZero(msk_roi)
        if nz is None:
            continue
        rx, ry, rw, rh = cv2.boundingRect(nz)
        abs_x, abs_y, abs_w, abs_h = x0 + rx, y0 + ry, rw, rh

        # mark if this placement overlapped in target range with any previous
        if force_overlap and not overlapped_pair_done and placed_boxes:
            for b in placed_boxes:
                if OVERLAP_IOU_RANGE[0] <= iou((abs_x, abs_y, abs_w, abs_h), b) <= OVERLAP_IOU_RANGE[1]:
                    overlapped_pair_done = True
                    break

        placed_boxes.append((abs_x, abs_y, abs_w, abs_h))

        # YOLO label (normalized)
        cx = (abs_x + abs_w / 2) / W
        cy = (abs_y + abs_h / 2) / H
        nw = abs_w / W
        nh = abs_h / H
        yolo_labels.append((cls_id, cx, cy, nw, nh))

    # If we promised overlap but couldn't achieve it (edge case), it's fine to return;
    # optionally you could discard this image and try again.
    return canvas, yolo_labels

def choose_random_scene(class_to_imgs):
    n_obj = np.random.randint(MIN_OBJECTS, MAX_OBJECTS + 1)
    classes = random.choices(list(class_to_imgs.keys()), k=n_obj)
    picks = []
    for c in classes:
        img_path = random.choice(class_to_imgs[c])
        picks.append((c, img_path))
    return picks

def choose_pair_scene(class_to_imgs, pair):
    c1, c2 = pair
    picks = []
    img1 = random.choice(class_to_imgs[c1])
    img2 = random.choice(class_to_imgs[c2])
    picks.append((c1, img1))
    picks.append((c2, img2))
    if MAX_OBJECTS >= 3 and random.random() < 0.3:
        rest = [k for k in class_to_imgs.keys()]
        c3 = random.choice(rest)
        picks.append((c3, random.choice(class_to_imgs[c3])))
    return picks

def write_yolo_labels(path_no_ext, labels):
    with open(str(OUT_LBL_DIR / f"{path_no_ext}.txt"), "w") as f:
        for cls, cx, cy, w, h in labels:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def main():
    class_to_imgs, class_to_id, backgrounds, class_names = load_assets()
    print("Classes:", class_names)
    print("Will generate images into:", OUT_IMG_DIR, "and labels into:", OUT_LBL_DIR)

    counter = 0
    for bg_path in backgrounds:
        bg = read_background(bg_path)

        # Targeted pair scenes (some with enforced overlap)
        for pair in PAIRS:
            if pair[0] not in class_to_imgs or pair[1] not in class_to_imgs:
                continue
            for _ in range(VARIATIONS_PER_PAIR_PER_BACKGROUND):
                obj_specs = choose_pair_scene(class_to_imgs, pair)
                objects = [(class_to_id[c], p) for (c, p) in obj_specs]
                force_overlap = random.random() < OVERLAP_SCENE_PROB
                img, labels = composite_one(bg, objects, force_overlap=force_overlap)
                stem = f"img_{counter:06d}"
                cv2.imwrite(str(OUT_IMG_DIR / f"{stem}.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                write_yolo_labels(stem, labels)
                counter += 1

        # Random mixes per background (also mixed overlap)
        for _ in range(VARIATIONS_PER_BACKGROUND_RANDOM):
            obj_specs = choose_random_scene(class_to_imgs)
            objects = [(class_to_id[c], p) for (c, p) in obj_specs]
            force_overlap = random.random() < OVERLAP_SCENE_PROB
            img, labels = composite_one(bg, objects, force_overlap=force_overlap)
            stem = f"img_{counter:06d}"
            cv2.imwrite(str(OUT_IMG_DIR / f"{stem}.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            write_yolo_labels(stem, labels)
            counter += 1

    with open("dataset.yaml", "w") as f:
        f.write("path: .\n")
        f.write(f"train: {OUT_IMG_DIR.as_posix()}\n")
        f.write(f"val: {OUT_IMG_DIR.as_posix()}\n")
        f.write("names:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")

    print(f"Done. Generated {counter} images.")

if __name__ == "__main__":
    main()