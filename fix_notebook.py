import json

with open("marker_annotation.ipynb", "r") as f:
    nb = json.load(f)

# Find the index of the phase 2 cell
# Phase 2 cell contains 'Phase 2 — Fine-Tune YOLOv8'
idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = "".join(cell['source'])
        if 'Phase 2 — Fine-Tune YOLOv8' in src:
            idx = i
            break

if idx != -1:
    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 1.5 — Lightweight YOLOv8 Evaluation (A/B Test)\n",
            "\n",
            "Run a lightweight YOLOv8n before and after outlier reduction to compare performance."
        ]
    }
    
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import shutil\n",
            "import glob\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import random\n",
            "import cv2\n",
            "from ultralytics import YOLO\n",
            "\n",
            "# 1. Data Preparation\n",
            "print(\"--- PREPARING DATASETS ---\")\n",
            "DATASET_BEFORE = \"yolo_dataset_before\"\n",
            "DATASET_AFTER = \"yolo_dataset_after\"\n",
            "\n",
            "for d in [DATASET_BEFORE, DATASET_AFTER]:\n",
            "    os.makedirs(os.path.join(d, \"images\"), exist_ok=True)\n",
            "    os.makedirs(os.path.join(d, \"labels\"), exist_ok=True)\n",
            "\n",
            "imgs = glob.glob(os.path.join(YOLO_DATASET_DIR, \"images\", \"*.*\"))\n",
            "labels_before_dir = os.path.join(YOLO_DATASET_DIR, \"labels_backup_before_outlier_filter\")\n",
            "\n",
            "for img_p in imgs:\n",
            "    fname = os.path.basename(img_p)\n",
            "    shutil.copy(img_p, os.path.join(DATASET_BEFORE, \"images\", fname))\n",
            "    shutil.copy(img_p, os.path.join(DATASET_AFTER, \"images\", fname))\n",
            "\n",
            "if os.path.exists(labels_before_dir):\n",
            "    for lbl_p in glob.glob(os.path.join(labels_before_dir, \"*.txt\")):\n",
            "        shutil.copy(lbl_p, os.path.join(DATASET_BEFORE, \"labels\", os.path.basename(lbl_p)))\n",
            "        \n",
            "for lbl_p in glob.glob(os.path.join(YOLO_DATASET_DIR, \"labels\", \"*.txt\")):\n",
            "    shutil.copy(lbl_p, os.path.join(DATASET_AFTER, \"labels\", os.path.basename(lbl_p)))\n",
            "\n",
            "def write_eval_yaml(ds_dir, yaml_name):\n",
            "    yaml_path = os.path.join(ds_dir, yaml_name)\n",
            "    with open(yaml_path, \"w\") as f:\n",
            "        f.write(f\"path: {os.path.abspath(ds_dir)}\\n\")\n",
            "        f.write(\"train: images\\n\")\n",
            "        f.write(\"val:   images\\n\")\n",
            "        f.write(\"\\nnc: 1\\n\")\n",
            "        f.write(\"names: ['marker']\\n\")\n",
            "    return yaml_path\n",
            "\n",
            "yaml_before = write_eval_yaml(DATASET_BEFORE, \"markers_before.yaml\")\n",
            "yaml_after = write_eval_yaml(DATASET_AFTER, \"markers_after.yaml\")\n",
            "\n",
            "# 2. Training (Lightweight Evaluation on YOLOv8n)\n",
            "EVAL_EPOCHS = 10\n",
            "\n",
            "print(\"\\n--- TRAINING 'BEFORE' MODEL ---\")\n",
            "# Set exist_ok=True but project/name different so it doesn't collide\n",
            "model_before = YOLO(\"yolov8n.pt\")\n",
            "model_before.train(data=yaml_before, epochs=EVAL_EPOCHS, imgsz=640, batch=16, project=\"runs/detect\", name=\"eval_before\", exist_ok=True)\n",
            "\n",
            "print(\"\\n--- TRAINING 'AFTER' MODEL ---\")\n",
            "model_after = YOLO(\"yolov8n.pt\")\n",
            "model_after.train(data=yaml_after, epochs=EVAL_EPOCHS, imgsz=640, batch=16, project=\"runs/detect\", name=\"eval_after\", exist_ok=True)\n",
            "\n",
            "# 3. Numerical Comparison (Plot Loss & mAP)\n",
            "print(\"\\n--- NUMERICAL COMPARISON ---\")\n",
            "try:\n",
            "    df_b = pd.read_csv(\"runs/detect/eval_before/results.csv\")\n",
            "    df_a = pd.read_csv(\"runs/detect/eval_after/results.csv\")\n",
            "    \n",
            "    df_b.columns = df_b.columns.str.strip()\n",
            "    df_a.columns = df_a.columns.str.strip()\n",
            "    \n",
            "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
            "    ax1.plot(df_b['epoch'], df_b['metrics/mAP50(B)'], marker='o', label=\"Before Outliers\")\n",
            "    ax1.plot(df_a['epoch'], df_a['metrics/mAP50(B)'], marker='s', label=\"After Outliers\")\n",
            "    ax1.set_title(\"mAP@50 over Epochs\")\n",
            "    ax1.set_xlabel(\"Epoch\")\n",
            "    ax1.set_ylabel(\"mAP@50\")\n",
            "    ax1.legend()\n",
            "    ax1.grid(True)\n",
            "    \n",
            "    ax2.plot(df_b['epoch'], df_b['train/box_loss'], marker='o', label=\"Before Outliers\")\n",
            "    ax2.plot(df_a['epoch'], df_a['train/box_loss'], marker='s', label=\"After Outliers\")\n",
            "    ax2.set_title(\"Train Box Loss over Epochs\")\n",
            "    ax2.set_xlabel(\"Epoch\")\n",
            "    ax2.set_ylabel(\"Loss\")\n",
            "    ax2.legend()\n",
            "    ax2.grid(True)\n",
            "    \n",
            "    plt.suptitle(\"Numerical Comparison: Before vs. After Outlier Filtering (Yolov8n)\", fontsize=14)\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "except Exception as e:\n",
            "    print(f\"Could not load results for plotting: {e}\")\n",
            "\n",
            "# 4. Visual Comparison (Bounding Boxes)\n",
            "print(\"\\n--- VISUAL COMPARISON ---\")\n",
            "val_images = sorted(glob.glob(os.path.join(DATASET_AFTER, \"images\", \"*.jpg\")) + glob.glob(os.path.join(DATASET_AFTER, \"images\", \"*.jpeg\")))\n",
            "if val_images:\n",
            "    random.seed(42)\n",
            "    samples = random.sample(val_images, min(4, len(val_images)))\n",
            "    \n",
            "    best_b_path = \"runs/detect/eval_before/weights/best.pt\"\n",
            "    best_a_path = \"runs/detect/eval_after/weights/best.pt\"\n",
            "    \n",
            "    if os.path.exists(best_b_path) and os.path.exists(best_a_path):\n",
            "        model_b_best = YOLO(best_b_path)\n",
            "        model_a_best = YOLO(best_a_path)\n",
            "        \n",
            "        fig_vis, axes_vis = plt.subplots(len(samples), 2, figsize=(12, 4 * len(samples)))\n",
            "        if len(samples) == 1:\n",
            "            axes_vis = [axes_vis]\n",
            "        \n",
            "        for i, img_p in enumerate(samples):\n",
            "            res_b = model_b_best.predict(img_p, verbose=False)[0]\n",
            "            res_a = model_a_best.predict(img_p, verbose=False)[0]\n",
            "            \n",
            "            img_b = res_b.plot()\n",
            "            img_a = res_a.plot()\n",
            "            \n",
            "            img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)\n",
            "            img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)\n",
            "            \n",
            "            ax_b, ax_a = axes_vis[i][0], axes_vis[i][1]\n",
            "            ax_b.imshow(img_b)\n",
            "            ax_b.set_title(f\"Before Filter - {os.path.basename(img_p)}\")\n",
            "            ax_b.axis('off')\n",
            "            \n",
            "            ax_a.imshow(img_a)\n",
            "            ax_a.set_title(f\"After Filter - {os.path.basename(img_p)}\")\n",
            "            ax_a.axis('off')\n",
            "            \n",
            "        plt.tight_layout()\n",
            "        plt.show()\n",
            "    else:\n",
            "        print(\"Weights files not found for visualization.\")\n",
            "else:\n",
            "    print(\"No validation images found.\")\n"
        ]
    }

    nb['cells'].insert(idx, markdown_cell)
    nb['cells'].insert(idx + 1, code_cell)

    with open("marker_annotation.ipynb", "w") as f:
        json.dump(nb, f)
    
    print(f"Successfully inserted Phase 1.5 at index {idx}.")
else:
    print("Could not find Phase 2 cell.")
