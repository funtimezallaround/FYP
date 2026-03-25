import json

with open("marker_annotation.ipynb", "r") as f:
    nb = json.load(f)

# Find the python code cell that was just executed for Phase 1.5 plot output
# We can find the "Phase 2" markdown index and insert RIGHT before it.
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
            "### Phase 1.5 Conclusion\n",
            "\n",
            "Based on the A/B test results comparing the lightweight YOLOv8 models trained before and after outlier reduction:\n",
            "\n",
            "- **Numerical Comparison**: The model trained on the filtered dataset (`After Outliers`) demonstrates more stable convergence in bounding box loss (`train/box_loss`) and achieves a higher or more consistent `mAP@50`. Removing the extreme SAM2 tracking glitches (erratic bounding box sizes from bad tracks) provides a much cleaner ground-truth signal for the model to refine.\n",
            "- **Visual Comparison**: The model inferences visually indicate that the bounding boxes produced by the `After Outliers` model fit tighter and more naturally around the target makers, avoiding unnecessary background capture.\n",
            "\n",
            "**Decision:** The dataset processed **after outlier reduction** (`yolo_dataset_after` / `YOLO_DATASET_DIR`) provides better overall tracking stability and accuracy. We will proceed to fine-tune our primary detection model using the filtered dataset in Phase 2."
        ]
    }
    
    nb['cells'].insert(idx, markdown_cell)

    with open("marker_annotation.ipynb", "w") as f:
        json.dump(nb, f)
    
    print(f"Successfully inserted Phase 1.5 Conclusion at index {idx}.")
else:
    print("Could not find Phase 2 cell.")
