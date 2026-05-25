# Project overview

This README describes the repository layout. Many large/derived files (build artifacts, models, frames, intermediate outputs) are intentionally excluded via `.gitignore` and are thus excluded from this overview.

Top-level files and folders :

- `.gitignore`: repository ignore rules.
- `environment.yaml`: Python dependencies.
- `Article/article.pdf`: final article PDF.
- `FYP Writeup/main.pdf`: thesis main PDF.
- `Progress Report/Progress Report.pdf`: thesis progress report PDF (submitted in December).
- `Readme.md`: this file.
- `Marker model files/`: Model files for the 3D-printed marker objects for this project:
	- `Anchor-Lid_V2-01_26-02-24.stp`
	- `BTM cover_V2-00_26-02-24.stp`
	- `Body-Module-A_V3-00_26-02-24.3mf`

- `code/`: key code related files:
	- `code/annotations.json`: manual annotations from bootstrapping the MITL annotation
	- `code/bbox_editor.py`: script to launch GUI for manual inspection & annotation corrections
	- `code/extractframesinbulk.py`: script to extract frames from multiple videos at a time using ffmpeg
	- `code/ground_truth_markers_P035.csv`: csv for comparing marker accuracy un unseen frames
	- `code/marker_annotation.ipynb`: python notebook for MITL annotation
	- `code/marker_detection_accuracy.ipynb`: python notebook for comparing fine-tuned model performance on unseen frames
	- `code/video_seq_processing_with_vitpose.ipynb`: main code for extracting swimmer kinematic data & evaluation
	- `code/tests and experiments/`: non-final code which is relevant nonetheless:
		- `counting_markers.ipynb`: experimental python notebook to give each marker a unique ID
		- `featuremap.ipynb`: early experimental python notebook to use primitive feature detection for scale extraction
		- `frame_seq_processing.ipynb`: experimental python notebook for scale extraction before ViTPose integration
		- `laneropesegment.ipynb`: early experimental python notebook to use primitive techniques for using the lanerope for scale extraction
		- `pixel_metre_demo.ipynb`: experimental python notebook for scale extraction before swimmer detection integration
		- `swimmer_velocity.ipynb`: experimental python notebook for swimmer speed estimation using marker and swimmer detection in a single model
		- `transformers-vitpose.ipynb`: experimental python notebook for using ViTPose
		- `vitpose_prototyping.ipynb`: experimental python notebook for using ViTPose on swimming footage with YOLO

Notes
- This README intentionally documents only files that are currently tracked by Git. Other directories such as raw `frames/`, model weights (`*.pt`), generated LaTeX sources, and many build artifacts are excluded by `.gitignore` and therefore not listed here.
- Due to ethical and privacy considerations, none of the original or output videos are included aside from short clips and images in notebook output cells as well as in report documents.
