# Bangladeshi Taka Note Detection using YOLOv8

This project detects and classifies Bangladeshi currency notes from images using the YOLOv8 object detection model. It was built as part of my university coursework on computer vision and deep learning.

## What it does

The model takes an image as input and draws bounding boxes around any Bangladeshi Taka notes it finds, along with the denomination label and confidence score. It can detect the following denominations:

- 2 Taka
- 5 Taka
- 10 Taka
- 20 Taka
- 50 Taka
- 100 Taka
- 200 Taka
- 500 Taka
- 1000 Taka

As a bonus, I also trained a separate model that can detect **2 Taka** and **5 Taka coins** in addition to the notes above.

## Dataset

- **Source:** [Bangladeshi Currency Detection on Roboflow Universe](https://universe.roboflow.com/tanvirtain/bangladeshi-currency-detection)
- **Total images:** ~1500
- **Format:** YOLOv8 (bounding box annotations in `.txt` files)
- **Split:** 70% training / 20% validation / 10% testing
- **Annotation tool:** Roboflow

The images include notes photographed in different lighting conditions, angles, backgrounds, and orientations.

## Model Details

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8n (nano) |
| Pretrained on | COCO dataset |
| Epochs | 50 |
| Batch size | 16 |
| Image size | 640x640 |
| Early stopping | 15 epochs patience |
| Framework | Ultralytics |

## Project Structure

```
Bangladeshi_Taka_Detection/
|-- taka_note_detection.ipynb     # main notebook (training + evaluation)
|-- README.md
|
|-- dataset/                      # downloaded from Roboflow
|   |-- data.yaml
|   |-- train/images/, train/labels/
|   |-- valid/images/, valid/labels/
|   |-- test/images/, test/labels/
|
|-- runs/detect/taka_detection/
|   |-- train_v1/                 # training output
|   |   |-- weights/best.pt      # trained model weights
|   |   |-- results.csv          # epoch-wise metrics
|   |   |-- results.png          # loss/metric curves
|   |   |-- confusion_matrix.png
|   |-- eval_test/                # test set evaluation
|   |-- predictions/              # detection outputs on test images
|   |-- train_with_coins/         # bonus task training
|   |-- eval_coins/               # bonus task evaluation
|   |-- predictions_with_coins/   # bonus task predictions
```

## How to Run

1. Open `taka_note_detection.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set runtime to **T4 GPU** (Runtime > Change runtime type)
3. Click **Runtime > Run all**
4. Wait around 30-40 minutes for training to finish
5. Results, plots, and model weights are saved automatically

## Results

The model was evaluated on a held-out test set. Key metrics:

- **mAP@50:** reported in the notebook after evaluation
- **Precision & Recall:** per-class breakdown available in the notebook
- Confusion matrix, F1 curve, and PR curve are generated automatically

Detection examples with bounding boxes are saved in the `predictions/` folder.

## Bonus Task - Coin Detection

I extended the model to also detect Bangladeshi coins (2 Taka coin and 5 Taka coin). The coin model was trained separately with the same YOLOv8n architecture and evaluated on the same test set. Results are in the notebook under the "Bonus Section".

## Tools Used

- Python 3
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com/) for dataset and annotation
- Google Colab for training
- OpenCV, Matplotlib, Pandas for visualization

## Author

**Emon Rahman**
