# 🍽️ Smart Food Detection & Pricing System

> 📩 **Need the code and the full pipeline?** Feel free to email me at [binhadjehahmed@gmail.com](mailto:binhadjehahmed@gmail.com).

An AI system that looks at a photo of a food tray, identifies each item, estimates how much of it there is, and prices the meal based on the actual portion size.

Final Year Project, Electrical and Electronics Engineering, Universiti Tenaga Nasional (UNITEN).

---

## How It Works

| Stage | What it does | Tech |
|---|---|---|
| 1. Detection and classification | Finds each item and labels it: `bread`, `cake`, `drink`, `fruit`, `rice`, `salad` | YOLOv5m |
| 2. Scale calibration | Converts pixels to real centimeters using a printed 5.7 × 5.7 cm marker | OpenCV ArUco |
| 3a. Solid portion estimation | Predicts weight (g) from the image, shape features and depth | EfficientNet-B3 + MiDaS depth |
| 3b. Drink portion estimation | Estimates volume (ml) from bottle geometry and the liquid fill-line | Geometry + linear correction |
| 4. Calibration | Per-class linear correction to cancel systematic bias | Huber regression |
| 5. Pricing | `price = base price × clamp(portion / standard portion)`, summed into a receipt | Python |

---

## Results

**Detector (YOLOv5m, validation set):** 99.5% mAP@50, 97.7% mAP@50-95

**Portion estimation (final holdout set):** 242 crops from 84 physical portions, never seen during training or calibration.

| Food | Samples | MAE | MAPE |
|---|---|---|---|
| Bread | 15 | 5.1 g | 16.5% |
| Cake | 65 | 9.6 g | 11.5% |
| Drink | 33 | 16.8 ml | 5.3% |
| Fruit | 39 | 10.6 g | 10.8% |
| Rice | 27 | 18.5 g | 12.1% |
| Salad | 63 | 13.1 g | 8.9% |
| **Overall** | **242** | **12.3** | **10.2%** |

> The geometry-based drink path is evaluated separately: **35.0 ml MAE** on its own held-out set (n = 49).

---

## Data

- **Detection dataset:** labeled bounding boxes for 6 food classes (Roboflow, YOLOv5 format).
- **Portion dataset:** 1,607 photos with measured ground-truth weight (g) or volume (ml), giving 1,534 usable crops from 556 physical samples.
- **Split (grouped by physical sample, so there is no leakage):** Train 1,012 crops, Test/calibration 280 crops, Final holdout 242 crops.

---

## Running the Notebook (Google Colab, GPU)

1. Open `Smart_Food_Detection_and_Pricing_System_Fi.ipynb` in Google Colab and select a GPU runtime.
2. Add your own Roboflow API key in Cell 2.
3. Run the cells in order:
   - **Cells 1-4:** install dependencies, download the dataset, train and load YOLOv5
   - **Cells 5-8:** upload portion data, build features, split the data
   - **Cells 9-11c:** train the regressor, calibrate, evaluate on the holdout
   - **Cells 12-13:** pricing and the full pipeline
   - **Cells 14-16:** export model files
   - **Cell 15:** interactive dashboard (webcam or upload)
   - **Cell 18:** quick test on a single photo
4. Place the ArUco marker (`DICT_6X6_250`, ID 0, 5.7 cm) in every photo, otherwise the scale falls back to a rough default and accuracy drops.

### Requirements

Python 3.10+, PyTorch, YOLOv5, timm, albumentations, OpenCV, scikit-learn, pandas, roboflow.

---

## Exported Files

`best_yolo.pt`, `best_regressor.pt`, `fold_0/1/2.pt` (optional ensemble), `calibration.json`, `liquid_calibration.json`, `feature_scaler.json`, `bottle_profile.json`, `pipeline_config.json`

---

## Known Limitations

- Needs the ArUco marker visible in the photo for accurate sizing.
- Drink volume depends on measured bottle dimensions (`BOTTLE_PROFILE_CM`). Replace the placeholder values with your real bottle's measurements before deploying.
- Covers 6 food types only.
- Prices in the pricing table are example values and can be adjusted per class.

---

## Tech Stack

Python · PyTorch · YOLOv5 · EfficientNet-B3 · MiDaS · OpenCV · scikit-learn · Google Colab
