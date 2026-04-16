# Bangladeshi Taka Note Detection - Phase 2

This is the second phase of my Bangladeshi Taka currency detection project. In phase 1, I trained a YOLOv8 model to detect different denominations of Bangladeshi Taka notes. Now in phase 2, I wrapped that model into a REST API using FastAPI and containerized the whole thing with Docker so it can run anywhere.

## What this project does

You send an image of a Bangladeshi Taka note to the API, and it tells you which denomination it is (like 500 taka, 100 taka, etc.), how confident the model is, and where exactly in the image the note was found (bounding box coordinates). The API returns everything in JSON format.

Denominations it can detect:
- 2, 5, 10, 20, 50, 100, 200, 500, and 1000 Taka

## Project structure

```
Bangladeshi_Taka_Detection/
├── app.py                 # FastAPI server with /predict endpoint
├── inference.py           # standalone inference script for testing
├── best.pt                # trained YOLOv8 model weights from phase 1
├── requirements.txt       # python dependencies
├── Dockerfile             # docker container setup
├── test_images/           # sample images I used for testing
├── dataset/               # roboflow dataset (train split)
├── taka_note_detection.ipynb  # phase 1 training notebook
└── README.md
```

## How to run locally (without Docker)

First install the dependencies:

```bash
pip install -r requirements.txt
```

Then start the server:

```bash
python app.py
```

or

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be live at `http://localhost:8000`. You can open `http://localhost:8000/docs` in your browser to get the interactive documentation page where you can test it directly.

## How to run the inference script

If you just want to test the model on a single image without starting the server:

```bash
python inference.py test_images/test_500_taka.jpg
```

It will print out the detections and save a result image with bounding boxes drawn on it.

## How to build and run with Docker

Make sure Docker Desktop is installed and running on your machine.

Build the image:

```bash
docker build -t taka-detection-api .
```

This takes a few minutes the first time because it has to download python and install all the packages.

Run the container:

```bash
docker run -d --name taka-api -p 8000:8000 taka-detection-api
```

Now the API is running inside Docker. You can access it at `http://localhost:8000` same as before.

To stop the container:

```bash
docker stop taka-api
```

To remove it:

```bash
docker rm taka-api
```

## How to use the API

### Check if the API is running

```bash
curl http://localhost:8000/
```

You should get back:

```json
{"message": "Bangladeshi Taka Detection API is running", "status": "ok"}
```

### Send an image for prediction

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@test_images/test_500_taka.jpg;type=image/jpeg"
```

You get back something like this:

```json
{
    "filename": "test_500_taka.jpg",
    "total_detections": 3,
    "detections": [
        {
            "class": "500 taka",
            "confidence": 0.6787,
            "bbox": [48, 20, 127, 91]
        }
    ]
}
```

You can also just go to `http://localhost:8000/docs` in the browser, click on the POST /predict endpoint, hit "Try it out", upload a file, and click Execute. Way easier than curl honestly.

## Model details

- Model: YOLOv8n (nano version, pretrained on COCO)
- Trained for 50 epochs with batch size 16 and image size 640x640
- Dataset from Roboflow with around 1500 images
- Training was done on Google Colab with T4 GPU

## Tools I used

- Python 3.11
- YOLOv8 (ultralytics)
- FastAPI + Uvicorn
- Docker
- Roboflow for the dataset
- Google Colab for training
- Postman for API testing

## Author

Md. Shadman Sakib Rahman
