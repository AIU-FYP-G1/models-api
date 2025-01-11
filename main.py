import base64

import pandas as pd
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
import boto3
import numpy as np
from settings import Settings
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
import cv2
from typing import Dict
import os

from fastapi import FastAPI
import threading

app = FastAPI()

settings = Settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

feature_extractor = None
a4c_model = None
psax_model = None


def extract_video_features(video_data, feature_extractor, sequence_length=30, interval=1):
    try:
        # Decode base64 video data
        video_bytes = base64.b64decode(video_data)

        # Save to temporary file
        temp_path = '/tmp/temp_video.mp4'
        with open(temp_path, 'wb') as f:
            f.write(video_bytes)

        # Process video
        cap = cv2.VideoCapture(temp_path)
        frame_features = []
        count = 0

        while cap.isOpened() and len(frame_features) < sequence_length:
            ret, frame = cap.read()
            if not ret:
                break

            if count % interval == 0:
                frame = cv2.resize(frame, (224, 224))
                frame = np.expand_dims(frame, axis=0)
                frame = tf.keras.applications.vgg16.preprocess_input(frame)

                features = feature_extractor.predict(frame, verbose=0)
                frame_features.append(features[0])

            count += 1

        cap.release()

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Pad sequence if needed
        while len(frame_features) < sequence_length:
            frame_features.append(np.zeros_like(frame_features[0]))

        return np.array(frame_features)

    except Exception as e:
        raise Exception(f"Error extracting video features: {str(e)}")


def process_demographic_data(demographic_data, volume_tracings, view):
    """Process demographic and volume data."""
    try:
        height_m = demographic_data['height'] / 100
        bmi = demographic_data['weight'] / (height_m ** 2)

        # Convert volume tracings to numpy arrays if they're lists
        volume_df = pd.DataFrame({
            'X': np.array(volume_tracings['X']),
            'Y': np.array(volume_tracings['Y'])
        })

        volume_stats = {
            'X_mean': np.mean(volume_df['X']),
            'X_std': np.std(volume_df['X']),
            'X_min': np.min(volume_df['X']),
            'X_max': np.max(volume_df['X']),
            'X_median': np.median(volume_df['X']),
            'X_q1': np.percentile(volume_df['X'], 25),
            'X_q3': np.percentile(volume_df['X'], 75),
            'Y_mean': np.mean(volume_df['Y']),
            'Y_std': np.std(volume_df['Y']),
            'Y_min': np.min(volume_df['Y']),
            'Y_max': np.max(volume_df['Y']),
            'Y_median': np.median(volume_df['Y']),
            'Y_q1': np.percentile(volume_df['Y'], 25),
            'Y_q3': np.percentile(volume_df['Y'], 75),
        }

        x_range = volume_stats['X_max'] - volume_stats['X_min']
        y_range = volume_stats['Y_max'] - volume_stats['Y_min']
        aspect_ratio = x_range / y_range

        AGE_BINS = [0, 30, 45, 60, 75, float('inf')]
        AGE_CATEGORIES = ['Young', 'Middle-Age', 'Early-Senior', 'Senior', 'Elderly']

        age_idx = next(i for i, threshold in enumerate(AGE_BINS[1:])
                       if demographic_data['age'] <= threshold)

        age_encoding = [1 if i == age_idx else 0 for i in range(len(AGE_CATEGORIES))]
        view_value = 1 if view == "a4c" else 0

        numerical_features = [
            demographic_data['age'],
            demographic_data['weight'],
            demographic_data['height'],
            bmi,
            volume_stats['X_mean'],
            volume_stats['X_std'],
            volume_stats['X_min'],
            volume_stats['X_max'],
            volume_stats['X_median'],
            volume_stats['X_q1'],
            volume_stats['X_q3'],
            volume_stats['Y_mean'],
            volume_stats['Y_std'],
            volume_stats['Y_min'],
            volume_stats['Y_max'],
            volume_stats['Y_median'],
            volume_stats['Y_q1'],
            volume_stats['Y_q3'],
            x_range,
            y_range,
            aspect_ratio
        ]

        all_features = np.concatenate([
            numerical_features,
            age_encoding,
            [view_value]
        ])

        return np.array(all_features)
    except Exception as e:
        raise Exception(f"Error processing demographic data: {str(e)}")

app = FastAPI()

is_loading = False
loading_complete = False
loading_error = None


def load_models_task():
    """Background task to load models"""
    global feature_extractor, a4c_model, psax_model, is_loading, loading_complete, loading_error

    try:
        is_loading = True

        s3 = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )

        # Create tmp directory if it doesn't exist
        os.makedirs('/tmp', exist_ok=True)

        # Check and download models if needed
        for model_name in ['a4c_model.keras', 'psax_model.keras']:
            model_path = f'/tmp/{model_name}'
            if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
                print(f"Downloading {model_name} from S3...")
                s3.download_file(
                    'fyp-models',
                    f'{model_name}',
                    model_path
                )
            else:
                print(f"Found existing {model_name}, skipping download")

        # Load VGG16 and create feature extractor
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

        # Load models from local files
        a4c_model = load_model('/tmp/a4c_model.keras')
        psax_model = load_model('/tmp/psax_model.keras')

        loading_complete = True

    except Exception as e:
        loading_error = str(e)
    finally:
        is_loading = False


@app.on_event("startup")
async def startup_event():
    thread = threading.Thread(target=load_models_task)
    thread.start()


@app.get("/status")
async def get_status():
    return {
        "is_loading": is_loading,
        "loading_complete": loading_complete,
        "error": loading_error
    }


@app.get("/")
async def root():
    return {"message": "EF Prediction API"}


@app.post("/predict")
async def predict(
        video: UploadFile,
        view: str,
        demographic_data: Dict,
        volume_tracings: Dict
):
    try:
        video_bytes = await video.read()
        temp_path = '/tmp/temp_video.mp4'
        with open(temp_path, 'wb') as f:
            f.write(video_bytes)

        video_features = extract_video_features(temp_path)

        demographic_features = process_demographic_data(demographic_data, volume_tracings, view)

        combined_input = {
            'input_layer': np.expand_dims(video_features, axis=0),
            'input_layer_1': np.expand_dims(demographic_features, axis=0)
        }

        # Select model and predict
        model = a4c_model if view == 'a4c' else psax_model
        prediction = model.predict(combined_input)

        os.remove(temp_path)

        return {
            "ef_prediction": int(prediction[0][0])
        }

    except Exception as e:
        return {"error": str(e)}
