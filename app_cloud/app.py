import os
import gradio as gr
import numpy as np
import time
from fastai.vision.all import *

# Load classification models
models = {
    'base': load_learner('models/base.pkl'),
    'tiny': load_learner('models/tiny.pkl'),
    'resnet': load_learner('models/resnet.pkl')
}

frame_counter = 0
FRAME_SKIP = 1
last_classification = None
TARGET_FPS = 10
stream_every_dynamic = 0.3  # Initial stream rate


# Function to classify an image
def classify(model_name, image):
    learn = models[model_name]
    pred, _, probs = learn.predict(image)
    labels = learn.dls.vocab
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# Function to process frames and return classification + inference time
def process_frame(frame, model_name):
    global frame_counter, last_classification, FRAME_SKIP,  stream_every_dynamic

    frame_counter += 1

    # Start timing
    start_time = time.time()

    # Convert image to PIL format and classify
    image_pil = PILImage.create(frame)
    last_classification = classify(model_name, image_pil)

    # Measure inference time
    inference_time = round((time.time() - start_time) * 1000, 2)  # Convert to milliseconds

    # Adjust FPS dynamically
    if inference_time > (1000 / TARGET_FPS):
        FRAME_SKIP = min(FRAME_SKIP + 1, 30)
    else:
        FRAME_SKIP = max(FRAME_SKIP - 1, 1)

    # Adjust stream_every dynamically
    stream_every_dynamic = max(inference_time / 1000, 0.1)  # Ensure it's at least 100ms

    return last_classification, f"{inference_time:.2f} ms"

# Gradio UI
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            model_select = gr.Radio(choices=['base', 'tiny', 'resnet'],
                                    value='base', label="Select Model")
            input_img = gr.Image(sources=["webcam"], type="numpy")

        with gr.Column():
            prediction_output = gr.Label(label="Prediction")
            inference_time_output = gr.Textbox(label="Inference Time (ms)")

        dep = input_img.stream(process_frame, [input_img, model_select],
                               [prediction_output, inference_time_output],
                               time_limit=30, stream_every=stream_every_dynamic, concurrency_limit=1)

demo.launch()
