import os
os.system("pip install -r app_cloud/requirements.txt")


import gradio as gr
from fastrtc import Stream, AdditionalOutputs, get_hf_turn_credentials
from fastai.vision.all import *

credentials = get_hf_turn_credentials()
# Load classification models
models = {
    'base': load_learner('models/base.pkl'),
    'tiny': load_learner('models/tiny.pkl'),
    'resnet': load_learner('models/resnet.pkl')
}

frame_counter = 0
FRAME_SKIP = 1  # Start with no skipping, will adjust dynamically
last_classification = None
TARGET_FPS = 30  # Adjust to desired FPS


# Function to classify an image with the selected model
def classify(model_name, image):
    learn = models[model_name]
    pred, _, probs = learn.predict(image)
    labels = learn.dls.vocab
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# Function to process webcam frames
def process_frame(image, model_name):
    global frame_counter, last_classification, FRAME_SKIP\

    frame_counter += 1  # Update counter

    # Skip frames based on FRAME_SKIP
    if frame_counter % FRAME_SKIP != 0:
        return image, AdditionalOutputs(last_classification)

    # Measure start time
    start_time = time.time()

    # Convert image for classification
    image_pil = PILImage.create(image)
    last_classification = classify(model_name, image_pil)

    # Measure inference time
    last_inference_time = round((time.time() - start_time) * 1000, 2)  # Convert to milliseconds
    # print(last_inference_time)
    # Adjust FRAME_SKIP dynamically to match target FPS
    if last_inference_time > (1000 / TARGET_FPS):  # If inference is slower than target FPS
        FRAME_SKIP = min(FRAME_SKIP + 1, TARGET_FPS)  # Increase skipping
    else:
        FRAME_SKIP = max(FRAME_SKIP - 1, 1)  # Decrease skipping

    # Return image and classification results with inference time
    return image, AdditionalOutputs(last_classification)


# Gradio Streaming
stream = Stream(
    handler=process_frame,
    modality="video",
    mode="send-receive",
    additional_inputs=[
        gr.Radio(choices=['base', 'tiny', 'resnet'], value='base', label="Select Model")
    ],
    additional_outputs=[gr.Label(label="Prediction")],  # Define additional outputs
    additional_outputs_handler=lambda component, additional_output: additional_output
)

stream.ui.launch()