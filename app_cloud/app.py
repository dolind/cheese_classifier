# import os
# os.system("pip install -r app_cloud/requirements.txt")
#
#
# import gradio as gr
# from fastrtc import Stream, AdditionalOutputs, get_hf_turn_credentials
# from fastai.vision.all import *
#
# # Retrieve the secret stored in HF Spaces
# credentials = os.getenv("HF_TOKEN")
#
# if credentials is None:
#     raise ValueError("HF_TOKEN secret not found!")
# credentials = get_hf_turn_credentials(token=credentials)
# # Load classification models
# models = {
#     'base': load_learner('models/base.pkl'),
#     'tiny': load_learner('models/tiny.pkl'),
#     'resnet': load_learner('models/resnet.pkl')
# }
#
# frame_counter = 0
# FRAME_SKIP = 1  # Start with no skipping, will adjust dynamically
# last_classification = None
# TARGET_FPS = 30  # Adjust to desired FPS
#
#
# # Function to classify an image with the selected model
# def classify(model_name, image):
#     learn = models[model_name]
#     pred, _, probs = learn.predict(image)
#     labels = learn.dls.vocab
#     return {labels[i]: float(probs[i]) for i in range(len(labels))}
#
# # Function to process webcam frames
# def process_frame(image, model_name):
#     global frame_counter, last_classification, FRAME_SKIP\
#
#     frame_counter += 1  # Update counter
#
#     # Skip frames based on FRAME_SKIP
#     if frame_counter % FRAME_SKIP != 0:
#         return image, AdditionalOutputs(last_classification)
#
#     # Measure start time
#     start_time = time.time()
#
#     # Convert image for classification
#     image_pil = PILImage.create(image)
#     last_classification = classify(model_name, image_pil)
#
#     # Measure inference time
#     last_inference_time = round((time.time() - start_time) * 1000, 2)  # Convert to milliseconds
#     # print(last_inference_time)
#     # Adjust FRAME_SKIP dynamically to match target FPS
#     if last_inference_time > (1000 / TARGET_FPS):  # If inference is slower than target FPS
#         FRAME_SKIP = min(FRAME_SKIP + 1, TARGET_FPS)  # Increase skipping
#     else:
#         FRAME_SKIP = max(FRAME_SKIP - 1, 1)  # Decrease skipping
#
#     # Return image and classification results with inference time
#     return image, AdditionalOutputs(last_classification)
#
#
# # Gradio Streaming
# stream = Stream(
#     handler=process_frame,
#     rtc_configuration=credentials,
#     modality="video",
#     mode="send-receive",
#     additional_inputs=[
#         gr.Radio(choices=['base', 'tiny', 'resnet'], value='base', label="Select Model")
#     ],
#     additional_outputs=[gr.Label(label="Prediction")],  # Define additional outputs
#     additional_outputs_handler=lambda component, additional_output: additional_output
# )
#
# stream.ui.launch()


import gradio as gr
import numpy as np
import cv2

def transform_cv2(frame, transform):
    if transform == "cartoon":
        # prepare color
        img_color = cv2.pyrDown(cv2.pyrDown(frame))
        for _ in range(6):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
        img_color = cv2.pyrUp(cv2.pyrUp(img_color))

        # prepare edges
        img_edges = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        img_edges = cv2.adaptiveThreshold(
            cv2.medianBlur(img_edges, 7),
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9,
            2,
        )
        img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)
        # combine color and edges
        img = cv2.bitwise_and(img_color, img_edges)
        return img
    elif transform == "edges":
        # perform edge detection
        img = cv2.cvtColor(cv2.Canny(frame, 100, 200), cv2.COLOR_GRAY2BGR)
        return img
    else:
        return np.flipud(frame)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            transform = gr.Dropdown(choices=["cartoon", "edges", "flip"],
                                    value="flip", label="Transformation")
            input_img = gr.Image(sources=["webcam"], type="numpy")
        with gr.Column():
            output_img = gr.Image(streaming=True)
        dep = input_img.stream(transform_cv2, [input_img, transform], [output_img],
                                time_limit=30, stream_every=0.1, concurrency_limit=30)

demo.launch()
