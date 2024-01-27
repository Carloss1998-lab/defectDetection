import numpy as np
import json
import boto3
import copy
import os
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from PIL import Image, ImageColor
import streamlit as st
from io import BytesIO

def query_Type2(runtime,image_file_name, endpoint_name, num_predictions=3):


    try:
        with open(image_file_name, "rb") as file:
            input_img_rb = file.read()
    except:
        input_img_rb = image_file_name.getvalue()
    
    query_response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/x-image",
        Body=input_img_rb,
        Accept=f'application/json;verbose;n_predictions={num_predictions}'
    )
    # If we remove ';n_predictions={}' from Accept, we get all the predicted boxes.
    query_response = query_response['Body'].read()

    model_predictions = json.loads(query_response)
    normalized_boxes, classes, scores, labels = (
        model_predictions["normalized_boxes"],
        model_predictions["classes"],
        model_predictions["scores"],
        model_predictions["labels"],
    )
    # Substitute the classes index with the classes name
    class_names = [labels[int(idx)] for idx in classes]
    return normalized_boxes, class_names, scores


# Copied from albumentations/augmentations/functional.py
# Follow albumentations.Normalize, which is used in sagemaker_defect_detection/detector.py
def normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


def query_DDN(runtime,image_file_name, endpoint_name, num_predictions=3):

    with Image.open(image_file_name) as im:
        image_np = np.array(im)

    # Follow albumentations.Normalize, which is used in sagemaker_defect_detection/detector.py
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    max_pixel_value = 255.0

    image_np = normalize(image_np, mean, std, max_pixel_value)
    image_np = image_np.transpose(2, 0, 1)
    image_np = np.expand_dims(image_np, 0) # CHW

    query_response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(image_np.tolist())
    )
    query_response = query_response["Body"].read()

    model_predictions = json.loads(query_response.decode())[0]
    unnormalized_boxes = model_predictions['boxes'][:num_predictions]
    class_names = model_predictions['labels'][:num_predictions]
    scores = model_predictions['scores'][:num_predictions]
    return unnormalized_boxes, class_names, scores



def query_Type1(runtime,image_file_name, endpoint_name, num_predictions=3):

    with open(image_file_name, "rb") as file:
        input_img_rb = file.read()

    query_response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/x-image",
        Body=input_img_rb
    )
    query_response = query_response["Body"].read()

    model_predictions = json.loads(query_response)['prediction'][:num_predictions]
    class_names = [int(pred[0])+1 for pred in model_predictions]  # +1 for index starts from 1
    scores = [pred[1] for pred in model_predictions]
    normalized_boxes = [pred[2:] for pred in model_predictions]
    return normalized_boxes, class_names, scores


def plot_result(image, categories, d):
    # d - dictionary of endpoint responses

    colors = list(ImageColor.colormap.values())
    with Image.open(image) as im:
        image_np = np.array(im)
    fig = plt.figure(figsize=(25, 25),facecolor = 'black', dpi = 96)


    # Predictions
    counter = 2
    for k, v in d.items():
        axi = fig.add_subplot(2, 3, counter)
        counter += 1

        if "Type2-HPO" in k:
            k = "Type2-HPO"
        elif "Type2" in k:
            k = "Type2"
        elif "Type1-HPO" in k:
            k = "Type1-HPO"
        elif "Type1" in k:
            k = "Type1"
        elif k.endswith("finetuned-endpoint"):
            k = "DDN"

        plt.title(f'Prediction: {k}')
        plt.axis('off')

        for idx in range(len(v['normalized_boxes'])):
            left, bot, right, top = v['normalized_boxes'][idx]
            if k == 'DDN':
                x, w = left, right - left
                y, h = bot, top - bot
            else:
                x, w = [val * image_np.shape[1] for val in [left, right - left]]
                y, h = [val * image_np.shape[0] for val in [bot, top - bot]]
            color = colors[hash(v['classes_names'][idx]) % len(colors)]
            rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=color, facecolor="none")
            axi.add_patch(rect)
            axi.text(x, y,
                "{} {:.0f}%".format(categories[v['classes_names'][idx]], v['confidences'][idx] * 100),
                bbox=dict(facecolor="red", alpha=0.5),
            )

        axi.imshow(image_np)

    plt.tight_layout()
    
    
    return fig

def query_endpoint(runtime,image,od_type2_hpo_endpoint_name,categories):

    d = {}
    # Inference. Could find all endpoints from Inference / Endpoints in the Sagemaker Dashboard
    for endpoint_name in [
        od_type2_hpo_endpoint_name,
    ]:
        query_function = query_Type2 if "Type2" in endpoint_name else query_Type1
        normalized_boxes, classes_names, confidences = query_function(
            runtime, image, endpoint_name=endpoint_name
        )
        d[endpoint_name] = {
            "normalized_boxes": normalized_boxes,
            "classes_names": classes_names,
            "confidences": confidences,
        }
    return  plot_result(image, categories, d)
