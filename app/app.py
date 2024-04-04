# app/app.py
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '../uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# model = load_model('../model/best_model.h5', compile=False)  # Sesuaikan dengan nama model Anda
# model = load_model('../model/best_model_complex.h5', compile=False)  # Sesuaikan dengan nama model Anda
# model = load_model('../model/best_model_1k_v1.h5', compile=False)  # Sesuaikan dengan nama model Anda
model = load_model('../model/best_model_1k_v2.h5', compile=False)  # Sesuaikan dengan nama model Anda

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def apply_clahe_green(image):
    green_channel = image[:, :, 1]
    clahe_green = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(green_channel)
    return clahe_green

def preprocess_image(img_path, img_dim=(256, 256)):
    image = cv2.imread(img_path, -1)
    clahe_green = apply_clahe_green(image)
    resized_image = cv2.resize(clahe_green, (img_dim[1], img_dim[0]))
    normalized_image = resized_image / 255.0  # Normalize to [0, 1]
    input_image = np.expand_dims(normalized_image, axis=-1)
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
    return input_image

def segment_single_image(model, img_path):
    input_image = preprocess_image(img_path)
    # Perform segmentation using the model
    predicted_mask = model.predict(input_image)
    # Threshold the predicted mask if needed
    threshold = 0.5  # Adjust as needed
    segmented_mask = (predicted_mask > threshold).astype(np.uint8)
    return segmented_mask.squeeze()

# def save_images(original, green_channel, clahe_green, segmented_mask):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(green_channel, cmap='gray')
    plt.title('Green Channel')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(clahe_green, cmap='gray')
    plt.title('CLAHE on Green Channel')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(segmented_mask, cmap='gray')
    plt.title('Segmented Mask')
    plt.axis('off')

    plt.savefig('static/result.png')
    plt.savefig('static/original.png')
    plt.savefig('static/green_channel.png')
    plt.savefig('static/clahe_green.png')
    plt.close()
def save_images(original, green_channel, clahe_green, segmented_mask):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(green_channel, cmap='gray')
    plt.title('Green Channel')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(clahe_green, cmap='gray')
    plt.title('CLAHE on Green Channel')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(segmented_mask, cmap='gray')
    plt.title('Segmented Mask')
    plt.axis('off')

    # Crop the figure tightly
    plt.tight_layout()

    # Save only the part of the figure without extra white space
    plt.savefig('static/result.png', bbox_inches='tight', pad_inches=0.0)
    plt.close()

def extract_8x8_matrix(image):
    return image[:8, :8]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Perform segmentation on the single image
            segmented_mask = segment_single_image(model, filepath)

            # Display the original image and the segmented mask using matplotlib
            original_image = cv2.imread(filepath)
            green_channel = original_image[:, :, 1]
            clahe_green = apply_clahe_green(original_image)

            # Save all images
            save_images(original_image, green_channel, clahe_green, segmented_mask)

            # Extract 8x8 matrices
            original_matrix = extract_8x8_matrix(original_image)
            green_channel_matrix = extract_8x8_matrix(green_channel)
            clahe_green_matrix = extract_8x8_matrix(clahe_green)

            # Check if there are white areas (cracks) in the segmented mask
            if np.any(segmented_mask == 1):
                result_text = "Retak"
            else:
                result_text = "Tidak Retak"

            # return render_template('index.html', filename=filename)
            # Render the template with filename and matrices
            return render_template('index.html', filename=filename,
                                original_matrix=original_matrix.tolist(),
                                green_channel_matrix=green_channel_matrix.tolist(),
                                clahe_green_matrix=clahe_green_matrix.tolist(),
                                result_text=result_text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
