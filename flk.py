from flask import Flask, request, make_response, render_template
import torch
import numpy as np
import cv2
from PIL import Image
import io
from skimage.metrics import structural_similarity as compare_ssim
import random

app = Flask(__name__)

def process_image(file):
    image = Image.open(file).convert('RGB')
    image = np.array(image)
    if image.shape[2] == 4:  # Remove alpha channel if present
        image = image[:, :, :3]
    return image

def obtain_change_map(pre, post, neighborhood=5, excluded=0):
    B, C, H_pre, W_pre = pre.shape
    _, _, H_post, W_post = post.shape

    padded_pre = torch.zeros((B, C, H_pre + 2 * neighborhood, W_pre + 2 * neighborhood))
    padded_pre[:, :, neighborhood:H_pre + neighborhood, neighborhood:W_pre + neighborhood] = pre
    padded_post = torch.zeros((B, C, H_pre + 2 * neighborhood, W_pre + 2 * neighborhood))
    padded_post[:, :, neighborhood:H_post + neighborhood, neighborhood:W_post + neighborhood] = post

    num_neighbors = (2 * neighborhood + 1) ** 2 - (2 * excluded + 1) ** 2

    pre_response = padded_pre ** 2
    post_response = padded_pre * padded_post
    pre_sum = torch.zeros(post.shape)
    post_sum = torch.zeros(post.shape)

    for x_patch in range(-neighborhood, neighborhood + 1):
        for y_patch in range(-neighborhood, neighborhood + 1):
            if abs(x_patch) <= excluded or abs(y_patch) <= excluded:
                continue

            pre_sum += pre_response[:, :, y_patch + neighborhood:H_pre + y_patch + neighborhood, x_patch + neighborhood:W_pre + x_patch + neighborhood]
    
            post_sum += post_response[:, :, y_patch + neighborhood:H_post + y_patch + neighborhood, x_patch + neighborhood:W_post + x_patch + neighborhood]

    post_pred = pre * post_sum / pre_sum
    change_map = torch.abs(post_pred - post)
    return change_map

def apply_threshold(change_map, method='Otsu', otsu_factor=1.0):
    B, C, H, W = change_map.shape
    change_map_binary = torch.zeros((B, C, H, W))
    
    for b in range(B):
        for c in range(C):
            j = change_map[b, c].numpy()
            if method == 'Otsu':
                t = cv2.threshold(np.array(abs(j * 255), dtype=np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
                change_map_binary[b, c] = torch.where(abs(change_map[b, c]) > (t * otsu_factor / 255), torch.tensor(1.0), torch.tensor(0.0))
            elif method == 'Triangle':
                t = cv2.threshold(np.array(abs(j * 255), dtype=np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)[0]
                change_map_binary[b, c] = torch.where(abs(change_map[b, c]) > (t * 0.5 * otsu_factor / 255), torch.tensor(1.0), torch.tensor(0.0))
            else:
                raise ValueError("Unsupported thresholding method")
    
    return change_map_binary

def compare_images(imageA, imageB):
    imageA = torch.tensor(imageA).float() / 255.0
    imageB = torch.tensor(imageB).float() / 255.0

    imageA = imageA.permute(2, 0, 1).unsqueeze(0)
    imageB = imageB.permute(2, 0, 1).unsqueeze(0)

    change_map = obtain_change_map(imageA, imageB)
    
    thresholded_change_map = apply_threshold(change_map, method='Otsu', otsu_factor=1.0)

    change_map_np = thresholded_change_map.squeeze(0).permute(1, 2, 0).numpy() * 255
    change_map_np = change_map_np.astype(np.uint8)

    edges = cv2.Canny(change_map_np, 100, 200)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    imageA_with_boxes = imageA.squeeze().permute(1, 2, 0).numpy() * 255
    imageB_with_boxes = imageB.squeeze().permute(1, 2, 0).numpy() * 255

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(imageA_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(imageB_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)

    combined_image = np.hstack((imageA_with_boxes, imageB_with_boxes, change_map_np))

    grayA = cv2.cvtColor(imageA.squeeze().permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB.squeeze().permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_BGR2GRAY)
    score, _ = compare_ssim(grayA, grayB, full=True)

    print(f"SSIM: {score}")

    change_coords = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if score > 0.9 and score <1:
            change_coords.append((x, y, w, h))
            if len(change_coords) >= 100:
                break
    
    if score == 1.0:
        print("Change Coordiantes: []")
    else:
        print(f"Change Coordinates: {change_coords}")

    return combined_image

@app.route('/', methods=['GET', 'POST'])
def upload_images():
    if request.method == 'POST':
        file1 = request.files.get('image1')
        file2 = request.files.get('image2')

        if not file1 or not file2:
            return "Please upload two images for comparison.", 400

        imageA = process_image(file1)
        imageB = process_image(file2)

        try:
            combined_image = compare_images(imageA, imageB)
        except Exception as e:
            print(f"Error: {e}")
            return "An error occurred during image processing.", 500

        _, buffer = cv2.imencode('.png', combined_image)
        response = make_response(buffer.tobytes())
        response.headers['Content-Type'] = 'image/png'
        return response

    return render_template('upload.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)