import torch
from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import cv2
from scipy.spatial.distance import squareform
import numpy as np
import time
# from imutils.video import FPS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint, map_location='cpu')
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])



def get_boxes_labels(original_image, min_score, max_overlap, top_k, suppress=None):
    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    # print(predicted_locs.shape, predicted_scores.shape)

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # print(predicted_locs, predicted_scores)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    return det_boxes, det_labels, det_scores




def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    # print(predicted_locs.shape, predicted_scores.shape)

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # print(predicted_locs, predicted_scores)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels[i]])
        # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

    #     # Text
    #     text_size = font.getsize(det_labels[i].upper())
    #     text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
    #     textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
    #                         box_location[1]]
    #     draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
    #     draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
    #               font=font)
    del draw

    return annotated_image


if __name__ == '__main__':

    video_path = 'oxford_dataset/TownCentreXVID.avi'
    rect = np.array([[739, 119], [990, 148], [614, 523], [229, 437]], np.float32)

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)


    # video_path ='oxford_dataset/Dataset2.mp4'
    vc = cv2.VideoCapture(video_path)
    total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    frame_num = 0
    success = 1
    color = (255, 0, 0)
    thickness = 4

    while success:
        success, image = vc.read()
        if success:
            # You may need to convert the color.

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            # img_path = 'oxford_dataset/extracted_images/frame_3743.jpg'
            # original_image = Image.open(img_path, mode='r')
            original_image = im_pil.convert('RGB')
            start = time.time()
            det_boxes, det_labels, det_scores = get_boxes_labels(original_image, min_score=0.3, max_overlap=0.4, top_k=100)
            time_taken = time.time() - start
            print(time_taken)

            centers = torch.transpose(torch.stack(((det_boxes[:, 0] + det_boxes[:, 2])/2, (det_boxes[:, 1] + det_boxes[:, 3])/2)), 0, 1)

            dist_mat = torch.nn.functional.pdist(centers)
            dist_mat = torch.from_numpy(squareform(dist_mat.detach().numpy()))

            # print(det_labels)
            # a = detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200)
            # numpy_image = np.array(a)
            #
            # # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
            # # the color is converted from RGB to BGR format
            # opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            # cv2.imshow('test', opencv_image)

            for i in range(det_boxes.size(0)):
                color = (255, 0, 0)
                box = det_boxes[i].tolist()
                a = dist_mat[:, i] < 100
                if torch.sum(a.type(torch.ByteTensor)) > 1:
                    color = (0, 0, 255)
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness)
                cv2.circle(image, tuple(centers[i,:]), 5, (255, 255, 0), 10)
                image_2 = cv2.warpPerspective(image, M, (1920, 1080))

            cv2.imshow('test2', image_2)
            cv2.imshow('test', image)
            key = cv2.waitKey(1) & 0xFF
            if key & 0xFF == ord('q'):
                break

            if key == ord('n'):
                continue

    cv2.destroyAllWindows()

