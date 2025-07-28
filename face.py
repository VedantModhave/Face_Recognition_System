import cv2
import numpy as np
import onnxruntime as ort
import torch
import os
from collections import OrderedDict
from tqdm import tqdm  
from torchvision.models import resnet50, ResNet50_Weights
from models.retinaface import RetinaFace
from data.config import cfg_re50, cfg_mnet
from utils.box_utils import decode, decode_landm
from layers.functions.prior_box import PriorBox

# -------- Config -------- #
ONNX_MODEL_PATH = "facenet512.onnx"
THRESHOLD = 0.55                  # Cosine similarity threshold
DETECTION_PROB_THRESHOLD = 0.90   # RetinaFace confidence threshold
RETINAFACE_NETWORK = "resnet50"   # or "mobilenet0.25"
RETINAFACE_MODEL_PATH = "weights/Resnet50_Final.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- RetinaFace Init -------- #
if RETINAFACE_NETWORK == "resnet50":
    cfg = cfg_re50
elif RETINAFACE_NETWORK == "mobilenet0.25":
    cfg = cfg_mnet
else:
    raise ValueError("Unsupported network.")

net = RetinaFace(cfg=cfg, phase='test')
state_dict = torch.load(RETINAFACE_MODEL_PATH, map_location=lambda storage, loc: storage, weights_only=True)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace('module.', '') if k.startswith('module.') else k
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)
net.eval()
if DEVICE == "cuda":
    net = net.cuda()

# -------- FaceNet Init -------- #
session = ort.InferenceSession("facenet512.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

def preprocess(img):
    img = cv2.resize(img, (160, 160))
    img = img.astype(np.float32)
    mean, std = img.mean(), img.std()
    img = (img - mean) / (std + 1e-6)
    img = np.expand_dims(img, axis=0)
    return img

def get_embedding(img):
    img = preprocess(img)
    emb = session.run(None, {input_name: img})[0][0]
    return emb / np.linalg.norm(emb)

def cosine_similarity(a, b):
    return np.dot(a, b)

def extract_faces(frame):
    img = np.float32(frame)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([im_width, im_height, im_width, im_height])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    if DEVICE == "cuda":
        img = img.cuda()
        scale = scale.cuda()
    with torch.no_grad():
        loc, conf, landms = net(img)
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        if DEVICE == "cuda":
            priors = priors.cuda()
        boxes = decode(loc.data.squeeze(0), priors, cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    inds = np.where(scores > DETECTION_PROB_THRESHOLD)[0]
    boxes = boxes[inds]
    scores = scores[inds]
    faces = []
    for box, prob in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        face = frame[y1:y2, x1:x2]
        if face.size > 0:
            faces.append((face, prob))
    return faces

def seconds_to_timestamp(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02}:{m:02}:{s:02}"

def main(reference_img_path, video_path):
    if not os.path.exists(reference_img_path) or not os.path.exists(video_path):
        print("Reference image or video not found.")
        return

    print("[+] Loading reference image...")
    ref_img = cv2.imread(reference_img_path)
    ref_faces = extract_faces(ref_img)

    if not ref_faces:
        print("No face detected in reference image.")
        return

    ref_face, ref_prob = ref_faces[0]
    ref_embedding = get_embedding(ref_face)

    print(f"[+] Reference face detected with probability: {ref_prob:.2f}")

    print("[+] Processing video...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    matched_frames = []
    detection_probs = []

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = extract_faces(frame)

            for face, prob in faces:
                try:
                    emb = get_embedding(face)
                    sim = cosine_similarity(ref_embedding, emb)

                    if sim > THRESHOLD:
                        timestamp_sec = frame_idx / fps
                        matched_frames.append(timestamp_sec)
                        detection_probs.append(prob)
                        break 
                except Exception:
                    continue

            frame_idx += 1
            pbar.update(1)
    cap.release()

    if not matched_frames:
        print("No matching face found in the video.")
        return

    ranges = []
    start = matched_frames[0]
    for i in range(1, len(matched_frames)):
        if matched_frames[i] - matched_frames[i - 1] > 1.5:
            end = matched_frames[i - 1]
            ranges.append((start, end))
            start = matched_frames[i]
    ranges.append((start, matched_frames[-1]))

    avg_prob = np.mean(detection_probs)

    print(f"\n Face matched in {len(matched_frames)} frames.")
    print(f"   Average detection probability of matched faces: {avg_prob:.2f}")
    print(f"   First appearance: {seconds_to_timestamp(matched_frames[0])}")
    print(f"   Last appearance: {seconds_to_timestamp(matched_frames[-1])}")
    print("\n Appearance intervals:")
    for start, end in ranges:
        print(f"   {seconds_to_timestamp(start)} → {seconds_to_timestamp(end)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to reference face image")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    args = parser.parse_args()

    main(args.image, args.video)