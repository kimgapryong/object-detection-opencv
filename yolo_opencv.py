import argparse
from ultralytics import YOLO
import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', required=True, help='Path to input image')
parser.add_argument('-m', '--model', default='best.pt', help='YOLOv8 classification model to use')
args = parser.parse_args()


model = YOLO(args.model)

img = cv2.imread(args.image)

if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없음: {args.image}")


if len(img.shape) == 2 or img.shape[2] == 1:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


results = model(args.image)


probs = results[0].probs.data
top3_idx = np.argsort(probs)[::-1][:3]
top3 = [(model.names[i], probs[i]) for i in top3_idx]


for idx, (cls_name, prob) in enumerate(top3):
    label = f"{idx+1}. {cls_name}: {prob:.2f}"
    y = 40 + idx * 40
    cv2.putText(img, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)


cv2.imshow("Character Classification (Top 3)", img)
cv2.imwrite("character-classified.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("✅ 예측 결과:")
for rank, (name, p) in enumerate(top3, 1):
    print(f"{rank}. {name} ({p:.2f})")

print("✅ 결과 저장 완료: character-classified.jpg")
