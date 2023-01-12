import cv2
from torchvision import models
from torchvision import transforms
import torch
import torch.nn as nn
import re

red= (0, 0, 255)
font =  cv2.FONT_HERSHEY_PLAIN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 가로
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 세로

# model
net = models.resnet18(pretrained=False)
net.fc = nn.Linear(in_features=512, out_features=10)

# 학습한 모델 불러오기
models_loader_path = "/Users/sihyun/Downloads/code/best.pt"
net.load_state_dict(torch.load(models_loader_path, map_location=device))
net.to(device)

val_transforms = transforms.Compose([
    transforms.Resize((224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def preprocess(image, device):
    from PIL import Image
    image = Image.fromarray(image)
    image = val_transforms(image)
    image = image.float()
    image = image.to(device)
    image = image.unsqueeze(0)

    return image


if not webcam.isOpened():
    print("카메라 열기 실패 !!!")
    exit()

while webcam.isOpened():
    status, frame = webcam.read()
    frame = cv2.flip(frame, 1)  # 좌우 대칭
    net.eval()
    with torch.no_grad():
        if status:
            image = preprocess(frame, device=device)
            output = net(image)
            _, argmax = torch.max(output, 1)
            print("output", argmax)

            num = int(argmax[0])

            if num == 0:
                name = 'rock'
            elif num == 1:
                name = 'paper'
            else:
                name = 'scissors'

            frame = cv2.putText(frame, name, (350, 40), font, 2, red, 1, cv2.LINE_AA)
            cv2.imshow("test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()
