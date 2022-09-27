import cv2
import tkinter
import torch
from PIL import Image
from PIL import ImageTk
import numpy as np
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from torchvision.models import ShuffleNet_V2_X0_5_Weights

# 모델 불러오기
model = torch.jit.load('model_scripted.pt')
model.eval()
transform = ShuffleNet_V2_X0_5_Weights.DEFAULT.transforms()
tot = ToTensor()

# 카메라 번호 보통 0은 후면, 1은 전면
CAMERA_NUM = 0

# 화면 열기
window=tkinter.Tk()
window.title("얼굴상 테스트")
window.geometry("500x800+10+10")

# 카메라 할당
capture = cv2.VideoCapture(CAMERA_NUM)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 카메라 보이기
def show_camera():
  global tempimage
  # ret: 프레임이 올바로 읽혔는지 여부
  # frame: 읽은 화면
  ret, frame = capture.read()
  if not ret:
    return
  # CV로 읽은 화면은 BGR형식이므로 RGB로 변환
  img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Numpy배열을 Image객체로 변환 후 tkinter와 호환되는 객체로 변환
  img = Image.fromarray(img)
  imgtk = ImageTk.PhotoImage(image=img)
  tempimage = imgtk
  label.config(image=tempimage)
  label.after(50, show_camera)

def Classify_image():
    global tempimage
    im = ImageTk.getimage(tempimage).convert("RGB")
    im = tot(im)
    im = im.unsqueeze(0)  # (1,3,360,480)
    im = transform(im) # 360*480 -> 28*28 (1, 3,28,28)
    prediction = model(im).squeeze(0).softmax(0) # [[0.98],[0.01],[0.01]] => [0.98,0.01,0.01]
    sorted_prediction = prediction.argsort(descending=True)[:5]
    name=[]
    percentage=[]
    for class_id in sorted_prediction:
        score = prediction[class_id].item()
        category_name = ['bear', 'dog', 'fox'][class_id]
        name.append(category_name)
        percentage.append(score*100)
    # 그래프 그리기
    figure, ax = plt.subplots(figsize=(2, 4))
    sns.barplot(x=percentage,y=name)
    sns.despine(left=True, bottom=True)
    ax.set(xticks=[])
    ax.bar_label(ax.containers[0],fmt="%.1f%%")
		# 캔버스에 그래프를 업데이트하여 그리기
    canvas.figure = figure
    canvas.draw()


# 카메라 열고 닫기
def toggle_capture():

  ret, _ = capture.read()
  if ret: 
    button.config(text="다시 찍기")
    capture.release()
    Classify_image()
  else:
    button.config(text="사진 찍기")
    capture.open(CAMERA_NUM)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    show_camera()

# GUI 위젯 배치
label=tkinter.Label(window)
label.pack(side="top")
button = tkinter.Button( 
    window,  # 대상이 되는 창
    text="사진 찍기", # 버튼에 표시할 텍스트
    overrelief="solid",  # 버튼에 마우스를 올렸을 때 모양
    width=15, # 너비
    command=toggle_capture, # 버튼을 눌렀을 때 실행될 함수
    
    repeatdelay=1000, # 눌러진 상태에서 반복될 때까지 딜레이
    repeatinterval=100 ) # 눌러진 상태에서 반복될 시간 간격
button.pack()

canvas = FigureCanvasTkAgg(master=window)
canvas.get_tk_widget().pack()

# 카메라에 담기는 이미지 변수
tempimage=None
show_camera()
window.mainloop()
capture.release()


