import os
from time import sleep
import time
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import ElementNotInteractableException
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
import pandas as pd
import numpy as np
import os
import cv2
import onnxruntime as rt
import onnx
import urllib
from PIL import Image

from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# headless 모드로 실행
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')

driver = webdriver.Chrome(options=options, executable_path="./chromedriver_win32/chromedriver.exe")

psswd = 'my_password'

url = 'my_router_url'
driver.get(url)

# a태그 클릭
a = driver.find_element_by_css_selector('#form1 > div.midarea > div > div > dl.deviceinfo > dt > a')
a.click()

# 패스워드 입력
password = driver.find_element_by_css_selector('#form1 > div.modalpop > div.popupwrap.deviceconnect > div > p:nth-child(7) > input')

# CAPTCHA 문자열 입력
captcha_str = driver.find_element_by_css_selector('#captcha_str')

captcha_img = driver.find_element_by_css_selector('#captcha_img')
url = captcha_img.get_attribute('src') 
print(url)

# 이미지파일 저장 경로
img_file = "captcha_img.gif"

urllib.request.urlretrieve(url, img_file)

# cv에서 읽을 수 있는 png로 변경
with Image.open('captcha_img.gif') as im:
    im.save('captcha_img.png')

# 로그인 버튼 정의
login_button = driver.find_element_by_css_selector('#form1 > div.modalpop > div.popupwrap.deviceconnect > div > p.btncenter > button:nth-child(3)')

img_width = 200
img_height = 50

characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
dst_ = np.array([1,0,2])

# 그레이 스케일로 인코딩
def encode_single_sample_opencv0(img_path):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape([70,200,1]) ## img == tfimg

    img = np.float32(img)
    img = img/255.
    img = cv2.resize(img, (img_width, img_height))

    img = cv2.transpose(img, dst_)

    return img

def CtoN(a): 
    tmp = a.lower()
    res = ord(tmp) - ord('a') +1
    return res

def NtoC(num):
    characters = ['','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '']
    return characters[num]

def deleter(l):
    new = [l[0]]
    
    for i in range(1, len(l)):
        if l[i] == '':
            new.append(l[i])
        elif l[i] != new[-1]:
            new.append(l[i])
    return ''.join(new)

# 디코딩
def decoder_made(pred):
    ll = []
    for z in pred[0][0].tolist():
        ll.append(NtoC(z.index(max(z))))
        
    return deleter(ll)


input_array = encode_single_sample_opencv0('captcha_img.png')
input_array = input_array.reshape([1,200,50,1])


output_path = "./ONNX/onnx_model.onnx"
onnx_model = onnx.load(output_path) # .onnx 파일 불러오기

output_names = [n.name for n in onnx_model.graph.output]
providers = ['CPUExecutionProvider']
m = rt.InferenceSession(output_path, providers=providers) # onnx runtime으로 실행
onnx_pred = m.run(output_names, {'input': input_array})
result = decoder_made(onnx_pred) # 디코딩
print('ONNX Predicted:', result)


password.send_keys(psswd) # 패스워드 입력
captcha_str.send_keys(result) # 디코딩한 값을 입력
login_button.click()
time.sleep(1)

okay_button = driver.find_element_by_css_selector('#proceed_alert')
okay_button.click()
time.sleep(2)

wait = WebDriverWait(driver, 10)
button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#wol_btn1")))
button.click()

driver.close()

