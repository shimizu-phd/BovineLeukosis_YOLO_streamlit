import io

import streamlit as st
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import random
import os
import tempfile

tag = ['N', 'A', 'M', 'G']
colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 0, 255)]
n_class = len(tag)
img_size = 1280
n_result = 1

st.title('Bovine Leukosis Estimator')
st.write('伝染性牛リンパ腫の判定のため異常リンパ球を検出します.')
st.write('ギムザ染色した血液塗抹画像が必要です.')
st.write('対物レンズは40倍を推奨します.')
st.write('')


@st.cache_resource
def load_model():
    return YOLO('models/best.pt')


tab1, tab2 = st.tabs(["解析", "係数計算"])

with tab1:
    def plot_image(img_path, boxes, labels, color=None, line_thickness=None):
        # Plots predicted bounding boxes on the image
        tl = line_thickness or round(0.001 * max(img_path.shape[0:2])) + 1  # line/font thickness
        img = img_path
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for box, label in zip(boxes, labels):
            box = [int(i) for i in box]
            x, y, w, h = box
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            cv2.rectangle(img,
                          pt1=(x1, y1),
                          pt2=(x2, y2),
                          color=colors[label],
                          thickness=tl,
                          lineType=cv2.LINE_4)
            cv2.putText(img,
                        text=tag[label],
                        org=(x1, y1),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2,
                        color=colors[label],
                        thickness=tl,
                        lineType=cv2.LINE_4)
        return img


    model = load_model()

    uploaded_file = st.file_uploader("ファイルアップロード", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True)
    st.write('')
    st.write('倍率の係数を係数計算タブで計算してください.')
    mag = st.number_input('倍率の係数を入力してください.', value=1.0, step=0.1)


    if uploaded_file:
        classes = []
        results_list = []

        with st.spinner('解析中です...'):
            for img in uploaded_file:
                image = Image.open(img)
                image = image.convert("RGB")
                image_show = np.array(image)

                results = model(image, stream=True, imgsz=int(1280 / mag))  # generator of Results objects
                results_list += results

            for each_img in results_list:
                classes += [int(i) for i in each_img.boxes.cls.tolist()]

        # normal lymphocyte
        norm = classes.count(0)
        # abnormal lymphocyte
        abnorm = classes.count(1)
        # monocyte
        mono = classes.count(2)
        # Granulocytes
        gran = classes.count(3)
        # ratio
        if norm + abnorm == 0:
            abnorm_ratio = 0
        else:
            abnorm_ratio = abnorm / (abnorm + norm)

        st.write('正常リンパ球:', norm)
        st.write('異常リンパ球:', abnorm)
        st.write('単球:', mono)
        st.write('顆粒球:', gran)
        st.write('-------------------------')
        st.write('異常リンパ球の割合:', abnorm_ratio)
        st.write('-------------------------')
        if len(results_list) < 2:
            img_path1 = results_list[0].orig_img
            boxes1 = results_list[0].boxes.xywh
            labels1 = [int(i) for i in results_list[0].boxes.cls.tolist()]
            img1 = plot_image(img_path1, boxes1, labels1, color=None, line_thickness=None)
            st.write('サンプルを表示しています')
            st.image(img1, use_column_width=True)
            st.write('N: 正常リンパ球, A: 異常リンパ球, M: 単球, G: 顆粒球')
        else:
            n = random.sample(range(len(results_list)), 2)

            img_path1 = results_list[n[0]].orig_img
            img_path2 = results_list[n[1]].orig_img
            boxes1 = results_list[n[0]].boxes.xywh
            boxes2 = results_list[n[1]].boxes.xywh
            labels1 = [int(i) for i in results_list[n[0]].boxes.cls.tolist()]
            labels2 = [int(i) for i in results_list[n[1]].boxes.cls.tolist()]

            img1 = plot_image(img_path1, boxes1, labels1, color=None, line_thickness=None)
            img2 = plot_image(img_path2, boxes2, labels2, color=None, line_thickness=None)
            st.write('サンプルを2個表示しています')
            st.image([img1, img2], caption=['sample1', 'sample2'], use_column_width=True)
            st.write('N: 正常リンパ球, A: 異常リンパ球, M: 単球, G: 顆粒球')

with tab2:
    def resize(img, max_length):
        height, width = img.shape[:2]
        if height > width:
            new_height = max_length
            new_width = int((new_height / height) * width)
        else:
            new_width = max_length
            new_height = int((new_width / width) * height)
        resized_img = cv2.resize(img, (new_width, new_height))
        trimmed_img = resized_img[0:max_length, 0:max_length]
        return resized_img


    def trim(img, crop_width, crop_height):
        height, width = img.shape[:2]
        start_x = width // 2 - crop_width // 2
        start_y = height // 2 - crop_height // 2
        end_x = start_x + crop_width
        end_y = start_y + crop_height
        crop_img = img[start_y:end_y, start_x:end_x]
        return crop_img


    max_length = 1280
    crop_width = 320
    crop_height = 240

    uploaded_file = st.file_uploader("ファイルアップロード", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=False)
    st.write('解析したいファイルと同じシステムで撮影した赤血球の写った画像をアップロードして、リファレンスの画像と大きさを比較してください.')
    if uploaded_file:
        img_ref = Image.open('reference.jpg')
        img_ref = img_ref.convert("RGB")
        img_ref = np.array(img_ref)
        img_ref_resize = resize(img_ref, max_length)
        img_ref_trim = trim(img_ref_resize, crop_width, crop_height)
        image = Image.open(uploaded_file)
        image = image.convert("RGB")
        image = np.array(image)
        image_resize = resize(image, max_length)
        image_trim = trim(image_resize, crop_width, crop_height)
        st.image([img_ref_trim, image_trim], caption=['リファレンス', 'アップロード画像'], use_column_width=True)
        st.write('リファレンスの赤血球の直径を１としたときの入力画像の赤血球の直径を計算し、係数としてください.')
