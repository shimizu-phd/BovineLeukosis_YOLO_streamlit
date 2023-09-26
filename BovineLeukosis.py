import streamlit as st
from ultralytics import YOLO
import pandas as pd
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

select = st.selectbox('解析対象を選択してください.', ('画像', '動画'))

if select == '画像':
    uploaded_file = st.file_uploader("ファイルアップロード", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True)

    if uploaded_file:
        with st.spinner('解析中です...'):
            classes = []
            results_list = []

            for img in uploaded_file:
                image = Image.open(img)
                image = image.convert("RGB")
                image_show = np.array(image)

                results = model(image, stream=True, imgsz=1280)  # generator of Results objects
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

elif select == '動画':
    uploaded_file = st.file_uploader("ファイルアップロード", type=['avi', 'mp4'])
    classes = []

    if uploaded_file:
        # YOLOによる解析にファイルが必要なので、一時ファイルに保存する。拡張子が必要。byteデータでは解析できない。
        tfile = tempfile.NamedTemporaryFile(delete=True, suffix='.mp4', dir='./temp')
        tfile.write(uploaded_file.read())
        st.write(tfile.name)
        # ビデオを表示するようにbyteデータを用意する。一時ファイルを使うと自動で消去できず、エラーになる。
        video = uploaded_file.getvalue()
        st.video(video)
        # loading iconを表示する
        with st.spinner('解析中です...'):
            results = model.track(tfile.name, stream=True, imgsz=1280)  # generator of Results objects
            results_list = list(results)

        classes = []
        conf = []
        identity = []
        for each_img in results_list:
            if each_img.boxes.id is not None:
                classes += [int(i) for i in each_img.boxes.cls.tolist()]
                conf += [i for i in each_img.boxes.conf.tolist()]
                identity += [i for i in each_img.boxes.id.tolist()]

        df = pd.DataFrame({'class': classes, 'conf': conf, 'ID': identity})
        max_conf_id = df.groupby('ID')['conf'].idxmax()
        # フレームによって判定が違う細胞のclassを拾いたい場合
        # max_conf_id = df.groupby(['ID','class'])['conf'].idxmax()
        max_conf_df = df.iloc[max_conf_id]

        norm = max_conf_df['class'][max_conf_df['class'] == 0].count()
        abnorm = max_conf_df['class'][max_conf_df['class'] == 1].count()
        mono = max_conf_df['class'][max_conf_df['class'] == 2].count()
        gran = max_conf_df['class'][max_conf_df['class'] == 3].count()
        abnorm_ratio = abnorm / (abnorm + norm)

        st.write('正常リンパ球:', norm)
        st.write('異常リンパ球:', abnorm)
        st.write('単球:', mono)
        st.write('顆粒球:', gran)
        st.write('-------------------------')
        st.write('異常リンパ球の割合:', abnorm_ratio)

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
