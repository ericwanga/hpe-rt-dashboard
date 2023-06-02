import os
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta, tzinfo
import glob  # get image files
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
import streamlit as st
from millify import millify
# import sys
# from IPython.display import clear_output
# import math
# import plotly.express as px
# import mediapipe as mp
# from mediapipe.python.solutions import pose as mp_pose

st.set_page_config(page_title='Real-time group dynamics dashboard v0.1', page_icon='Video', layout='wide')

# set model
yolo_model = None
cfg_model_path = 'models/yolov5s.pt'
confidence = .55
device_option = 'cpu'

# set path
sample_image_path = r'C:\Users\EricWang\Pictures\sample_images'
# sample_video_path = r'C:\Users\EricWang\Videos'

# set image source
# sample_images = [cv2.imread(file) for file in glob.glob(os.path.join(sample_image_path, '*.png'))]
# print('INFO: {} sample images'.format(len(sample_images)))

area_threshold = 43000
ymin = 150
storage = {}
# df_result_final = pd.DataFrame()
# placeholder = st.empty()

# get model
# yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# set model class, 0: only detect person
# yolo_model.classes = [0]
# yolo_model.conf = confidence


@st.cache_data
def video_input(input_option):
    video_file = None
    if input_option == 'video':
        video_file = 'data/vlc_WIN_2-2212-2_12_28_17_Pro_cut.mp4'

    if video_file:
        cap = cv2.VideoCapture(video_file)
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # N = 0
        # GPD = 0
        # SD_sum_current = 0
        # SD_avg_current = 0
        i = 0
        N_prev = 0
        GPD_prev = 0.0
        SD_sum_current_prev = 0.0
        SD_avg_current_prev = 0.0
        SD_sum_all_prev = 0.0
        SD_avg_all_prev = 0.0

        placeholder = st.empty()
        while True:
            # while cap.isOpened():
            success, img = cap.read()
            if not success:
                st.write("Can't read frame, exiting ...")
                break

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result, image = infer_image(img)

            # resize result image according to scale_percent ()
            # scale_percent = 40 # %
            # width_ = int(result_width * scale_percent / 100)
            # height_ = int(result_height * scale_percent / 100)
            # dim = (width_, height_)
            # resized = cv2.resize(result.ims[0], dim, interpolation=cv2.INTER_AREA)

            # write result image ()
            output_path = 'output'
            file_name = 'image_detected_' + str(i) + '.jpg'
            cv2.imwrite(os.path.join(output_path, file_name), result.ims[0])
            print('INFO: result image saved to {}\{}'.format(output_path, file_name))
            i += 1

            # display result image ()
            # cv2.imshow('result', result.ims[0])
            # cv2.imshow('Resized image', result.ims[0])
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # get result image size
            result_height = result.ims[0].shape[0]
            result_width = result.ims[0].shape[1]
            print('INFO: result image height: {}\nINFO: result image width: {}'.format(result_height, result_width))

            # result details
            df_result_xywh = result.pandas().xywh[0]
            # merge xywh and xyxy: detection results of 'person', i.e., 'name'=='person' (or 'class'==0)
            df_result_person = result.pandas().xyxy[0]
            df_result_person[['xcenter', 'ycenter', 'width', 'height']] = df_result_xywh[
                ['xcenter', 'ycenter', 'width', 'height']]

            # filter out 'background' person
            df_result_person['filter'] = 'front'
            df_result_person.loc[(df_result_person['width'] * df_result_person['height'] < area_threshold) &
                                 #                      (df_result_person['xmax'] < xmax) &
                                 (df_result_person['ymin'] > ymin), 'filter'] = 'background'

            # filter 'front' & 'person', sort by confidence
            # (if have set model.classes = [0] then don't need the filtering)
            df_result_person = df_result_person[(df_result_person['name'] == 'person') &
                                                (df_result_person['filter'] == 'front')].sort_values('confidence',
                                                                                                     ascending=False)

            # reset index
            df_result_person.index = range(df_result_person.shape[0])
            # add person id column
            df_result_person['p'] = ['p' + str(i) for i in df_result_person.index.to_list()]

            # N
            N = df_result_person.shape[0]
            print('INFO: Number of detected person: {}'.format(N))

            # GPD & SD
            if N != 0:
                # GPD
                # bb_width: Xmax of the most RHS person - Xmin of the most LHS person, i.e., largest Xmax - lowest Xmin
                bb_width = df_result_person['xmax'].max() - df_result_person['xmin'].min()
                # bb_height: for each person, image total height - Ymin of each person
                avg_height = np.mean([result_height - df_result_person.loc[i, 'ymin'] for i in range(N)])
                # get Group Physical Density
                GPD = 1 * 1e6 / (bb_width * avg_height)
                print('INFO: Person bounding boxes width: {}\nINFO: Person average height: {}'.format(bb_width, avg_height))
                print('INFO: Group Physical Density (GPD): {}'.format(GPD))

                # SD
                df_SD = df_result_person[['p', 'xcenter']].assign(key=1).merge(
                    df_result_person[['p', 'xcenter']].assign(key=1), on='key', suffixes=('', '_y'))
                df_SD['distance'] = df_SD.apply(lambda x: x.xcenter - x.xcenter_y, axis=1)
                df_SD = (df_SD.set_index(df_result_person[['p', 'xcenter']].columns.to_list() + ['p_y'])['distance']
                         .unstack()
                         .add_prefix('dist_')
                         .reset_index())

                # merge SD in
                df_result_person = df_result_person.merge(df_SD)

                # calculate SD
                SD_sum = df_SD.iloc[:, -N:].abs().sum().sum() / 2
                SD_avg = SD_sum / N
            else:
                GPD = 0.0
                SD_sum = 0.0
                SD_avg = 0.0
            print('INFO: total SD: {}\nINFO: average SD: {}'.format(SD_sum, SD_avg))

            # construct results
            # add timestamp
            current_time = datetime.now()
            current_time_str = current_time.strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3]

            # save results
            storage.update({current_time_str: {'N': N, 'GPD': GPD, 'SD_sum': SD_sum, 'SD_avg': SD_avg}})
            df_result_final = pd.DataFrame.from_dict(storage).T

            # # add delta
            # nrows = df_result_final.shape[0]
            # df_result_final['N_delta'] = [0] + [df_result_final['N'][i] - df_result_final['N'][i - 1] for i in
            #                                     range(1, nrows)]
            # df_result_final['GPD_delta'] = [0] + [df_result_final['GPD'][i] - df_result_final['GPD'][i - 1] for i in
            #                                       range(1, nrows)]
            # df_result_final['SD_sum_delta'] = [0] + [df_result_final['SD_sum'][i] - df_result_final['SD_sum'][i - 1] for
            #                                          i in range(1, nrows)]
            # df_result_final['SD_avg_delta'] = [0] + [df_result_final['SD_avg'][i] - df_result_final['SD_avg'][i - 1] for
            #                                          i in range(1, nrows)]

            # remove incorrect frames (if N >= 6)
            df_result_final.drop(df_result_final[df_result_final['N'] >= 6].index, inplace=True)
            # expand 2 timestamp columns (split from index)
            df_result_final[['process_date', 'timestamp']] = df_result_final.index.to_frame()[0].str.split('_', expand=True)

            N = df_result_final.iloc[-1, 0]
            N_delta = N - N_prev
            N_prev = N
            GPD = df_result_final.iloc[-1, 1]
            GPD_delta = GPD - GPD_prev
            GPD_prev = GPD
            SD_sum_current = df_result_final.iloc[-1, 2]
            SD_sum_current_delta = SD_sum_current - SD_sum_current_prev
            SD_sum_current_prev = SD_sum_current
            SD_avg_current = df_result_final.iloc[-1, 3]
            SD_avg_current_delta = SD_avg_current - SD_avg_current_prev
            SD_avg_current_prev = SD_avg_current
            SD_sum_all = df_result_final.iloc[:, 2].sum()
            SD_sum_all_delta = SD_sum_all - SD_sum_all_prev
            SD_sum_all_prev = SD_sum_all
            SD_avg_all = df_result_final.iloc[:, 3].mean()
            SD_avg_all_delta = SD_avg_all - SD_avg_all_prev
            SD_avg_all_prev = SD_avg_all

            with placeholder.container():

                # metrics
                st.markdown('## Current KPIs x4: N, GPD, SD, Average SD')
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric(label="Number of people", value=int(N), delta=int(N_delta))
                kpi2.metric(label="Group Physical Density (GPD)", value=round(GPD, 2), delta=round(GPD_delta, 2))
                kpi3.metric(label="Current social distance", value=millify(SD_sum_current), delta=millify(SD_sum_current_delta))
                kpi4.metric(label='Current average distance', value=millify(SD_avg_current), delta=millify(SD_avg_current_delta))

                st.markdown('## Overall KPIs x2: Total SD, Overall Average SD')
                kpi5, kpi6 = st.columns(2)
                kpi5.metric(label='Overall Total Distance', value=millify(SD_sum_all), delta=millify(SD_sum_all_delta))
                kpi6.metric(label="Overall Average Distance", value=millify(SD_avg_all), delta=millify(SD_avg_all_delta))

                # result, plot, video
                st.markdown('## Timestamped group dynamics measurements')
                col7, col8, col9 = st.columns(3)
                with col7:
                    st.markdown("### N, GPD by timestamp")
                    # fig = px.line_chart(data_frame=df_result_final[['N', 'GPD', 'SD_avg', 'SD_sum']])
                    # st.write(fig)
                    st.line_chart(df_result_final[['N', 'GPD']], use_container_width=False)
                with col8:
                    st.markdown("### SD by timestamp")
                    # fig = px.line_chart(data_frame=df_result_final[['N', 'GPD', 'SD_avg', 'SD_sum']])
                    # st.write(fig)
                    st.line_chart(df_result_final[['SD_avg', 'SD_sum']], use_container_width=False)
                with col9:
                    st.markdown('### Student Group 1')
                    # col9.image(resized, use_column_width=True, channels='BGR', caption='Group dynamics')
                    st.image(result.ims[0], use_column_width=True, channels='BGR', caption='Table 1')
                    # st.image(image, use_column_width=True, caption='Group 1')

                # add plots
                col10, col11 = st.columns(2)
                with col10:
                    st.markdown("### KPI distribution")
                    # df_result_final.GPD.plot(kind='hist', bins=40)
                    plt_fig, axs = plt.subplots(nrows=2, ncols=2)
                    df_result_final.plot(kind='hist', y='N', ax=axs[0, 0])
                    df_result_final.plot(kind='hist', y='GPD', ax=axs[0, 1])
                    df_result_final.plot(kind='hist', y='SD_avg', ax=axs[1, 0])
                    df_result_final.plot(kind='hist', y='SD_sum', ax=axs[1, 1])
                    st.pyplot(plt_fig)
                    plt.close(plt_fig)
                with col11:
                    st.markdown('### Group dynamics detail')
                    st.dataframe(df_result_final, use_container_width=False)

                # person detection detail
                st.markdown('## Detection detail')
                st.dataframe(df_result_person, use_container_width=True)
                # time.sleep(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        placeholder.empty()
        print('Completed')


# @st.experimental_singleton
@st.cache_resource
def load_model(path, device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to(device)
    print("model to ", device)
    return model_


@st.cache_data
def infer_image(img, size=None):
    yolo_model.conf = confidence
    result = yolo_model(img, size=size) if size else yolo_model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    return result, image


def main():
    global yolo_model, confidence, cfg_model_path, device_option

    # setup streamlit
    st.title('Real-time Group Dynamics Dashboard')
    st.sidebar.title('Settings')

    # select model
    model_src = st.sidebar.radio('Select model', ['Use default model yolov5s', 'Use your own model'])
    if model_src == 'Use your own model':
        print('Work In Progress ...')
        # st.write('Work In Progress ...')
        # user_model_path = get_user_model()
        # cfg_model_path = user_model_path

    if not os.path.isfile(cfg_model_path):
        st.warning('Model file not available, please add to the model folder', icon='⚠️️')
    else:
        if torch.cuda.is_available():
            device_option = st.sidebar.radio('Select Device', ['cpu', 'cuda'], disabled=False, index=0)
        else:
            device_option = st.sidebar.radio('Select Device', ['cpu', 'cuda'], disabled=True, index=0)

        # confidence slider
        confidence = st.sidebar.slider('Confidence', min_value=0.2, max_value=1.0, value=.55)

        st.sidebar.markdown('---')

        student_group = st.sidebar.radio('Select Student Group', ['Group 1', 'Group 2', 'Group 3', 'Group 4'])
        student_group = 'Group 1'

        # load model
        yolo_model = load_model(cfg_model_path, device_option)

        # set to detect person only
        yolo_model.classes = [0]

        # input source
        # input_option = st.sidebar.radio('Select input type: ', ['video', 'IP camera'])
        input_option = 'video'

        video_input(input_option)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        cv2.destroyAllWindows()
        print('Completed')
        pass
