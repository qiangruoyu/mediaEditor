from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2
import numpy as np
import os


# #模型下载
# from modelscope import snapshot_download
# model_dir = snapshot_download('model')

def create_video_mask(video_path, output_mask_dir):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 获取视频帧率
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 读取第一帧以获取帧的宽高信息
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    height, width, channels = frame.shape

    # 创建一个全黑的mask图像
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    # 生成mask文件名
    mask_filename = f"mask_00000_{frame_count - 1:05d}.png"
    mask_filepath = os.path.join(output_mask_dir, mask_filename)

    # # 创建输出目录（如果不存在）
    # os.makedirs(output_mask_dir, exist_ok=True)

    # 保存mask图像
    cv2.imwrite(mask_filepath, mask)

    print(f"Mask file saved as {mask_filepath}")

    # 释放视频捕获对象
    cap.release()


# 示例调用
model_name = "model/cv_video-inpainting"
video_output_path = "/home/qiangyu/work_space/mediaEditor/workspace/cv_video-inpainting/mask"
mask_path = '/home/qiangyu/work_space/mediaEditor/workspace/cv_video-inpainting/mask'
video_input_path = '/home/qiangyu/work_space/mediaEditor/upload_video/5月26日修改版_1.mp4'
word_dic = {'video_input_path':video_input_path,
                           'mask_path':mask_path,
                           'video_output_path':video_output_path}
create_video_mask(video_input_path, mask_path)
video_inpainting = pipeline(Tasks.video_inpainting, 
                       model=model_name)
result_status = video_inpainting(word_dic)
result = result_status[OutputKeys.OUTPUT]





