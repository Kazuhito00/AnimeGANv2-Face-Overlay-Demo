#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2 as cv
import numpy as np
import onnxruntime
import mediapipe as mp

from utils import CvFpsCalc
from utils import CvOverlayImage


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--debug_subwindow_ratio", type=float, default=0.5)

    # カメラ
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    # Face Detection
    parser.add_argument("--fd_model_selection", type=int, default=0)
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)

    # AnimeGANv2
    parser.add_argument("--animegan_model",
                        type=str,
                        default='model/face_paint_512_v2_0.onnx')
    parser.add_argument("--animegan_input_size", type=int, default=512)

    # Selfie Sgmentation
    parser.add_argument("--ss_model_selection", type=int, default=0)
    parser.add_argument("--ss_score_th", type=float, default=0.1)

    args = parser.parse_args()

    return args


def run_single_face_detection(
        face_detection_model,
        image,
        expansion_rate=(0.5, 0.5, 0.5, 0.5),  # 上、右、下、左
        offset_rate=(0, -0.15),  # X, Y
):
    # 推論
    temp_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = face_detection_model.process(temp_image)

    bbox_list = None

    if results.detections is not None:
        for detection in results.detections:
            image_width, image_height = image.shape[1], image.shape[0]

            # バウンディングボックスの中心
            bbox = detection.location_data.relative_bounding_box
            center_x = int((bbox.xmin + (bbox.width / 2)) * image_width)
            center_y = int((bbox.ymin + (bbox.height / 2)) * image_height)

            # バウンディングボックスを正方形として扱う際の一辺の長さ
            side_length = None
            if bbox.width > bbox.height:
                side_length = int(bbox.width * image_width)
            else:
                side_length = int(bbox.height * image_height)

            # バウンディングボックス座標
            bbox_list = [
                int(center_x - (side_length / 2) -
                    (side_length * expansion_rate[3])),
                int(center_y - (side_length / 2) -
                    (side_length * expansion_rate[0])),
                int(center_x + (side_length / 2) +
                    (side_length * expansion_rate[1])),
                int(center_y + (side_length / 2) +
                    (side_length * expansion_rate[2])),
            ]

            # バウンディングボックスをX座標、Y座標に指定オフセット分ずらす
            bbox_width = bbox_list[2] - bbox_list[0]
            bbox_height = bbox_list[3] - bbox_list[1]

            bbox_list[0] = bbox_list[0] + int(bbox_width * offset_rate[0])
            bbox_list[1] = bbox_list[1] + int(bbox_height * offset_rate[1])
            bbox_list[2] = bbox_list[2] + int(bbox_width * offset_rate[0])
            bbox_list[3] = bbox_list[3] + int(bbox_height * offset_rate[1])

            # 上限加減クリッピング
            bbox_list[0] = 1 if bbox_list[0] <= 0 else bbox_list[0]
            bbox_list[1] = 1 if bbox_list[1] <= 0 else bbox_list[1]
            bbox_list[2] = image_width - 1 if bbox_list[
                2] >= image_width else bbox_list[2]
            bbox_list[3] = image_height - 1 if bbox_list[
                3] >= image_height else bbox_list[3]

            # 顔検出は一つまで
            break

    return bbox_list


def run_animegan_v2(onnx_session, input_size, image):
    # リサイズ
    temp_image = copy.deepcopy(image)
    resize_image = cv.resize(temp_image, dsize=(input_size, input_size))
    x = cv.cvtColor(resize_image, cv.COLOR_BGR2RGB)

    # 前処理
    x = np.array(x, dtype=np.float32)
    x = x.transpose(2, 0, 1).astype('float32')
    x = x * 2 - 1
    x = x.reshape(-1, 3, input_size, input_size)

    # 推論
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # 後処理
    onnx_result = np.array(onnx_result).squeeze()
    onnx_result = (onnx_result * 0.5 + 0.5).clip(0, 1)
    onnx_result = onnx_result * 255

    onnx_result = onnx_result.transpose(1, 2, 0).astype('uint8')
    onnx_result = cv.cvtColor(onnx_result, cv.COLOR_RGB2BGR)

    return resize_image, onnx_result


def run_selfie_segmentation(
    selfie_segmentation_model,
    image,
    score_th,
):
    # 推論
    temp_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = selfie_segmentation_model.process(temp_image)

    mask = np.stack((results.segmentation_mask, ) * 3, axis=-1) >= score_th

    return mask


def face_segmentation_and_anime_gan(
    image,
    face_detection,
    anime_gan_v2,
    animegan_input_size,
    selfie_segmentation,
    ss_score_th,
):
    face_image = None
    anime_face_image = None
    seg_mask = None
    masked_anime_face = None

    # 顔検出
    bbox = run_single_face_detection(face_detection, image)

    if bbox is not None:
        # 顔画像切り抜き、リサイズ
        face_image = copy.deepcopy(image[bbox[1]:bbox[3], bbox[0]:bbox[2]])
        face_image = cv.resize(
            face_image,
            dsize=(animegan_input_size, animegan_input_size),
        )

        # AnimeGANv2
        _, anime_face_image = run_animegan_v2(
            anime_gan_v2,
            animegan_input_size,
            face_image,
        )

        # 人物セグメンテーション
        seg_mask = run_selfie_segmentation(
            selfie_segmentation,
            face_image,
            ss_score_th,
        )

        # アニメ画像の人物のみ切り抜いて合成
        masked_anime_face = np.where(
            seg_mask,
            anime_face_image,
            face_image,
        )

        masked_anime_face = cv.resize(
            masked_anime_face,
            dsize=(bbox[2] - bbox[0], bbox[3] - bbox[1]),
        )

    result = Result(
        bbox,
        face_image,
        anime_face_image,
        seg_mask,
        masked_anime_face,
    )

    return result


class Result(object):
    def __init__(
        self,
        bbox,
        face_image,
        anime_face_image,
        seg_mask,
        masked_anime_face,
    ):
        self._bbox = bbox
        self._face_image = face_image
        self._anime_face_image = anime_face_image
        self._seg_mask = seg_mask
        self._masked_anime_face = masked_anime_face

    @property
    def bbox(self):
        return self._bbox

    @property
    def face_image(self):
        return self._face_image

    @property
    def anime_face_image(self):
        return self._anime_face_image

    @property
    def seg_mask(self):
        return self._seg_mask

    @property
    def masked_anime_face(self):
        return self._masked_anime_face


def main():
    # 引数 #####################################################################
    args = get_args()

    debug = args.debug
    debug_subwindow_ratio = args.debug_subwindow_ratio

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    fd_model_selection = args.fd_model_selection
    min_detection_confidence = args.min_detection_confidence

    animegan_model = args.animegan_model
    animegan_input_size = args.animegan_input_size

    ss_model_selection = args.ss_model_selection
    ss_score_th = args.ss_score_th

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    # Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=fd_model_selection,
        min_detection_confidence=min_detection_confidence,
    )

    # AnimeGANv2
    anime_gan_v2 = onnxruntime.InferenceSession(animegan_model)

    # Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
        model_selection=ss_model_selection)

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=3)

    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 推論実施 #############################################################
        result = face_segmentation_and_anime_gan(
            image,
            face_detection,
            anime_gan_v2,
            animegan_input_size,
            selfie_segmentation,
            ss_score_th,
        )

        # 描画
        debug_image, mask_image, green_image = draw_detection(
            debug_image,
            result,
            display_fps,
            debug,
        )

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('AnimeGANv2 Face Overlay Demo', debug_image)
        if debug and result.bbox is not None:
            face_image = cv.resize(
                result.face_image,
                dsize=None,
                fx=debug_subwindow_ratio,
                fy=debug_subwindow_ratio,
            )
            cv.imshow('Debug01 : Face', face_image)
            anime_face_image = cv.resize(
                result.anime_face_image,
                dsize=None,
                fx=debug_subwindow_ratio,
                fy=debug_subwindow_ratio,
            )
            cv.imshow('Debug02 : AnimeGANv2 Face', anime_face_image)
            mask_image = cv.resize(
                mask_image,
                dsize=None,
                fx=debug_subwindow_ratio,
                fy=debug_subwindow_ratio,
            )
            cv.imshow('Debug03 : Mask', mask_image)
            green_image = cv.resize(
                green_image,
                dsize=None,
                fx=debug_subwindow_ratio,
                fy=debug_subwindow_ratio,
            )
            cv.imshow('Debug04 : Masked AnimeGANv2 Face', green_image)

    cap.release()
    cv.destroyAllWindows()


def draw_detection(image, result, display_fps, debug_mode):
    mask_image = None
    green_image = None

    if result.bbox is not None:
        # AnimeGAN画像の上書き
        image = CvOverlayImage.overlay(image, result.masked_anime_face,
                                       (result.bbox[0], result.bbox[1]))

        if debug_mode:
            # デバッグ用：バウンディングボックス
            cv.rectangle(image, (result.bbox[0], result.bbox[1]),
                         (result.bbox[2], result.bbox[3]), (255, 255, 255), 1)

            # デバッグ用：セグメンテーションマスク
            mask_image = np.zeros(result.anime_face_image.shape,
                                  dtype=np.uint8)
            mask_image[:] = (255, 255, 255)
            mask_image2 = np.zeros(result.anime_face_image.shape,
                                   dtype=np.uint8)
            mask_image2[:] = (0, 0, 0)
            mask_image = np.where(
                result.seg_mask,
                mask_image,
                mask_image2,
            )

            # デバッグ用：セグメンテーション緑背景
            green_image = np.zeros(result.anime_face_image.shape,
                                   dtype=np.uint8)
            green_image[:] = (0, 255, 0)
            green_image = np.where(
                result.seg_mask,
                result.anime_face_image,
                green_image,
            )

    cv.putText(image, "FPS:" + str(display_fps), (30, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv.LINE_AA)

    return image, mask_image, green_image


if __name__ == '__main__':
    main()
