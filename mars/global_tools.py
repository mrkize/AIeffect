import logging
import json
import re
import os
from urllib.request import urlopen
import cv2
import numpy as np
from mars.afts_sdk import Afts
from mars.github_oss import OssManager
afts = Afts("mars", "key", "apwallet")
oss_manager = OssManager()

def open_url_json(url):
    """
        Open json from url.

    :param url: json's url
    :return: json object
    """
    json_string = urlopen(url).read().decode("utf-8")
    return json.loads(json_string)


def open_local_json(filepath):
    """
        Open json from local file.
    :param filepath: json's local filepath
    :return: json object
    """
    with open(filepath, "r") as f:
        return json.load(f)


def open_url_image(url):
    """
        Open image from url.

    :param url: url
    :return:
        image array
    """
    try:
        img_data = np.asarray(bytearray(urlopen(url).read()), dtype="uint8")
        img = cv2.imdecode(img_data, -1).copy()
        if img.dtype == 'uint16':
            img = cv2.convertScaleAbs(img, alpha=(255.0 / 65535.0))
    except Exception as e:
        print("Warning load image failed", e, url)
        return None
    return img


def upload_file(file_path, file_id=None):
    file_name = file_path.split("/")[-1]
    if not os.path.exists(file_path):
        logging.error("{} not exists".format(file_path))
        return None
    else:
        # with open(file_path, "rb") as f:
        # file_data = f.read()
        _, ext = os.path.splitext(file_path)
        file_url = oss_manager.upload_file(f"mars_pic/{file_id}_{file_name}{ext}", file_path)
        return file_url
        # if file_id is not None:
        #     return "https://mdn.alipayobjects.com/mars/afts/" + file_type + "/{}".format(file_id)
        # else:
        #     return None


def upload_image(image, path):
    """
        Upload image to AFTS.

    :param image: image array
    :param path: temporal local path
    :return:
        AFTS image url
    """
    cv2.imwrite(path, image)

    return upload_file(path, "img")


def save_image(image, path):
    """
        Save image to local pth.

    :param image: image array
    :param path: temporal local path

    """
    cv2.imwrite(path, image)


def get_name_from_url(url):
    """
        Get file name from url
    :param url: url
    :return:
        file name
    """
    rstr = r"[\/\\\:\*\?\"\<\>\|%]"  # '/ \ : * ? " < > |'
    name = re.sub(rstr, "_", url.split("/")[-1])
    return name
