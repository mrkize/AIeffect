# -*- coding: utf-8 -*-
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
from itertools import islice
import os
import logging
import time
import random
from datetime import datetime
import requests
import base64
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def upload_image_to_github(token,
                           repo_owner,
                           repo_name,
                           image_path,
                           branch='main'):
    with open(image_path, 'rb') as file:
        image_data = file.read()

    encoded_image = base64.b64encode(image_data).decode('utf-8')

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"image_{timestamp}.{image_path.split('.')[-1]}"
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{filename}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "message": f"Upload image {filename}",
        "content": encoded_image,
        "branch": branch
    }
    response = requests.put(url, json=data, headers=headers)
    if response.status_code == 201:
        raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{filename}"
        return raw_url
    else:
        raise Exception(f"Failed to upload image: {response.text}")


class OssManager:

    def __init__(self):
        self.mytoken = ""
        self.owner = ""
        self.repo = ""
        self.image_path = "test_pic/lcar.png"

        self.endpoint = "https://oss-cn-hangzhou.aliyuncs.com"
        self.region = "cn-hangzhou"
        # 默认Bucket名称
        self.bucket_name = "ant-aie"
        # self.bucket = oss2.Bucket(self.auth,
        #                           self.endpoint,
        #                           self.bucket_name,
        #                           region=self.region)

    def upload_file(self, object_name, data):
        try:
            raw_url = upload_image_to_github(self.mytoken, self.owner, self.repo, data)
            return raw_url
        except oss2.exceptions.OssError as e:
            logging.error(f"Failed to upload file: {e}")
            raise e

    def download_file(self, object_name):
        try:
            file_obj = self.bucket.get_object(object_name)
            content = file_obj.read().decode('utf-8')
            logging.info("File content:")
            logging.info(content)
            return content
        except oss2.exceptions.OssError as e:
            logging.error(f"Failed to download file: {e}")
            raise e

    def list_objects(self):
        try:
            objects = list(islice(oss2.ObjectIterator(self.bucket), 10))
            for obj in objects:
                logging.info(obj.key)
            return [obj.key for obj in objects]
        except oss2.exceptions.OssError as e:
            logging.error(f"Failed to list objects: {e}")
            raise e

    def delete_objects(self):
        try:
            objects = list(islice(oss2.ObjectIterator(self.bucket), 100))
            if objects:
                for obj in objects:
                    self.bucket.delete_object(obj.key)
                    logging.info(f"Deleted object: {obj.key}")
            else:
                logging.info("No objects to delete")
        except oss2.exceptions.OssError as e:
            logging.error(f"Failed to delete objects: {e}")
            raise e

    def delete_bucket(self):
        try:
            self.bucket.delete_bucket()
            logging.info("Bucket deleted successfully")
        except oss2.exceptions.OssError as e:
            logging.error(f"Failed to delete bucket: {e}")
            raise e

# 示例用法
if __name__ == "__main__":
    oss_manager = OssManager()
    # 示例：上传文件
    upload_result = oss_manager.upload_file("test_pic/10jifen.png", "test_pic/10jifen.png")
    print(f"Upload result: {upload_result}")
