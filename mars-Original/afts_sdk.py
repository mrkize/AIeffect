# -*- coding: utf-8 -*-

"""
Copyright (C) 2020 antfin.com. All rights reserved.

@file: afts.py
@author: 轩勇
@date: 20200603
"""

import re
import time
import requests
import hashlib


###### uncomment the following lines to enable debug information to display on console ######
# import httplib
# import logging
# httplib.HTTPConnection.debuglevel = 1
# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)
# requests_log = logging.getLogger("requests.packages.urllib3")
# requests_log.setLevel(logging.DEBUG)
# requests_log.propagate = True
#############################################################################################

class Afts:
    """
    This class is used to upload and/or download file data to/from afts.
    The public api is "download_file" and "upload_file", these two function will not throw exception.
    """

    def __init__(self, biz_key, biz_secret, appid, endpoint_config={}):

        self.__biz_key = biz_key
        # __biz_secret 区分环境，请填写对应的环境秘钥
        self.__biz_secret = biz_secret
        self.__appid = appid
        self.__err_msg = ""

        # 上传/下载/令牌权限域名同样也区区分环境，哪个环境资源就填写哪个环境的配置，和__biz_secret一样
        # 不同环境域名配置参考 https://yuque.antfin-inc.com/afts/document/tfklky
        # 上传域名
        self.__upload_endpoint_source = endpoint_config.get("upload_endpoint_source", "mass.alipay.com")
        # 下载域名
        self.__download_endpoint_source = endpoint_config.get("download_endpoint_source", "mass.alipay.com")
        # 令牌权限域名
        self.__authority_endpoint = endpoint_config.get("authority_endpoint", "mmtcapi.alipay.com")

        if self.__authority_endpoint.find('alipay.com') != -1:
            self.__http_schema = 'https'
        else:
            # 线下是http
            self.__http_schema = 'http'

    def err_msg(self):
        """
        This function return error message.
        """
        return self.__err_msg

    """
    获取op token
    """

    def get_op_token(self):

        time_stamp = str(int(time.time() * 1000))
        authority_url = self.__http_schema + "://" + self.__authority_endpoint + "/token/1.0/op"
        url_params = {}
        url_params["timestamp"] = time_stamp
        url_params["bizKey"] = self.__biz_key
        url_params["appId"] = self.__appid

        md5_handle = hashlib.md5()
        md5_handle.update((self.__appid + self.__biz_key + time_stamp + self.__biz_secret).encode('utf-8'))
        sign = md5_handle.hexdigest()
        url_params["sign"] = sign

        response = requests.get(authority_url, params=url_params)
        if response.status_code != 200:
            self.__err_msg = "Error:get_op_token:http status code != 200,message:" + response.text
            return None
        else:
            res_json = response.json()
            if res_json["code"] != 0:
                self.__err_msg = "Error:get_op_token:server response code != 0,code:{}".format(res_json["code"])
                return None
            else:
                return res_json["data"]["token"]

    """
    获取 acktoken
    """

    def get_acl_token(self, file_id):
        authority_url = self.__http_schema + "://" + self.__authority_endpoint + "/token/1.0/acl"
        time_stamp = str(int(time.time() * 1000))
        url_params = {}
        url_params["timestamp"] = time_stamp
        url_params["bizKey"] = self.__biz_key
        url_params["fileId"] = file_id
        url_params["appId"] = self.__appid

        md5_handle = hashlib.md5()
        md5_handle.update((self.__appid + self.__biz_key + time_stamp + file_id + self.__biz_secret).encode('utf-8'))
        sign = md5_handle.hexdigest()
        url_params["sign"] = sign

        response = requests.get(authority_url, params=url_params)
        if response.status_code != 200:
            self.__err_msg = "Error:get_acl_token:http status code != 200,message:" + response.text
            return None
        else:
            res_json = response.json()
            if res_json["code"] != 0:
                self.__err_msg = "Error:get_acl_token:server response code != 0,code:" + str(res_json["code"])
                return None
            else:
                return res_json["data"]

    """
    获取 mass token
    """

    def get_mass_token(self):
        authority_url = self.__http_schema + "://" + self.__authority_endpoint + "/token/1.0/mass"
        time_stamp = str(int(time.time() * 1000))
        url_params = {}
        url_params["appId"] = self.__appid
        url_params["bizKey"] = self.__biz_key
        url_params["opToken"] = self.get_op_token()
        url_params["massType"] = '1'
        url_params["timestamp"] = time_stamp
        url_params["value"] = self.__biz_key

        md5_handle = hashlib.md5()
        md5_handle.update((self.__appid + self.__biz_key + url_params["value"] + time_stamp).encode('utf-8'))
        sign = md5_handle.hexdigest()
        url_params["sign"] = sign

        response = requests.get(authority_url, params=url_params)
        if response.status_code != 200:
            self.__err_msg = "Error:get_mass_token:http status code != 200,message:" + response.text
            return None
        else:
            res_json = response.json()
            if res_json["code"] != 0:
                self.__err_msg = "Error:get_mass_token:server response code != 0,code:" + res_json["code"]
                return None
            else:
                return res_json["data"]

    """
    下载文件
    """

    def __download_file(self, file_id):
        acl_token = self.get_acl_token(file_id)
        download_url = "https://" + self.__download_endpoint_source + "/afts/file/" + file_id
        url_params = {}
        url_params["bizType"] = self.__biz_key
        url_params["token"] = acl_token

        response = requests.get(download_url, params=url_params, allow_redirects=False)
        if response.status_code != 200:
            print("download file failed." + "fileid=" + file_id + " url=" + download_url, "params=", url_params)
            self.__err_msg = "Error:__download_file:http status code != 200,message:" + response.text
            return None
        else:
            # succeeded!
            print("download file success." + "fileid=" + file_id + " url=" + download_url, "params=", url_params)
            return response.content

    def __upload_file(self, file_data, file_name, setpublic):
        mass_token = self.get_mass_token()
        upload_url = "https://" + self.__upload_endpoint_source + "/file/auth/upload"
        url_params = {}
        url_params["bz"] = self.__biz_key
        url_params["public"] = str(setpublic).lower()
        url_params["mt"] = mass_token

        # compute file data md5
        # md5_handle = hashlib.md5()
        # md5_handle.update((file_data).encode('utf-8'))
        # file_data_md5 = md5_handle.hexdigest()
        # form_param = {"md5":file_data_md5}

        # file data to be uploaded, we use the md5 as the file name
        # form_file = {"file":(file_data_md5, file_data, "application/octet-stream")}
        form_file = {"file": (file_name, file_data, "application/octet-stream")}

        # response = requests.post(upload_url, params=url_params, data=form_param, files=form_file)
        response = requests.post(upload_url, params=url_params, files=form_file)
        if response.status_code != 200:
            self.__err_msg = "Error:__upload_file:http status code != 200"
            return None
        else:
            res_json = response.json()
            if res_json["code"] != 0:
                self.__err_msg = "Error:__upload_file:server response code != 0"
                return None
            else:
                return res_json["data"]["id"]

    def download_file(self, file_id):
        """
        Wrapper function that will not throw exception
        """
        try:
            file_data = self.__download_file(file_id)
            return file_data
        except Exception as e:
            self.__err_msg = repr(e)
            return None

    """
    上传文件
    param: file_data: 文件流
    file_name: 设置文件名，与下载的content-type 关联
    setpublic: 设置文件公私有
    """

    def upload_file(self, file_data, file_name, setpublic=False):
        """
        Wrapper function that will not throw exception
        """
        try:
            file_id = self.__upload_file(file_data, file_name, setpublic)
            return file_id
        except Exception as e:
            self.__err_msg = repr(e)
            return None

    """
    获取url
    """

    def get_url(self, file_id):
        acl_token = self.get_acl_token(file_id)
        download_url = "https://" + self.__download_endpoint_source + "/afts/file/" + file_id + "?" + "bizType=" + self.__biz_key + "&token=" + acl_token
        return download_url


def afts_test():
    """
    This function shows example usage and can be used for test purpose
    """

    # dev end points for download, upload and authentication
    endpoint_config = {"download_endpoint_source": "mass.alipay.com",
                       "upload_endpoint_source": "mass.alipay.com",
                       "authority_endpoint": "mmtcapi.alipay.com"}

    afts = Afts("your_biz_key", "your_biz_secret", "your_appid", endpoint_config=endpoint_config)
    # with open("./_antvip_client.so", "rb") as f:
    #    file_data = f.read()
    file_data = "this is my test file data"

    # perform the upload
    for i in range(1):
        file_id = afts.upload_file(file_data)
        if file_id is None:
            print(afts.err_msg())
        else:
            print("upload ok, file_id: %s" % file_id)
            # perform the download
            file_data = afts.download_file(file_id)
            if file_data is None:
                print("download error: %s" % afts.err_msg())
            else:
                md5_handle = hashlib.md5()
                md5_handle.update(file_data)
                file_data_md5 = md5_handle.hexdigest()
                print("download ok, file md5: %s" % file_data_md5)
