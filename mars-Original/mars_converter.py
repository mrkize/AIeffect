import json
import time
import os
import cv2
# import global_tools
import numpy as np
from mars import mars_generator_layered, mars_curve_fitter, mars_generator_base, global_tools
fps = 30.3  # 1000/33


def convert_to_lastest_mars_json(mars_json_data):
    """
        convert old mars json to lastest mars json
    :param mars_json_data: mars_json_data
    :return:
        new mars_json_data
    """
    if mars_json_data["version"].startswith("0."):
        mars_json_data = mars_generator_base.convert_to_json_v1(mars_json_data)
    return mars_json_data


def get_image_split_from_texture(tex_img, split):
    """
        Get image split from mars texture image
    :param tex_img: tex img
    :param split: split data [x,y,w,h,rot](scaled)
    :return:
        image split
    """
    x = int(split[0] * tex_img.shape[1])
    y = int(split[1] * tex_img.shape[0])
    w = int(split[2] * tex_img.shape[1])
    h = int(split[3] * tex_img.shape[0])

    if split[4] == 0:
        y = tex_img.shape[1] - y - h
        img_split = tex_img[y:(y + h), x:(x + w), :]
    else:
        y = tex_img.shape[1] - y - w
        img_split = tex_img[y:(y + w), x:(x + h), :]
        img_split = cv2.rotate(img_split, cv2.ROTATE_90_CLOCKWISE)
    return img_split


def get_image_from_texture(tex_img, splits):
    """
        Get whole image from mars texture image
    :param tex_img: tex img
    :param splits: split data [x,y,w,h,rot](scaled) array
    :return:
        image
    """
    if len(splits) == 1:
        img = get_image_split_from_texture(tex_img, splits[0])
    elif len(splits) == 4:
        assert splits[0][2] == splits[1][2] == splits[2][2] == splits[3][2]
        assert splits[0][3] == splits[1][3] == splits[2][3] == splits[3][3]
        w, h = int(splits[0][2] * tex_img.shape[1]), int(splits[0][3] * tex_img.shape[0])
        img = np.zeros((h*2, w*2, tex_img.shape[2]))
        for i in range(4):
            img_split = get_image_split_from_texture(tex_img, splits[i])
            img[(1 - i // 2) * h: (2 - i // 2) * h, (i % 2) * w: (i % 2 + 1) * w] = img_split
    return img


def get_anchor_from_particle_origin(particle_origin):
    anchor_list = [
        [0.5, 0.5],
        [0, 0],
        [0, 0.5],
        [0, 1],
        [0.5, 0],
        [0.5, 1],
        [1, 0],
        [1, 0.5],
        [1, 1]
    ]
    return anchor_list[particle_origin]


def parse_dict_value(dict_value):
    if "value" in dict_value:
        value = dict_value["value"]
    elif "min" in dict_value and "max" in dict_value:
        value = (dict_value["min"] + dict_value["max"]) / 2
    else:
        raise Exception("Unknown dict value")
    return value


def get_keyframe_data_from_jsons(mars_json_data, name, mars_keyframe_json_data):
    """
        Get keyframe data from two json objects(mars_json_data, mars_keyframe_json_data)

    :param mars_json_data: mars_json_data
    :param mars_keyframe_json_data: mars_json_data
    :return:
        Array of layer_info({"img": img_url, "data": keyframe_data})
    """
    layers = []
    # mars json数据
    mars_items = mars_json_data["compositions"][0]["items"]
    # mars 关键帧数据
    mars_keyframe_items = mars_keyframe_json_data["compositions"][0]["items"]
    # 关键帧是否是全局transform
    is_global = mars_keyframe_json_data["globalTransform"] if "globalTransform" in mars_keyframe_json_data else False
    # mars 合成时间和帧数
    duration = mars_json_data["compositions"][0]["duration"]
    total_frames = int(duration * fps)
    # mars 合成尺寸
    if "previewSize" in mars_json_data["compositions"][0]:
        g_width = mars_json_data["compositions"][0]["previewSize"][0]
        g_height = mars_json_data["compositions"][0]["previewSize"][1]
    else:
        g_width = g_height = 400

    # 获取关键帧数据中所有的id
    mars_keyframe_id_dict = dict()
    for i, item in enumerate(mars_keyframe_items):
        mars_keyframe_id_dict[item["id"]] = i

    # 获取所有的图片
    if "images" in mars_json_data:
        images = [global_tools.open_url_image(image["url"]) for image in mars_json_data["images"]]
        images = [image for image in images if image is not None]

    # mars图层类型列表
    item_type_list = ["base", "sprite", "particle", "null", "interact", "plugin", "model", "composition", "filter"]
    for i, item in enumerate(mars_items):
        # 过滤掉不在数据列表中的图层
        if item["id"] not in mars_keyframe_id_dict:
            continue

        # 目前只保留图层、空节点元素
        item_type = item_type_list[int(item["type"])]
        if item_type not in ["sprite", "null"]:
            continue

        # 获取关键帧数据
        keyframe_item = mars_keyframe_items[mars_keyframe_id_dict[item["id"]]]
        start_time = keyframe_item["delay"]
        if type(keyframe_item["startSize"]) is dict:
            keyframe_item["startSize"] = parse_dict_value(keyframe_item["startSize"])
        if type(keyframe_item["startAspect"]) is dict:
            keyframe_item["startAspect"] = parse_dict_value(keyframe_item["startAspect"])
        start_width = keyframe_item["startSize"]
        start_height = keyframe_item["startSize"] / keyframe_item["startAspect"]
        start_alpha = keyframe_item["startAlpha"]

        # 获取贴图并保存贴图到本地
        local_layer_image_path = None
        anchor = [0.5, 0.5]
        if item_type in ["sprite"]:
            # 过滤掉没有贴图信息掉图层
            if "renderer" not in item["content"] or "texture" not in item["content"]["renderer"]:
                continue

            # 获取来源贴图
            texture_id = item["content"]["renderer"]["texture"]
            if "textures" in mars_json_data:
                source_image = images[mars_json_data["textures"][texture_id]["source"]]
            else:
                source_image = images[texture_id]

            # 如果贴图是合并过的，需要对合并过对贴图中抽取原始图片
            if "splits" in item["content"]:
                layer_image = get_image_from_texture(source_image, item["content"]["splits"])
            else:
                layer_image = source_image
            # 如果有锚点的话
            if "anchor" in item["content"]["renderer"]:
                anchor = item["content"]["renderer"]["anchor"]
            # 旧版锚点particleOrigin切换anchor
            if "particleOrigin" in item["content"]["renderer"]:
                anchor = get_anchor_from_particle_origin(item["content"]["renderer"]["particleOrigin"])

            # 保存贴图到本地
            local_layer_image_path = "dataset_v1/images/" + name + "_" + str(i) + ".png"
            global_tools.save_image(layer_image, local_layer_image_path)
            # local_layer_image_path = layer_image

        # 修复因为锚点导致到位置偏移
        for j in range(len(keyframe_item["position"])):
            keyframe_item["position"][j][0] -= (anchor[0] - 0.5) * start_width
            keyframe_item["position"][j][1] += (anchor[1] - 0.5) * start_height

        # 修复clipMode导致的大小位置差异
        if "clipMode" in mars_json_data["compositions"][0]["camera"] and \
                mars_json_data["compositions"][0]["camera"]["clipMode"] == 0:
            for j in range(len(keyframe_item["position"])):
                keyframe_item["position"][j][0] *= g_height / g_width
                keyframe_item["position"][j][1] *= g_height / g_width
            if is_global or "parentId" not in item:
                start_width *= g_height / g_width
                start_height *= g_height / g_width

        # 转换关键帧数据（归一化数据，填补到合成时间）
        start_keyframe = max(0, int(start_time * fps))
        dur_keyframe = len(keyframe_item["position"])
        # dur_keyframe = min(len(keyframe_item["position"], len(keyframe_item["opacity"], len(keyframe_item["size"], len(keyframe_item["rotation"], len(keyframe_item["opacity"])

        keyframe_data = {}
        for j in range(len(keyframe_item["size"])):
            keyframe_item["size"][j][0] *= start_width
            keyframe_item["size"][j][1] *= start_height
        for j in range(len(keyframe_item["opacity"])):
            keyframe_item["opacity"][j] *= start_alpha

        for k in ["position", "opacity", "size", "rotation"]:
            keyframe_item[k] = [keyframe_item[k][0]] * start_keyframe + keyframe_item[k] + [keyframe_item[k][-1]] * (total_frames - start_keyframe - dur_keyframe)
            while len(keyframe_item[k]) > total_frames:
                keyframe_item[k].pop()
            while len(keyframe_item[k]) < total_frames:
                keyframe_item[k].append(keyframe_item[k][-1])
            keyframe_data[k] = np.array(keyframe_item[k]).tolist()
        keyframe_data["visible"] = np.array([0] * start_keyframe + [1] * dur_keyframe + [0] * (total_frames - start_keyframe - dur_keyframe)).tolist()

        assert len(keyframe_item["position"]) == total_frames

        # 图层信息汇总
        item_info = {"type": item_type, "data": keyframe_data, "name": item["name"], "id": item["id"], "anchor": anchor}
        if local_layer_image_path is not None:
            item_info["img"] = local_layer_image_path
        if not is_global and "parentId" in item:
            item_info["parent"] = item["parentId"]
        layers.append(item_info)

    res = {
        "width": g_width,
        "height": g_height,
        "time": duration,
        "name": name,
        "layers": layers
    }
    return res


def generate_json_from_keyframe_data_v0(keyframe_data):
    """
        Generate json from keyframe data.

    :param keyframe_data: keyframe_data
    :return:
        mars json string
    """
    g_width = keyframe_data["width"]
    g_height = keyframe_data["height"]
    name = keyframe_data["name"]
    layer_keyframe_data = keyframe_data["layers"]

    total_frames = layer_keyframe_data[0]["data"]["position"].shape[0]
    parsed_layers = []

    for i, layer_data in enumerate(layer_keyframe_data):
        layer_frames = layer_data["total_frames"] if "total_frames" in layer_data else total_frames
        layer_info = mars_generator_layered.layer_info_template.copy()
        layer_info["id"] = layer_data["id"]
        layer_info["nm"] = layer_data["name"]
        layer_info["type"] = "sprite"
        layer_info["st"] = 0
        layer_info["ed"] = layer_frames / fps
        layer_info["w"] = 1
        layer_info["h"] = 1
        layer_info["p"] = [0, 0]
        layer_info["a"] = [layer_info["w"] / 2, layer_info["h"] / 2]
        layer_info["img"] = global_tools.upload_file(layer_data["img"])
        layer_info["img_type"] = "path"
        layer_info["oa"] = layer_info["sa"] = layer_info["pa"] = layer_info["rxa"] = layer_info["rya"] = layer_info["rza"] = True
        layer_info["okf"] = layer_info["skf"] = layer_info["pkf"] = layer_info["rxkf"] = layer_info["rykf"] = layer_info["rzkf"] = True

        # deal with alpha
        opacity_kf = layer_data["data"]["opacity"].tolist()
        visiable_kf = layer_data["data"]["visible"].tolist()
        layer_info["o"] = [[]]
        for j in range(layer_frames):
            layer_info["o"][0].append([j / (layer_frames - 1), opacity_kf[j] * visiable_kf[j]])

        # deal with scale
        scale_kf = layer_data["data"]["size"].tolist()
        layer_info["s"] = [[], [], []]
        for j in range(layer_frames):
            for k in range(3):
                layer_info["s"][k].append([j / (layer_frames - 1), scale_kf[j][k]])

        # deal with rotation
        rotation_kf = layer_data["data"]["rotation"].tolist()
        layer_info["rx"] = [[]]
        layer_info["ry"] = [[]]
        layer_info["rz"] = [[]]
        for j in range(layer_frames):
            layer_info["rx"][0].append([j / (layer_frames - 1), rotation_kf[j][0]])
            layer_info["ry"][0].append([j / (layer_frames - 1), rotation_kf[j][1]])
            layer_info["rz"][0].append([j / (layer_frames - 1), rotation_kf[j][2]])

        # deal with path
        position_kf = layer_data["data"]["position"].tolist()
        layer_info["p"] = [[]]
        layer_info["pp"] = []
        layer_info["cp"] = []
        for j in range(layer_frames):
            layer_info["p"][0].append([j / (layer_frames - 1), j / (layer_frames - 1), 0, 0])
            layer_info["pp"].append([[position_kf[j][0], position_kf[j][1], position_kf[j][2]], [position_kf[j][0], position_kf[j][1], position_kf[j][2]]])
            layer_info["cp"].append([[position_kf[j][0], position_kf[j][1], position_kf[j][2]], [position_kf[j][0], position_kf[j][1], position_kf[j][2]]])

        parsed_layers.append(layer_info)
    parsed_layers.reverse()
    mars_json = mars_generator_layered.generate_mars_json_by_layers(
        parsed_layers, g_width, g_height, name, total_frames / fps, "local")
    return mars_json


def generate_json_from_keyframe_data_v1(keyframe_data):
    """
        Generate json from keyframe data.

    :param keyframe_data: keyframe_data

    :return:
        mars json string
    """
    g_width = keyframe_data["width"]
    g_height = keyframe_data["height"]
    name = keyframe_data["name"]
    layer_keyframe_data = keyframe_data["layers"]
    total_frames = len(layer_keyframe_data[0]["data"]["position"])
    parsed_layers = []

    def check_keypoint_all_same(keyframes):
        return np.allclose(keyframes, keyframes[0], 1e-3, 1e-3)

    for i, layer_data in enumerate(layer_keyframe_data):
        layer_frames = layer_data["total_frames"] if "total_frames" in layer_data else total_frames
        layer_size = layer_data["origin_size"] if "origin_size" in layer_data else None
        layer_info = mars_generator_layered.layer_info_template.copy()
        layer_info["id"] = layer_data["id"] if "id" in layer_data else str(i)
        layer_info["nm"] = layer_data["name"] if "name" in layer_data else "layer_" + str(i)
        layer_info["type"] = layer_data["type"]
        layer_info["p"] = [0, 0]
        layer_info["anchor"] = layer_data["anchor"]
        if "img" in layer_data:
            layer_info["img"] = global_tools.upload_file(layer_data["img"])
            layer_info["img_type"] = "path"
        if "parent" in layer_data:
            layer_info["parent"] = layer_data["parent"]
        layer_info["pkf"] = layer_info["okf"] = layer_info["skf"] = layer_info["rxkf"] = layer_info["rykf"] = layer_info["rzkf"] = False

        # 获取起始帧和结束帧
        start_frame = 0
        end_frame = layer_frames

        # 处理透明度和可见性
        opacity_kf = layer_data["data"]["opacity"]
        visiable_kf = layer_data["data"]["visible"]
        opacity_kf_new = [round(opacity_kf[j] * visiable_kf[j] * 10000) / 10000 for j in range(layer_frames)]
        while opacity_kf_new[start_frame] == 0 and start_frame < layer_frames - 1:
            start_frame += 1
        while opacity_kf_new[end_frame - 1] == 0 and end_frame > 0:
            end_frame -= 1
        if start_frame >= end_frame:
            continue

        opacity_kf_new = opacity_kf_new[start_frame: end_frame]
        if check_keypoint_all_same(opacity_kf_new):
            layer_info["oa"] = False
            layer_info["o"] = opacity_kf_new[0]
        else:
            layer_info["oa"] = True
            layer_info["o"] = [mars_curve_fitter.fit_hermite_curves_from_points(opacity_kf_new)]

        layer_info["st"] = start_frame / fps
        layer_info["ed"] = end_frame / fps

        # 处理缩放
        scale_kf = layer_data["data"]["size"]
        if layer_size is None:
            layer_size = [1e-6] * 3
            for k in range(3):
                for j in range(start_frame, end_frame):
                    layer_size[k] = max(layer_size[k], scale_kf[j][k])

        check_scale_all_same = True
        scale_k = [[], [], []]
        for k in range(3):
            scale_k[k] = [round(scale_kf[j][k] / layer_size[k] * 10000) / 10000 for j in range(start_frame, end_frame)]
            if not check_keypoint_all_same(scale_k[k]):
                check_scale_all_same = False
        if check_scale_all_same:
            layer_info["sa"] = False
            layer_info["s"] = [scale_k[0][0], scale_k[1][0], scale_k[2][0]]
        else:
            layer_info["sa"] = True
            if np.allclose(scale_k[0], scale_k[1], 1e-3, 1e-3):
                layer_info["s"] = [mars_curve_fitter.fit_hermite_curves_from_points(scale_k[0])]
            else:
                layer_info["s"] = [[], [], []]
                for k in range(3):
                    layer_info["s"][k] = mars_curve_fitter.fit_hermite_curves_from_points(scale_k[k])

        layer_info["w"] = layer_size[0]
        layer_info["h"] = layer_size[1]
        layer_info["a"] = [layer_info["w"] / 2, layer_info["h"] / 2]

        # 处理旋转
        rotation_kf = layer_data["data"]["rotation"]
        rotation_kf_x = [rotation_kf[j][0] for j in range(start_frame, end_frame)]
        rotation_kf_y = [rotation_kf[j][1] for j in range(start_frame, end_frame)]
        rotation_kf_z = [rotation_kf[j][2] for j in range(start_frame, end_frame)]

        if check_keypoint_all_same(rotation_kf_x):
            layer_info["rxa"] = False
            layer_info["rx"] = rotation_kf_x[0]
        else:
            layer_info["rxa"] = True
            layer_info["rx"] = [mars_curve_fitter.fit_hermite_curves_from_points(rotation_kf_x)]

        if check_keypoint_all_same(rotation_kf_y):
            layer_info["rya"] = False
            layer_info["ry"] = rotation_kf_y[0]
        else:
            layer_info["rya"] = True
            layer_info["ry"] = [mars_curve_fitter.fit_hermite_curves_from_points(rotation_kf_y)]

        if check_keypoint_all_same(rotation_kf_z):
            layer_info["rza"] = False
            layer_info["rz"] = rotation_kf_z[0]
        else:
            layer_info["rza"] = True
            layer_info["rz"] = [mars_curve_fitter.fit_hermite_curves_from_points(rotation_kf_z)]

        # 处理路径
        position_kf = layer_data["data"]["position"]
        layer_info["p"] = [[]]
        layer_info["pp"] = []
        layer_info["cp"] = []
        pa, path_fitter_results = mars_curve_fitter.fit_bezier_path_from_points(position_kf[start_frame: end_frame])
        if pa:
            layer_info["pa"] = True
            layer_info["p"][0] = path_fitter_results[0]
            layer_info["pp"] = path_fitter_results[1]
            layer_info["cp"] = path_fitter_results[2]
        else:
            layer_info["pa"] = False
            layer_info["p"] = path_fitter_results

        parsed_layers.append(layer_info)
    parsed_layers.reverse()
    mars_json = mars_generator_layered.generate_mars_json_by_layers(
        parsed_layers, g_width, g_height, name, total_frames / fps, "local")
    return mars_json


def get_mars_visualize_url(mars_json_str):
    """
        Get Mars visualize url.

    :param mars_json_str: mars_json_str
    :return:
        Mars preview url
    """
    with open("mars_json_output.json", "w") as f:
        f.write(mars_json_str)
        f.close()
        mars_json_url = global_tools.upload_file("mars_json_output.json")
        return "https://render.alipay.com/p/c/mars-demo/test.html?version=1.1.41&file=" + mars_json_url
