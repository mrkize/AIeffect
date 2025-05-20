import json
import os
import time
import numpy as np
# import global_tools
import re
from mars import mars_enum_data, mars_camera, global_tools


def get_basic_mars_json(name, width, height, duration=2, end_behavior="restart", clip_mode=1):
    """
        Get basic mars json
    :param name: name
    :param width: width
    :param height: height
    :param duration: duration
    :param end_behavior: end_behavior
    :param clip_mode: clip mode
    :return:
        mars json data
    """

    # load mars json template
    with open("./mars/sources/template_mars.json", "r") as f:
        mars_json_data = json.load(f)
        mars_json_data["compositions"][0]["name"] = name
        mars_json_data["compositions"][0]["previewSize"] = [width, height]
        mars_json_data["compositions"][0]["duration"] = duration
        mars_json_data["compositions"][0]["endBehavior"] = \
            mars_enum_data.CompositionEndBehavior[end_behavior.upper()].value
        mars_json_data["compositions"][0]["camera"]["clipMode"] = clip_mode
    return mars_json_data


def get_basic_sprite(id, img_id, w, h, x, y, z,
                     name=None, duration=2, delay=0, end_behavior="destroy"):
    """
        Get basic sprite item in Mars.

    :param id: layer id
    :param img_id: image/texture id
    :param w: width
    :param h: height
    :param x: x
    :param y: y
    :param z: z
    :param name: name (optional)
    :param duration: duration
    :param delay: delay
    :param end_behavior: end behavior(default is destroy)
    :return:
        sprite json
    """

    result = {}
    result["id"] = str(id)
    result["name"] = name if name else "sprite_" + str(id)
    result["type"] = mars_enum_data.ItemType.SPRITE.value

    result["delay"] = delay
    result["duration"] = duration
    result["endBehavior"] = mars_enum_data.ItemEndBehavior[end_behavior.upper()].value

    result["transform"] = {}
    result["transform"]["position"] = [x, y, z]
    result["transform"]["scale"] = [w, h, 1]

    result["content"] = {}
    result["content"]["options"] = {}
    result["content"]["options"]["startColor"] = [1, 1, 1, 1]
    result["content"]["renderer"] = {}
    result["content"]["renderer"]["renderMode"] = 1
    result["content"]["renderer"]["texture"] = img_id
    return result


def get_basic_particle(id, img_id, w, h, x, y, z,
                       name=None, duration=2, delay=0, end_behavior="destroy"):
    """
        Get basic particle item in Mars.

    :param id: layer id
    :param img_id: image/texture id
    :param w: width
    :param h: height
    :param x: x
    :param y: y
    :param z: z
    :param name: name (optional)
    :param duration: duration
    :param delay: delay
    :param end_behavior: end behavior(default is destroy)
    :return:
        particle json
    """

    result = {}
    result["id"] = str(id)
    result["name"] = name if name else "particle_" + str(id)
    result["type"] = mars_enum_data.ItemType.PARTICLE.value

    result["delay"] = delay
    result["duration"] = duration
    result["endBehavior"] = mars_enum_data.ItemEndBehavior[end_behavior.upper()].value

    result["transform"] = {}
    result["transform"]["position"] = [x, y, z]
    # result["transform"]["scale"] = [w, h, 1]

    result["content"] = {}
    result["content"]["renderer"] = {}
    result["content"]["renderer"]["renderMode"] = 1
    result["content"]["renderer"]["texture"] = img_id
    result["content"]["options"] = {}
    result["content"]["options"]["maxCount"] = 10
    result["content"]["options"]["startLifetime"] = [mars_enum_data.ValueType.CONSTANT.value, 0.6 * duration]
    result["content"]["options"]["startColor"] = [mars_enum_data.ValueType.RGBA_COLOR.value, [1, 1, 1, 1]]
    result["content"]["options"]["startSize"] = [mars_enum_data.ValueType.CONSTANT.value, w]
    result["content"]["options"]["sizeAspect"] = [mars_enum_data.ValueType.CONSTANT.value, w / h]
    result["content"]["emission"] = {"rateOverTime": [mars_enum_data.ValueType.CONSTANT.value, 6]}
    result["content"]["colorOverLifetime"] = {"opacity": [mars_enum_data.ValueType.LINES.value, [[0, 1], [1, 0]]]}
    result["content"]["positionOverLifetime"] = {"startSpeed": [mars_enum_data.ValueType.CONSTANT.value, w * 2]}

    return result


def get_basic_null(id, w, h, x, y, z,
                   name=None, duration=2, delay=0, end_behavior="destroy"):
    """
        Get basic cal item in Mars.

    :param id: layer id
    :param w: width
    :param h: height
    :param x: x
    :param y: y
    :param z: z
    :param duration: duration
    :param delay: delay
    :param name: name (optional)
    :param end_behavior: end behavior(default is destroy)
    :return:
        cal json
    """

    result = {}
    result["id"] = str(id)
    result["name"] = name if name else "null_" + str(id)
    result["type"] = mars_enum_data.ItemType.NULL.value

    result["delay"] = delay
    result["duration"] = duration
    result["endBehavior"] = mars_enum_data.ItemEndBehavior[end_behavior.upper()].value

    result["transform"] = {}
    result["transform"]["position"] = [x, y, z]
    result["transform"]["scale"] = [w, h, 1]

    result["content"] = {}

    result["content"]["options"] = {}
    result["content"]["options"]["startColor"] = [1, 1, 1, 1]
    # result["content"]["options"]["size"] = [w, h]

    result["content"]["renderer"] = {}
    result["content"]["renderer"]["renderMode"] = 1
    # result["cal"]["options"]["relative"] = True

    return result


def get_basic_light(id, img_id, w, x, y, z, duration=2, delay=0, end_behavior="destroy", duration_ratio=0.5833):
    """
        Get basic light item in Mars.

    :param id: layer id
    :param img_id: image/texture id
    :param w: width
    :param x: x
    :param y: y
    :param z: z
    :param duration: duration
    :param delay: delay
    :param end_behavior: end_behavior
    :param duration_ratio: duration_ratio
    :return:
        light json
    """

    result = {}
    result["id"] = str(id)
    result["name"] = "light_" + str(id)
    result["type"] = mars_enum_data.ItemType.SPRITE.value

    result["delay"] = delay
    result["duration"] = duration
    result["endBehavior"] = mars_enum_data.ItemEndBehavior[end_behavior.upper()].value

    result["transform"] = {}
    result["transform"]["position"] = [x, y, z]
    result["transform"]["scale"] = [w * 0.5, w * 2.5, 0]
    result["transform"]["rotation"] = [0, 0, 30]

    result["content"] = {}
    result["content"]["options"] = {}
    result["content"]["options"]["startColor"] = [1, 1, 1, 0.3]
    result["content"]["renderer"] = {}
    result["content"]["renderer"]["renderMode"] = 1
    result["content"]["renderer"]["maskMode"] = 2
    result["content"]["renderer"]["blending"] = 5
    result["content"]["renderer"]["texture"] = img_id
    result["content"]["positionOverLifetime"] = {"path": [mars_enum_data.ValueType.LINEAR_PATH.value,
                                                          [[[0, 0.11, 3.39, 3.39], [duration_ratio, 1, -0.02, -0.02]],
                                                           [[x - w / 1.3, y, z], [x + w / 1.3, y, z]]]]}

    return result


def get_basic_matte(id, w, h, x, y, z, shape_id, duration=2, delay=0, end_behavior="destroy"):
    """
        Get basic matte item in Mars.

    :param id: layer id
    :param w: width
    :param h: height
    :param x: x
    :param y: y
    :param z: z
    :param shape_id: matte shape id
    :param delay: delay
    :param end_behavior: end_behavior
    :return:
        matte json
    """

    result = {}
    result["id"] = str(id)
    result["name"] = "matte_" + str(id)
    result["type"] = mars_enum_data.ItemType.SPRITE.value

    result["delay"] = delay
    result["duration"] = duration
    result["endBehavior"] = mars_enum_data.ItemEndBehavior[end_behavior.upper()].value

    result["transform"] = {}
    result["transform"]["position"] = [x, y, z]
    result["transform"]["scale"] = [w, h, 1]

    result["content"] = {}
    result["content"]["options"] = {}
    result["content"]["options"]["startColor"] = [1, 1, 1, 0]
    result["content"]["renderer"] = {}
    result["content"]["renderer"]["renderMode"] = 1
    result["content"]["renderer"]["maskMode"] = 1
    result["content"]["renderer"]["shape"] = shape_id

    return result


def get_basic_shape(contours, samples=4):
    """
        get contour shape in Mars from contours array
    :param contours: contours array
    :param samples: sample num
    :return:
        contour shapes in Mars
    """

    def parse_contour(contour):
        contour_shape = dict()
        contour_shape["g"] = {}
        contour_shape["g"]["p"] = contour
        points = []
        splits = np.linspace(0, 1, samples + 1).tolist()
        for i in range(len(contour)):
            points.append(splits)
        contour_shape["g"]["s"] = points
        return contour_shape

    # if len(contours) == 1:
    #     contour_shapes = parse_contour(contours[0])
    # else:
    contour_shapes = [parse_contour(contour) for contour in contours]
    return contour_shapes


def get_matte(w, h, x, y, z, gw, gh, mars_effect, global_index, delay=0):
    """
        Get matte

    :param w: width
    :param h: height
    :param x: x
    :param y: y
    :param z: z
    :param gw: screen width
    :param gh: screen height
    :param mars_effect:
    :param global_index:
    :param delay: delay
    :return:
        matte
    """
    w, h, x, y, z = mars_camera.convert_2d_to_3d_bbox(x, y, z, w, h, gw, gh)
    time = mars_effect["time"] if "time" in mars_effect else 2
    end_behavior = mars_effect["end_behavior"] if "end_behavior" in mars_effect else "destroy"

    matte = get_basic_matte(global_index, w, h, x, y, z, None,
                            time, delay, end_behavior)
    return matte


def get_mars_tex(image_id):
    """
        Get mars texture.

    :param image_id: image id
    :return:
        Mars texture

    """
    mars_tex = {
        "source": image_id,
        "flipY": True
    }

    return mars_tex


def get_mars_image(image, add_webp=False, add_compress=False):
    """
        Get mars image.

    :param image: image data
    :param add_webp: convert and add webp source
    :param add_compress: convert and add compress source
    :param is_image_data: image is array data
    :return:
        Mars image

    """
    mars_image = {}

    image_data = None
    if type(image) == np.ndarray:
        image_data = image
        mars_image["url"] = global_tools.upload_image(image_data, global_tools.get_random_path("png"))
    elif type(image) == str:
        mars_image["url"] = image
    else:
        if "url" in image:
            mars_image["url"] = image["url"]
        if "webp" in image:
            mars_image["webp"] = image["webp"]
        if "template" in image:
            mars_image["template"] = image["template"]
        if "compress" in image:
            mars_image["compressed"] = image["compressed"]

    if add_webp and "webp" not in mars_image:
        if image_data is None:
            image_data = global_tools.open_url_image(image)
        mars_image["webp"] = global_tools.upload_image(
            image_data, global_tools.get_random_path("webp"))

    if add_compress:
        pass
    return mars_image


def convert_to_json_v1(mars_json_v0):
    """
        Convert Mars json v0 to v1.

    :param mars_json_v0:  mars json v0
    :return:
        mars json v1
    """

    mars_json_str = json.dumps(mars_json_v0)
    restr = r"[\/\\\:\*\?\"\'\<\>\|%]"
    name = re.sub(restr, "_", mars_json_v0["compositions"][0]["name"])
    mars_json_path = "temp/" + name + str(int(time.time() * 1000)) + ".json"

    with open(mars_json_path, "w", encoding="utf-8") as f:
        f.write(mars_json_str)
        f.close()

    cmd = "node -e \"require(\\\"%s\\\").init('%s')\"" % (
        "./js_scripts/mars_specification", mars_json_path)
    pipeline = os.popen(cmd)
    result = pipeline.read()

    with open(mars_json_path, "r", encoding="utf-8") as f:
        mars_json_v1 = json.load(f)
        f.close()
    return mars_json_v1
