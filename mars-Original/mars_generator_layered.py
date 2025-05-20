import threading
import json
from queue import Queue
from mars import mars_camera, mars_generator_base, mars_enum_data

layer_info_template = {
    "st": None,
    "ed": None,
    "nm": None,
    "img": None,
    "w": None,
    "h": None,
    "o": 1,
    "oa": False,
    "r": 0,
    "ra": False,
    "a": [0, 0, 0],
    "aa": False,
    "s": 1,
    "sa": False,
    "p": [0, 0, 0],
    "pa": False,
    "pp": None,
    "cp": None,
    "parent": None,
    "is_parent": False,
    "id": None,
    "type": None,
    "mars_id": None,
    "masked": 0
}


def set_item_alpha_by_layer(item, layer, item_type):
    """
        Set Alpha for Mars layer

    :param item: Mars item
    :param layer: layer animation info
    :param item_type: item type
    :return:
        Mars item JSON
    """

    # set opacity
    if layer["oa"]:
        # key frame mode
        if layer["okf"]:
            item["content"]["colorOverLifetime"] = {
                "opacity": [mars_enum_data.ValueType.LINES.value, [[i[0], i[1]] for i in layer["o"][0]]]
            }
        else:
            item["content"]["colorOverLifetime"] = {
                "opacity": [mars_enum_data.ValueType.CURVE.value, layer["o"][0]]
            }
    elif item_type != "null":  # Warning! lottie parent's alpha does not effect sons, but mars does.
        layer["o"] = layer["o"][0] if type(layer["o"]) is list else layer["o"]
        if layer["o"] != 1:
            item["content"]["colorOverLifetime"] = {
                "opacity": [mars_enum_data.ValueType.CONSTANT.value, layer["o"]]
            }
    return item


def set_item_animation_by_layer(item, layer):
    """
        Set Animation for Mars layer

    :param item: Mars item
    :param layer: layer animation info
    :return:
        Mars item JSON
    """

    # set rotation (disable axis rotation)
    if layer["ra"]:
        if layer["rkf"]:
            item["content"]["rotationOverLifetime"] = {
                "z": [mars_enum_data.ValueType.LINES.value, [[i[0], i[1]] for i in layer["r"][0]]],
                "asRotation": True
            }
        else:
            item["content"]["rotationOverLifetime"] = {
                "z": [mars_enum_data.ValueType.CURVE.value, layer["r"][0]],
                "asRotation": True
            }
    else:
        layer["r"] = layer["r"][0] if type(layer["r"]) is list else layer["r"]
        if layer["r"] != 0:
            item["content"]["rotationOverLifetime"] = {
                "z": [mars_enum_data.ValueType.CONSTANT.value, layer["r"]],
                "asRotation": True
            }

    for axis in ["x", "y", "z"]:
        r = "r" + axis
        if r in layer:
            if "rotationOverLifetime" in item["content"]:
                item["content"]["rotationOverLifetime"]["separateAxes"] = True
            else:
                item["content"]["rotationOverLifetime"] = {
                    "separateAxes": True,
                    "asRotation": True
                }
            if r + "a" in layer and layer[r + "a"]:
                if layer[r + "kf"]:
                    item["content"]["rotationOverLifetime"][axis] = [mars_enum_data.ValueType.LINES.value,
                                                                     [[i[0], i[1]] for i in layer[r][0]]]
                else:
                    item["content"]["rotationOverLifetime"][axis] = [mars_enum_data.ValueType.CURVE.value, layer[r][0]]
            else:
                layer[r] = layer[r][0] if type(layer[r]) is list else layer[r]
                if layer[r] != 0:
                    item["content"]["rotationOverLifetime"][axis] = [mars_enum_data.ValueType.CONSTANT.value, layer[r]]

    if "rotationOverLifetime" in item["content"]:
        if "x" not in item["content"]["rotationOverLifetime"] and "y" not in item["content"]["rotationOverLifetime"]:
            if "z" in item["content"]["rotationOverLifetime"]:
                item["content"]["rotationOverLifetime"]["separateAxes"] = False
            else:
                del item["content"]["rotationOverLifetime"]

    # set size
    if layer["sa"]:
        if layer["skf"]:
            if len(layer["s"]) == 3:
                item["content"]["sizeOverLifetime"] = {
                    "separateAxes": True,
                    "x": [mars_enum_data.ValueType.LINES.value, [[i[0], i[1]] for i in layer["s"][0]]],
                    "y": [mars_enum_data.ValueType.LINES.value, [[i[0], i[1]] for i in layer["s"][1]]],
                    "z": [mars_enum_data.ValueType.LINES.value, [[i[0], i[1]] for i in layer["s"][2]]]
                }
            elif len(layer["s"]) == 2:
                item["content"]["sizeOverLifetime"] = {
                    "separateAxes": True,
                    "x": [mars_enum_data.ValueType.LINES.value, [[i[0], i[1]] for i in layer["s"][0]]],
                    "y": [mars_enum_data.ValueType.LINES.value, [[i[0], i[1]] for i in layer["s"][1]]],
                    "z": [mars_enum_data.ValueType.CONSTANT.value, 1]
                }
            else:
                item["content"]["sizeOverLifetime"] = {
                    "size": [mars_enum_data.ValueType.LINES.value, [[i[0], i[1]] for i in layer["s"][0]]]
                }
        else:
            if len(layer["s"]) == 3:
                item["content"]["sizeOverLifetime"] = {
                    "separateAxes": True,
                    "x": [mars_enum_data.ValueType.CURVE.value, layer["s"][0]],
                    "y": [mars_enum_data.ValueType.CURVE.value, layer["s"][1]],
                    "z": [mars_enum_data.ValueType.CURVE.value, layer["s"][2]]
                }
            elif len(layer["s"]) == 2:
                item["content"]["sizeOverLifetime"] = {
                    "separateAxes": True,
                    "x": [mars_enum_data.ValueType.CURVE.value, layer["s"][0]],
                    "y": [mars_enum_data.ValueType.CURVE.value, layer["s"][1]],
                    "z": [mars_enum_data.ValueType.CONSTANT.value, 1]
                }
            else:
                item["content"]["sizeOverLifetime"] = {
                    "size": [mars_enum_data.ValueType.CURVE.value, layer["s"][0]]
                }
    else:
        if type(layer["s"]) is list:
            if len(layer["s"]) == 3 and (layer["s"][0] != 1 or layer["s"][1] != 1 or layer["s"][2] != 1):
                item["content"]["sizeOverLifetime"] = {
                    "separateAxes": True,
                    "x": [mars_enum_data.ValueType.CONSTANT.value, layer["s"][0]],
                    "y": [mars_enum_data.ValueType.CONSTANT.value, layer["s"][1]],
                    "z": [mars_enum_data.ValueType.CONSTANT.value, layer["s"][2]]
                }
            elif len(layer["s"]) == 2 and (layer["s"][0] != 1 or layer["s"][1] != 1):
                item["content"]["sizeOverLifetime"] = {
                    "separateAxes": True,
                    "x": [mars_enum_data.ValueType.CONSTANT.value, layer["s"][0]],
                    "y": [mars_enum_data.ValueType.CONSTANT.value, layer["s"][1]],
                    "z": [mars_enum_data.ValueType.CONSTANT.value, 1]
                }
            elif layer["s"][0] != 1:
                item["content"]["sizeOverLifetime"] = {
                    "size": [mars_enum_data.ValueType.CONSTANT.value, layer["s"][0]]
                }
        elif layer["s"] != 1:
            item["content"]["sizeOverLifetime"] = {
                "size": [mars_enum_data.ValueType.CONSTANT.value, layer["s"]]
            }

    if layer["pa"]:
        path = [layer["p"][0]]
        path_points = []
        for i in range(len(layer["pp"])):
            path_points.append([layer["pp"][i][0][0], layer["pp"][i][0][1], layer["pp"][i][0][2]])
        if len(path_points) < len(layer["p"][0]):
            path_points.append([layer["pp"][-1][1][0], layer["pp"][-1][1][1], layer["pp"][i][0][2]])

        path.append(path_points)
        ctrl_points = []
        for i in range(len(layer["cp"])):
            if layer["pkf"]:
                layer["cp"][i][0][0] = (layer["pp"][i][1][0] - layer["pp"][i][0][0]) / 3
                layer["cp"][i][0][1] = (layer["pp"][i][1][1] - layer["pp"][i][0][1]) / 3
                layer["cp"][i][1][0] = -(layer["pp"][i][1][0] - layer["pp"][i][0][0]) / 3
                layer["cp"][i][1][1] = -(layer["pp"][i][1][1] - layer["pp"][i][0][1]) / 3
            if len(ctrl_points) < 2 * (len(layer["p"][0]) - 1):
                ctrl_points.append(
                    [layer["cp"][i][0][0] + layer["pp"][i][0][0],
                     layer["cp"][i][0][1] + layer["pp"][i][0][1],
                     layer["cp"][i][0][2] + layer["pp"][i][0][2]])
                ctrl_points.append(
                    [layer["cp"][i][1][0] + layer["pp"][i][1][0],
                     layer["cp"][i][1][1] + layer["pp"][i][1][1],
                     layer["cp"][i][1][2] + layer["pp"][i][1][2]])

        path.append(ctrl_points)
        item["content"]["positionOverLifetime"] = {"path": [mars_enum_data.ValueType.BEZIER_PATH.value, path]}
        item["transform"]["position"] = [0, 0, 0]
    else:
        item["transform"]["position"] = layer["p"]

    return item


def generate_mars_json_by_layers(layers, g_width, g_height, name, duration, request_id):
    """
        Generate Mars JSON from layers info

    :param layers: layer info
    :param g_width: screen width
    :param g_height: screen height
    :param name: name
    :param duration: duration
    :return:
        Mars JSON String
    """

    def parse_image(q, i, img_data, img_type):
        q.put((i, img_data))

    mars_json_data = mars_generator_base.get_basic_mars_json(name, g_width, g_height, duration)

    mars_id = 1
    mars_tex_dict = {}
    for i, layer in enumerate(layers):
        layer["mars_id"] = mars_id
        if layer["type"] == "sprite":  # type is sprite
            mars_w = layer["w"]
            mars_h = layer["h"]

            mars_tex_id = mars_tex_dict[layer["img"]] if layer["img_type"] != "raw" and layer[
                "img"] in mars_tex_dict else len(mars_tex_dict)
            sprite_item = mars_generator_base.get_basic_sprite(layer["id"], mars_tex_id, mars_w, mars_h, 0, 0, 0,
                                                               layer["nm"], layer["ed"] - layer["st"], layer["st"])
            sprite_item = set_item_alpha_by_layer(sprite_item, layer, "sprite")
            sprite_item = set_item_animation_by_layer(sprite_item, layer)

            if "masked" in layer and layer["masked"]:
                sprite_item["content"]["renderer"]["maskMode"] = layer["masked"]

            if "mask" in layer:
                sprite_item["content"]["renderer"]["shape"] = len(mars_json_data["shapes"])
                sprite_item["content"]["renderer"]["maskMode"] = 1
                mars_json_data["shapes"].append(layer["mask"])

            if "anchor" in layer and layer["anchor"] != [0.5, 0.5]:
                sprite_item["content"]["renderer"]["anchor"] = layer["anchor"]

            if "parent" in layer:
                sprite_item["parentId"] = layer["parent"]

            mars_json_data["compositions"][0]["items"].append(sprite_item)
            mars_id += 1
            if layer["img_type"] == "raw":
                mars_tex_dict["raw" + str(i)] = len(mars_tex_dict)
                mars_json_data["images"].append({"img": layer["img"], "img_type": layer["img_type"]})
            else:
                if layer["img"] not in mars_tex_dict:
                    mars_tex_dict[layer["img"]] = len(mars_tex_dict)
                    mars_json_data["images"].append({"img": layer["img"], "img_type": layer["img_type"]})

        else:  # type is null
            null_item = mars_generator_base.get_basic_null(
                layer["id"], 1, 1, 0, 0, 0, layer["nm"], layer["ed"] - layer["st"], layer["st"])
            null_item = set_item_animation_by_layer(null_item, layer)
            if "parent" in layer:
                null_item["parentId"] = layer["parent"]
            mars_json_data["compositions"][0]["items"].append(null_item)
            mars_id += 1

    mars_json_data["compositions"][0]["items"].reverse()

    q = Queue()
    t_array = []
    for i in range(len(mars_json_data["images"])):
        t = threading.Thread(target=parse_image,
                             args=(q, i, mars_json_data["images"][i]["img"], mars_json_data["images"][i]["img_type"]))
        t_array.append(t)
        t.start()

    for i in range(len(t_array)):
        t_array[i].join()

    for i in range(len(t_array)):
        res = q.get()
        mars_json_data["images"][res[0]] = res[1]

    for i in range(len(t_array)):
        mars_json_data["images"][i] = mars_generator_base.get_mars_image(mars_json_data["images"][i])
        mars_json_data["textures"].append(mars_generator_base.get_mars_tex(i))

    for i in range(len(mars_json_data["images"])):
        mars_json_data["imgUsage"]["1"].append(i)
    json_str = json.dumps(mars_json_data)
    return json_str
