import math


# default camera parameters
default_clipmode = 1
default_camera_z = 8
default_camera_fov = 60


def get_2d_to_3d_ratio(g_width, g_height, camera_z=default_camera_z, camera_fov=default_camera_fov):
    """
       Get scale ratio from 2d screen to Mars

    :param g_width: whole image width
    :param g_height: whole image height
    :param camera_z:
    :param camera_fov:
    :return:
    """

    ratio = 2 / g_height * camera_z * math.tan(camera_fov/2 * math.pi / 180) + 1e-6

    if default_clipmode == 1:
        ratio *= g_height / g_width
    return ratio


def convert_2d_to_3d_bbox(x, y, z, width, height, g_width, g_height,
                          camera_z=default_camera_z, camera_fov=default_camera_fov):
    """
        Calculate 2D bounding box's coordinate in Mars

    :param x: position x
    :param y: position y
    :param z: position z
    :param width: bbox width
    :param height: bbox height
    :param g_width: whole image width
    :param g_height: whole image height
    :param camera_z: camera position z
    :param camera_fov: camera fov
    :return:
        width
        height
        offset_x
        offset_y
        offset_z
    """

    ratio = get_2d_to_3d_ratio(g_width, g_height, camera_z, camera_fov)
    adapt_ratio_z = (camera_z - z) / camera_z  # keep the size when setting different z
    w = ratio * width * adapt_ratio_z
    h = w / width * height
    x = (x - g_width / 2.0) * ratio * adapt_ratio_z
    y = (g_height / 2.0 - y) * ratio * adapt_ratio_z
    return w, h, x, y, z


def convert_2d_to_3d_point(x, y, z, g_width, g_height,
                           camera_z=default_camera_z, camera_fov=default_camera_fov):
    """
        Calculate 2D point's coordinate in Mars

    :param x: position x
    :param y: position y
    :param z: position z
    :param g_width: whole image width
    :param g_height: whole image height
    :param camera_z: camera position z
    :param camera_fov: camera fov
    :return:
        offset_x
        offset_y
        offset_z
    """

    ratio = get_2d_to_3d_ratio(g_width, g_height, camera_z, camera_fov)
    adapt_ratio_z = (camera_z - z) / camera_z
    x = (x - g_width / 2.0) * ratio * adapt_ratio_z
    y = (g_height / 2.0 - y) * ratio * adapt_ratio_z
    return x, y, z


def convert_2d_to_3d_by_axis(v, z, g_width, g_height, axis,
                             camera_z=default_camera_z, camera_fov=default_camera_fov):
    """
        Scale 2D point's coordinate in Mars by axis

    :param v: value
    :param z: position z
    :param g_width: whole image  width
    :param g_height: whole image  height
    :param axis: axis("x", "y")
    :param camera_z: camera position z
    :param camera_fov: camera fov
    :return:
        offset_by_axis
    """
    ratio = get_2d_to_3d_ratio(g_width, g_height, camera_z, camera_fov)
    adapt_ratio_z = (camera_z - z) / camera_z

    if axis == "x":
        return v * ratio * adapt_ratio_z
    elif axis == "y":
        return -v * ratio * adapt_ratio_z
    else:
        return v * ratio * adapt_ratio_z

def convert_3d_to_2d_point(x, y, z, g_width, g_height,
                           camera_z=default_camera_z, camera_fov=default_camera_fov):
    """
        Calculate 2D point's coordinate from Mars

    :param x: position x
    :param y: position y
    :param z: position z
    :param g_width: whole image  width
    :param g_height: whole image  height
    :param camera_z: camera position z
    :param camera_fov: camera fov
    :return:
        offset_x
        offset_y
    """

    ratio = get_2d_to_3d_ratio(g_width, g_height, camera_z, camera_fov)
    adapt_ratio_z = (camera_z - z) / camera_z
    screen_x = g_width / 2.0 + x / ratio / adapt_ratio_z
    screen_y = g_height / 2.0 - y / ratio * adapt_ratio_z
    return screen_x, screen_y
