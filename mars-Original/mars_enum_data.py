from enum import Enum


class ValueType(Enum):
    """
        基础数据类型
    """
    # 常数
    CONSTANT = 0
    # 二维常数
    CONSTANT_VEC2 = 1
    # 三维常数
    CONSTANT_VEC3 = 2
    # 四维常数
    CONSTANT_VEC4 = 3
    # 随机数
    RANDOM = 4
    # 直线
    LINES = 5
    # 曲线
    CURVE = 6
    # 贝塞尔路径
    BEZIER_PATH = 7
    # 颜色
    RGBA_COLOR = 8
    # 渐变色
    GRADIENT_COLOR = 9
    # 蒙版形状点集
    SHAPE_POINTS = 10
    # 蒙版形状切分
    SHAPE_SPLITS = 11
    # 直线路径
    LINEAR_PATH = 12
    # 多色
    COLORS = 13


class ItemType(Enum):
    """
        元素类型
    """
    # 错误元素
    BASE = "0"
    # 图层元素
    SPRITE = "1"
    # 粒子元素
    PARTICLE = "2"
    # 空节点元素
    NULL = "3"
    # 交互元素
    INTERACT = "4"
    # 插件元素
    PLUGIN = "5"
    # 模型元素
    MODEL = "6"
    # 预合成元素
    COMPOSITION = "7"
    # 滤镜图层
    FILTER = "8"


class ItemEndBehavior(Enum):
    """
        元素结束行为
    """
    # 销毁
    DESTROY = 0
    # 继续播放
    FORWARD = 4
    # 循环播放
    LOOP = 5


class CompositionEndBehavior(Enum):
    """
        合成结束行为
    """
    # 销毁
    DESTROY = 0
    # 暂停
    PAUSE = 1
    # 继续播放
    FORWARD = 2
    # 销毁并保留最后一帧
    PAUSE_DESTROY = 3
    # 冻结
    FREEZE = 4
    # 循环播放
    RESTART = 5


class ParticleShapeType(Enum):
    """
        粒子发射器形状
    """
    # 无
    NONE = 0
    # 圆球
    SPHERE = 1
    # 圆锥
    CONE = 2
    # 半球
    HEMISPHERE = 3
    # 圆
    CIRCLE = 4
    # 圆环
    DONUT = 5
    # 矩形
    RECTANGLE = 6
    # 矩形框
    RECTANGLE_EDGE = 7
    # 直线
    EDGE = 8
    # 贴图
    TEXTURE = 9


class MaskMode(Enum):
    """
        蒙版模式
    """
    # 无蒙版
    NONE = 0
    # 蒙版
    MASK = 1
    # 被遮挡
    OBSCURED = 2
    # 被反向遮挡
    REVERSE_OBSCURED = 3


class PluginType(Enum):
    """
        插件类型
    """
    # 陀螺仪
    GYROSCOPE = 0


class CompositionCheckErrorCode(Enum):
    """
        Result Code:
        0: success
        1: composition num != 1
        2: composition time != 2
        3: width height is not 400
        4: do not support 3d model (include camera animation)
        5: do not support spine
        6: camera parameter is not standard
        7: camera location is not [0,0,8]
        8: gravity in z is not 0
        9: direction in z is not 0
        10: do not support Z axis transform
        11: do not support orbitalX and orbitalY movement
        12: do not support Z axis movement
        13: render parse should >= 0
    """
    SUCCESS = 0
    COMP_NUM_ERR = 1
    COMP_TIME_ERR = 2
    COMP_SIZE_ERR = 3
    SUPPORT_MODEL_ERR = 4
    SUPPORT_SPINE_ERR = 5
    CAMERA_PARAM_ERR = 6
    CAMERA_LOC_ERR = 7
    GRAVITY_IN_Z_ERR = 8
    DIRECTION_IN_Z_ERR = 9
    PATH_IN_Z_ERR = 10
    ORBIT_IN_Z_ERR = 11
    MOVE_IN_Z_ERR = 12
    RENDER_ORDER_ERR = 13

