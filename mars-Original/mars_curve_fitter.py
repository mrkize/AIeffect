import numpy as np
from mars.bezier_fitter import fitCurve
import copy


def get_line_cof(p1, p2):
    """
        get line coefficient from two points in a line.

        # (y-y1)/(y2-y1)=(x-x1)/(x2-x1)
        # (y2-y1)x + (x1-x2) y - x1y2 + x2y1 = 0
        # a=y2-y1, b=x1-x2, c=-x1y2+x2y1
    :param p1: point in line 1
    :param p2: point in line 2
    :return:
        line coefficient
    """

    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = - p1[0] * p2[1] + p2[0] * p1[1]
    return a, b, c


def cal_dist(p, a, b, c):
    """
        Calculate distance from a point to a line(described by coefficient).

    :param p: point
    :param a: line cof a
    :param b: line cof b
    :param c: line cof c
    :return:
        distance
    """

    dist = abs(a * p[0] + b * p[1] + c) / pow(a * a + b * b, 0.5)
    return dist


def cal_dist_raw(p, p1, p2):
    """
        Calculate distance from a point to a line(described by two point).

    :param p: point
    :param p1: point in line 1
    :param p2: point in line 2
    :return:
        distance
    """

    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = - p1[0] * p2[1] + p2[0] * p1[1]
    dist = abs(a * p[0] + b * p[1] + c) / pow(a * a + b * b, 0.5)
    return dist


def douglas_peuker_splitter(points, epsilon=0.05):
    """
        Douglas peuker algorithm to simplify a list of points. Only support 2D now.

    :param points: points
    :param epsilon: epsilon to control simplify roughness
    :return:
        simplified_points, simplified_split_indexes
    """

    length = len(points)

    stack = [(0, length - 1)]
    flags = [False] * length
    while len(stack) > 0:
        inteval = stack.pop()
        begin, end = inteval
        dmax = -1
        index = -1
        a, b, c = get_line_cof(points[begin], points[end])
        for i in range(begin + 1, end):
            d = cal_dist(points[i], a, b, c)
            if d > dmax:
                index = i
                dmax = d

        if dmax > epsilon:
            stack.append((begin, index))
            stack.append((index, end))
        else:
            flags[begin] = True
            flags[end] = True

    simplified_points = []
    simplified_indexes = []
    for i in range(length):
        if flags[i]:
            simplified_points.append(points[i])
            simplified_indexes.append(i)

    return np.array(simplified_points), simplified_indexes


def direction_splitter(points, thresh=-0.1):
    """
        Direction based method to simplify a list of points. Support 2D and 3D now.

    :param points: points
    :param thresh: threshold to control simplify roughness
    :return:
        simplified_points, simplified_split_indexes
    """

    length = len(points)
    points = np.array(points)

    p_diff = [(points[i + 2] - points[i]) / 2 for i in range(length - 2)]
    p_diff.insert(0, points[1] - points[0])
    p_diff.append(points[length - 1] - points[length - 2])
    flags = [False] * length
    flags[0] = flags[-1] = True

    for i in range(length - 1):
        dot_product = np.dot(p_diff[i], p_diff[i + 1])
        norm_v1 = np.linalg.norm(p_diff[i])
        norm_v2 = np.linalg.norm(p_diff[i + 1])

        if norm_v1 >= 1e-6 and norm_v2 >= 1e-6:
            cos_angle = dot_product / (norm_v1 * norm_v2)
            if cos_angle <= thresh:
                flags[i] = True
        if norm_v1 >= 1e-6 and norm_v2 < 1e-6:
            flags[i] = True
        elif norm_v1 < 1e-6 and norm_v2 >= 1e-6:
            flags[i] = True

    simplified_points = []
    simplified_indexes = []
    for i in range(length):
        if flags[i]:
            simplified_points.append(points[i])
            simplified_indexes.append(i)

    return np.array(simplified_points), simplified_indexes


def solve_least_square(X, y):
    """
        Solve least square problem.

    :param X: LS matrix X
    :param y: LS target y
    :return:
        LS result
    """

    return np.linalg.inv(X.transpose() * X + 1e-6 * np.eye(X.shape[1])) * X.transpose() * y


def fit_hermite(points, kv1, kv2):
    """
        Fit Mars hermite curve (predict m0,m1) according follower code.

        function curveValueEvaluate(time: number, keyframe0: Array, keyframe1: Array) {
              const dt = keyframe1[ CURVE_PRO_TIME ] - keyframe0[ CURVE_PRO_TIME ];

              const m0 = keyframe0[ CURVE_PRO_OUT_TANGENT ] * dt;
              const m1 = keyframe1[ CURVE_PRO_IN_TANGENT ] * dt;

              const t = (time - keyframe0[ CURVE_PRO_TIME ]) / dt;
              const t2 = t * t;
              const t3 = t2 * t;

              const a = 2 * t3 - 3 * t2 + 1;
              const b = t3 - 2 * t2 + t;
              const c = t3 - t2;
              const d = -2 * t3 + 3 * t2;
              //(2*v0+m0+m1-2*v1)*(t-t0)^3/k^3+(3*v1-3*v0-2*m0-m1)*(t-t0)^2/k^2+m0 *(t-t0)/k+v0
              return a * keyframe0[ CURVE_PRO_VALUE ] + b * m0 + c * m1 + d * keyframe1[ CURVE_PRO_VALUE ];
        }

    :param points: sample points
    :return:
        m0 (tan_in), m1(tan_out)
    """
    num_points = len(points)
    X_array, Y_array = [], []
    for i in range(num_points):
        x = points[i][0]
        y = points[i][1]
        X_array.append([pow(x, 3) - 2 * pow(x, 2) + x, pow(x, 3) - pow(x, 2)])
        Y_array.append([y - (2 * pow(x, 3) - 3 * pow(x, 2) + 1) * kv1 - (-2 * pow(x, 3) + 3 * pow(x, 2)) * kv2])

    X, Y = np.mat(X_array), np.mat(Y_array)
    m = solve_least_square(X, Y)
    m0, m1 = np.array(m)[0][0], np.array(m)[1][0]

    # for i in range(num_points):
    #     x = i/(num_points-1)
    #     y = m0 * (pow(x, 3) - 2 * pow(x, 2) + x) + m1*(pow(x, 3) - pow(x, 2))- 2 * pow(x, 3) + 3 * pow(x, 2)
    return m0, m1


def fit_hermite_curves_from_points(values=None):
    # print(values)
    num_points = len(values)
    if num_points < 1:
        print("Error ! num_points should >=2")
    elif num_points == 1:
        num_points = 2
        values = [values[0], values[0]]

    points = [[i / (num_points - 1), v if v is not None else 0] for i, v in enumerate(values)]
    # print(points)
    simplified_splits, simplified_indexes = douglas_peuker_splitter(points, epsilon=0.05)

    # print(simplified_splits, simplified_indexes, num_points)

    if len(simplified_indexes) <= 1:
        simplified_indexes = [0, num_points - 1]

    results = []
    min_v = 1000000
    max_v = -1000000
    for i in range(len(simplified_indexes) - 1):
        start_idx = simplified_indexes[i]
        end_idx = simplified_indexes[i + 1] + 1
        points_clip = copy.deepcopy(points[start_idx: end_idx])
        start_x = points_clip[0][0]
        end_x = points_clip[-1][0]
        start_y = points_clip[0][1]
        end_y = points_clip[-1][1]
        for j in range(len(points_clip)):
            points_clip[j][0] = (points_clip[j][0] - start_x) / (end_x - start_x)
        m0, m1 = fit_hermite(points_clip, points[start_idx][1], points[end_idx - 1][1])
        min_v = min(min_v, points[start_idx][1], points[end_idx - 1][1])
        max_v = max(max_v, points[start_idx][1], points[end_idx - 1][1])
        if i == 0:
            results.append([start_x, start_y, 0, m0])
            results.append([end_x, end_y, m1, 0])
        elif i < len(simplified_indexes) - 2:
            # 平滑一下
            results[-1][3] = results[-1][2] = (results[-1][2] + m0) / 2
            results.append([end_x, end_y, m1, 0])
        else:
            results[-1][3] = m0
            results.append([end_x, end_y, m1, 0])

    for res in results:
        res[2] /= max(1e-6, max_v - min_v)
        res[3] /= max(1e-6, max_v - min_v)

    # print("hermite results", results)
    return results


def fit_bezier_path_from_points(points=None, maxError=3):
    num_points = len(points)
    simplified_splits, simplified_indexes = direction_splitter(points)
    # print("path p", points)
    # print(simplified_splits, simplified_indexes, num_points)
    if len(simplified_indexes) <= 1:
        simplified_indexes = [0, num_points - 1]
    # print(simplified_indexes)
    # check is all the same
    sum_length = 0
    for i in range(1, num_points):
        sum_length += np.linalg.norm(np.array(points[i]) - np.array(points[i - 1]))
    if sum_length < 1e-6 * num_points:
        return False, points[0]

    results = [[], [], []]
    results[0].append([0, 0, 1, 1])
    for i in range(len(simplified_indexes) - 1):
        start_idx = simplified_indexes[i]
        end_idx = simplified_indexes[i + 1] + 1
        points_clip = copy.deepcopy(points[start_idx: end_idx])
        # to do, max error should be scaled
        bezier_curves = fitCurve(np.array(points_clip), maxError)
        # 根据曲线的长度获取分段点的时间比例
        bezier_curves_length_sum = []
        bezier_curves_length_total = 0
        for curve in bezier_curves:
            length = 0
            p0 = curve[0]
            for splitId in range(101):
                t = splitId / 100
                p1 = (1 - t) ** 3 * curve[0] + 3 * (1 - t) ** 2 * t * curve[1] + 3 * (1 - t) * t ** 2 * curve[
                    2] + t ** 3 * curve[3]
                length += np.linalg.norm(p1 - p0)
                p0 = p1
            bezier_curves_length_total += length
            bezier_curves_length_sum.append(bezier_curves_length_total)

        for j, curve in enumerate(bezier_curves):
            # print("start_idx", start_idx, "end_idx", end_idx, "num_points", num_points, "leni", bezier_curves_length_sum[j], "lent", bezier_curves_length_total)
            if bezier_curves_length_total > 0:
                time = start_idx / num_points + bezier_curves_length_sum[j] / bezier_curves_length_total * (
                            end_idx - start_idx) / num_points
            else:
                time = (end_idx - start_idx) / num_points
            results[0].append([time, time, 1, 1])
            results[1].append([curve[0].tolist(), curve[3].tolist()])
            results[2].append([(curve[1] - curve[0]).tolist(), (curve[2] - curve[3]).tolist()])

    return True, results


if __name__ == '__main__':
    values = [i / 10 for i in range(10)] + [1 - i / 10 for i in range(10)]  # + [i*i/100 for i in range(11)]
    num_points = len(values)
    points = [[0, v] for i, v in enumerate(values)]
    print(points)
    simplified_splits, simplified_indexes = direction_splitter(points)
    print(simplified_splits, simplified_indexes)
    # fit_hermite_curves_from_points(values)
