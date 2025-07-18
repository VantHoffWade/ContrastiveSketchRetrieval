import os.path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import cv2
import random
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QGridLayout, QScrollArea
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

import global_defs
from data_utils.sketch_utils import get_allfiles, get_subdirs
import data_utils.sketch_utils as du
import encoders.spline as sp
from data_utils import sketch_file_read as fr
from data_utils.sketch_file_read import s5_read
from data_utils.utils import load_ret_file

def vis_sketch_folder(root=r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_all', shuffle=True, show_dot=True, dot_gap=3):
    files_all = get_allfiles(root)
    if shuffle:
        random.shuffle(files_all)

    for c_file in files_all:
        print(c_file)
        vis_sketch_orig(c_file, show_dot=show_dot, dot_gap=dot_gap)

    # classes = get_subdirs(root)
    # for c_class in classes:
    #     c_dir = os.path.join(root, c_class)
    #     c_files = get_allfiles(c_dir)
    #
    #     for idx in range(3):
    #         c_file_show = c_files[idx]
    #         print(c_file_show)
    #         vis_sketch_orig(c_file_show, show_dot=show_dot, dot_gap=dot_gap)


def vis_sketch_orig(root, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, title=None, show_dot=False, show_axis=False, dot_gap=1):
    """
    显示原始采集的机械草图
    存储的每行应该为： [x, y, state]
    :param root:
    :param pen_up: 抬笔指令
    :param pen_down: 落笔指令
    :param title: 落笔指令
    :param show_dot:
    :param show_axis:
    :return:
    """
    # -> [n, 4] col: 0 -> x, 1 -> y, 2 -> pen state (17: drawing, 16: stroke end), 3 -> None
    sketch_data = fr.load_sketch_file(root, delimiter=',')

    # 2D coordinates
    coordinates = sketch_data[:, :2]

    # sketch mass move to (0, 0), x y scale to [-1, 1]
    coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
    dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
    coordinates = coordinates / dist

    sketch_data[:, :2] = coordinates

    # 最后一行最后一个数改为17，防止出现空数组
    sketch_data[-1, 2] = pen_down

    # -------------------------------
    # 去掉点数过少的笔划
    # sketch_data = sp.stk_pnt_num_filter(sketch_data, 4)

    # split all strokes
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == pen_up)[0] + 1)

    # 重采样，使得点之间的距离近似相等
    # strokes = sp.batched_spline_approx(
    #     point_list=strokes,
    #     median_ratio=0.1,
    #     approx_mode='uni-arclength'
    # )

    for s in strokes:
        plt.plot(s[::dot_gap, 0], -s[::dot_gap, 1])

        if show_dot:
            plt.scatter(s[::dot_gap, 0], -s[::dot_gap, 1], s=80)

    if not show_axis:
        plt.axis('off')

    plt.axis("equal")
    plt.title(title)
    plt.show()


def vis_sketch_data(sketch_data, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, title=None, is_scale=False, show_dot=False):
    """
    显示原始采集的机械草图
    存储的每行应该为： [x, y, state]
    :param sketch_data:
    :param pen_up: 抬笔指令
    :param pen_down: 落笔指令
    :param title:
    :param is_scale: 是否将质心平移到 (0, 0)，且将草图大小缩放到 [-1, 1]^2
    :param show_dot:
    :return:
    """
    # 2D coordinates
    coordinates = sketch_data[:, :2]

    if is_scale:
        # sketch mass move to (0, 0), x y scale to [-1, 1]
        coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
        dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
        coordinates = coordinates / dist

    sketch_data[:, :2] = coordinates

    # 最后一行最后一个数改为17，防止出现空数组
    sketch_data[-1, 2] = pen_down

    # split all strokes
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == pen_up)[0] + 1)

    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])

        if show_dot:
            plt.scatter(s[:, 0], -s[:, 1])

    plt.axis('off')
    plt.title(title)
    plt.show()


def vis_s5_data(sketch_data, title=None, coor_mode="ABS",
                pen_up=global_defs.pen_up, pen_down=global_defs.pen_down):
    # 最后一行最后一个数改为17，防止出现空数组
    sketch_data[-1, 2] = pen_down

    # 获取当前tensor使用的设备和数据类型
    device = sketch_data.device
    dtype = sketch_data.dtype

    # 断定坐标模式在绝对模式和相对模式中
    if coor_mode not in ["ABS", "REL"]:
        raise ValueError("coor_mode must be 'ABS' or 'REL'")

    # 如果是相对模式则要对笔画进行一定处理
    if coor_mode == "REL":
        x, y = 0, 0
        strokes = []
        current_stroke = []
        sketch_data_detached = sketch_data.detach().cpu().numpy()
        for i, (dx, dy, p1, p2, p3) in enumerate(sketch_data_detached):
            x += dx
            y += dy
            current_stroke.append([x, y])

            if p2 == 1:  # pen-up：一个 stroke 结束
                strokes.append(torch.tensor(current_stroke, dtype=dtype, device=device))
                current_stroke = []
            elif p3 == 1:  # end-of-sequence：整幅图结束
                if current_stroke:
                    strokes.append(torch.tensor(current_stroke, dtype=dtype, device=device))
                break
    else:
        # split all strokes
        strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == pen_up)[0] + 1)

    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])

    if title is not None:
        plt.title(title)

    plt.axis('off')
    plt.show()


def vis_sketch_unified(root, n_stroke=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, show_dot=False):
    """
    显示笔划与笔划点归一化后的草图
    """
    # -> [n, 4] col: 0 -> x, 1 -> y, 2 -> pen state (17: drawing, 16: stroke end), 3 -> None
    sketch_data = np.loadtxt(root, delimiter=',')

    # 2D coordinates
    coordinates = sketch_data[:, :2]

    # sketch mass move to (0, 0), x y scale to [-1, 1]
    coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
    dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
    coordinates = coordinates / dist
    coordinates = coordinates.reshape([n_stroke, n_stk_pnt, 2])

    for i in range(n_stroke):
        plt.plot(coordinates[i, :, 0], -coordinates[i, :, 1])

        if show_dot:
            plt.scatter(coordinates[i, :, 0], -coordinates[i, :, 1])

    # plt.axis('off')
    plt.show()


def show_color(root, n_stroke=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt):
    # -> [n, 4] col: 0 -> x, 1 -> y, 2 -> pen state (17: drawing, 16: stroke end), 3 -> None
    sketch_data = np.loadtxt(root, delimiter=',')

    # 2D coordinates
    coordinates = sketch_data[:, :2]

    # sketch mass move to (0, 0), x y scale to [-1, 1]
    coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
    dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
    stroke = coordinates / dist
    stroke = stroke.reshape([n_stroke, n_stk_pnt, 2])

    stroke = [stroke[i] for i in range(stroke.shape[0])]

    stroke = du.order_strokes(stroke)
    stroke = np.vstack(stroke)

    # 获取点的数量
    n = stroke.shape[0]

    # 创建颜色映射，使用索引作为颜色值
    colors = np.linspace(0, 1, n)

    # 绘制笔划并为每个点上色
    plt.figure(figsize=(6, 4))
    scatter = plt.scatter(stroke[:, 0], -stroke[:, 1], c=colors, cmap='viridis', s=50)

    # 可选：连接相邻点形成笔划路径
    plt.plot(stroke[:, 0], -stroke[:, 1], color='gray', linestyle='--', alpha=0.5)

    # 添加颜色条以显示索引颜色映射
    plt.colorbar(scatter, label='Index')

    # 设置图形显示
    plt.title("Stroke Colored by Index")
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def vis_unified_sketch_data(sketch_data, n_stroke=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, show_dot=False, title=None):
    """
    显示笔划与笔划点归一化后的草图
    """
    # 2D coordinates
    coordinates = sketch_data[:, :2]

    # sketch mass move to (0, 0), x y scale to [-1, 1]
    coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
    dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
    coordinates = coordinates / dist

    coordinates = torch.from_numpy(coordinates)
    coordinates = coordinates.view(n_stroke, n_stk_pnt, 2)

    for i in range(n_stroke):
        plt.plot(coordinates[i, :, 0].numpy(), -coordinates[i, :, 1].numpy())

        if show_dot:
            plt.scatter(coordinates[i, :, 0].numpy(), -coordinates[i, :, 1].numpy())

    # plt.axis('off')
    plt.axis('equal')
    plt.title(title)
    plt.show()


def vis_sketch_list(strokes, show_dot=False, title=None):
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])

        if show_dot:
            plt.scatter(s[:, 0], -s[:, 1])

    # plt.axis('off')
    plt.axis("equal")
    plt.title(title)
    plt.show()


def save_format_sketch(sketch_points, file_path, is_smooth=False, is_near_merge=False, merge_dist=0.05, retreat=(0, 0), linewidth=5):
    """
    保存设定格式的草图
    :param sketch_points: [n_stk, n_stk_pnt, 2]
    :param file_path:
    :param is_smooth: 是否保存光顺后的草图
    :param is_near_merge:
    :param merge_dist: 笔划之间距离小于该值，合并笔划
    :param retreat: 合并之前将每个笔划左右各向内删减的点数
    :return:
    """
    def curve_smooth(x, y):
        tck, u = splprep([x, y], s=0.5)  # s 控制平滑程度
        new_u = np.linspace(0, 1, 100)
        new_x, new_y = splev(new_u, tck)
        return new_x, new_y

    n_stk, n_stk_pnt, channel = sketch_points.size()
    sketch_points = sketch_points.detach().cpu().numpy()

    # 每个笔划左右各向内缩减指定数量的点
    if retreat != (0, 0):
        sketch_points = sketch_points[:, retreat[0]:, :] if retreat[1] == 0 else sketch_points[:, retreat[0]: -retreat[1], :]

    # 将过近的笔划合并
    stroke_list = []
    for i in range(n_stk):
        stroke_list.append(sketch_points[i])

    if is_near_merge:
        stroke_list = du.stroke_merge_until(stroke_list, merge_dist)

    # 绘图
    plt.clf()
    for stk_idx in range(len(stroke_list)):
        c_stk = stroke_list[stk_idx]
        plt.plot(c_stk[:, 0], -c_stk[:, 1], linewidth=linewidth)
        # plt.scatter(s[:, 0], -s[:, 1])

    plt.axis('off')
    plt.savefig(file_path)

    if is_smooth:
        plt.clf()
        for stk_idx in range(n_stk):
            c_stk = sketch_points[stk_idx, :, :]
            fit_x, fit_y = curve_smooth(c_stk[:, 0], c_stk[:, 1])
            plt.plot(fit_x, -fit_y, linewidth=linewidth)
            # plt.scatter(s[:, 0], -s[:, 1])

        plt.axis('off')
        ahead, ext = os.path.splitext(file_path)
        plt.savefig(ahead + 'smooth' + ext)


def save_format_sketch_test(sketch_points, file_path, z_thres=0.0):
    """
    保存设定格式的草图
    :param sketch_points: [n_stk, n_stk_pnt, 3]
    :param file_path:
    :param z_thres: z 位置大于该值才判定为有效点
    :return:
    """

    n_stk = sketch_points.size(0)

    # -> [n_stk, n_stk_pnt, channel]
    sketch_points = sketch_points.detach().cpu().numpy()

    plt.clf()
    for stk_idx in range(n_stk):
        c_stk = sketch_points[stk_idx]  # -> [n_stk_pnt, channel]

        # 去掉无效点
        c_stk = c_stk[c_stk[:, 2] >= z_thres]

        plt.plot(c_stk[:, 0], -c_stk[:, 1])

    plt.axis('off')
    plt.savefig(file_path)


def vis_false_log(log_root: str) -> None:
    # 读取每行
    with open(log_root, 'r') as f:
        for c_line in f.readlines():
            c_line = c_line.strip()
            c_file_show = c_line.replace('/opt/data/private/data_set', 'D:/document/DeepLearning/DataSet')
            print(c_line.split('/')[-2])
            print(c_file_show)
            vis_sketch_unified(c_file_show)


def vis_seg_imgs(npz_root=r'D:\document\DeepLearning\DataSet\sketch_seg\SketchSeg-150K'):
    def canvas_size(sketch, padding: int = 30):
        """
        :param sketch: n*3 or n*4
        :param padding: white padding, only make impact on visualize.
        :return: int list,[x, y, h, w], [startX, startY, canvasH, canvasY]
        """
        # get canvas size
        x_point = np.array([0])
        y_point = np.array([0])
        xmin_xmax = np.array([0, 0])
        ymin_ymax = np.array([0, 0])
        for stroke in sketch:
            delta_x = stroke[0]
            delta_y = stroke[1]
            if x_point + delta_x > xmin_xmax[1]:
                xmin_xmax[1] = x_point + delta_x
            elif x_point + delta_x < xmin_xmax[0]:
                xmin_xmax[0] = x_point + delta_x
            if y_point + delta_y > ymin_ymax[1]:
                ymin_ymax[1] = y_point + delta_y
            elif y_point + delta_y < ymin_ymax[0]:
                ymin_ymax[0] = y_point + delta_y
            x_point += delta_x
            y_point += delta_y

        # padding
        assert padding >= 0 and isinstance(padding, int)
        xmin_xmax += np.array([-padding, +padding])  # padding
        ymin_ymax += np.array([-padding, +padding])

        w = xmin_xmax[1] - xmin_xmax[0]
        h = ymin_ymax[1] - ymin_ymax[0]
        start_x = np.abs(xmin_xmax[0])
        start_y = np.abs(ymin_ymax[0])
        # return the copy of sketch. you may use it.
        return [int(start_x), int(start_y), int(h), int(w)], sketch[:]

    def draw_sketch(sketch, window_name="sketch_visualize", padding=30,
                    thickness=2, random_color=True, draw_time=1, drawing=True):
        """
        Include drawing.
        Drawing under the guidance of positions and canvas's size given by canvas_size
        :param sketch: (n, 3) or (n, 4)
        :param window_name:
        :param padding:
        :param thickness:
        :param random_color:
        :param draw_time:
        :param drawing:
        :return: None
        """

        [start_x, start_y, h, w], sketch = canvas_size(sketch=sketch, padding=padding)
        canvas = np.ones((h, w, 3), dtype='uint8') * 255
        if random_color:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            color = (0, 0, 0)
        pen_now = np.array([start_x, start_y])
        first_zero = False
        for stroke in sketch:
            delta_x_y = stroke[0:0 + 2]
            state = stroke[2]
            if first_zero:  # the first 0 in a complete stroke
                pen_now += delta_x_y
                first_zero = False
                continue
            cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color, thickness=thickness)
            if int(state) != 0:  # next stroke
                first_zero = True
                if random_color:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                else:
                    color = (0, 0, 0)
            pen_now += delta_x_y
            if drawing:
                cv2.imshow(window_name, canvas)
                key = cv2.waitKeyEx(draw_time)
                if key == 27:  # esc
                    cv2.destroyAllWindows()
                    exit(0)

        if drawing:
            key = cv2.waitKeyEx()
            if key == 27:  # esc
                cv2.destroyAllWindows()
                exit(0)
        # cv2.imwrite("./visualize.png", canvas)
        return canvas

    count = 0
    npz_all = du.get_allfiles(npz_root, 'npz')

    for index, fileName in enumerate(npz_all):
        print(f"|{index}|{fileName}|{fileName.split('_')[-1].split('.')[0]}|")
        # choose latin1 encoding because we make this dataset by python2.
        sketches = np.load(fileName, encoding="latin1", allow_pickle=True)
        # randomly choose one sketch in one .npz .
        for key in list(sketches.keys()):  # key 只有 arr_0
            # print(f"This key is {key}")
            # print(len(sketches[key]))
            count += len(sketches[key])
            number = random.randint(0, len(sketches[key]))
            sample_sketch: np.ndarray = sketches[key][number]
            # In this part.
            # remove the first line. because in this visualize code, we do not need absolute start-up position.
            # we get the start position by ourselves in func canvas_size().

            # ** In fold ./augm, please comment the under line.
            # because in augm dataset, the first line is not Absolute position.
            sample_sketch[0][0:3] = np.array([0, 0, 0], dtype=sample_sketch.dtype)

            # in cv2, data type INT is allowed.
            # if dataset is normalized, you can make sample_sketch larger.
            # if you run this code in a non-desktop server, drawing=False is necessary.
            sample_sketch = (sample_sketch * 1).astype("int")
            print(sample_sketch)
            draw_sketch(sample_sketch, drawing=True)
    print(count)


def test_vis_sketch_orig(root, pen_up=global_defs.pen_up, pen_down=global_defs.pen_down, show_dot=False, show_axis=False):
    """
    显示原始采集的机械草图
    存储的每行应该为： [x, y, state]
    :param root:
    :param pen_up: 抬笔指令
    :param pen_down: 落笔指令
    :param show_dot:
    :param show_axis:
    :return:
    """
    # -> [n, 4] col: 0 -> x, 1 -> y, 2 -> pen state (17: drawing, 16: stroke end), 3 -> None
    sketch_data = np.loadtxt(root, delimiter=',')

    # 2D coordinates
    coordinates = sketch_data[:, :2]

    # sketch mass move to (0, 0), x y scale to [-1, 1]
    coordinates = coordinates - np.expand_dims(np.mean(coordinates, axis=0), 0)  # 实测是否加expand_dims效果一样
    dist = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)), 0)
    coordinates = coordinates / dist

    sketch_data[:, :2] = coordinates

    # 最后一行最后一个数改为17，防止出现空数组
    sketch_data[-1, 2] = pen_down

    # -------------------------------
    # 去掉点数过少的笔划
    # sketch_data = sp.stk_pnt_num_filter(sketch_data, 4)

    # split all strokes
    strokes = np.split(sketch_data, np.where(sketch_data[:, 2] == pen_up)[0] + 1)

    # 重采样，使得点之间的距离近似相等
    strokes = sp.batched_spline_approx(
        point_list=strokes,
        median_ratio=0.1,
        approx_mode='uni-arclength'
    )

    colors = [[31/255,119/255,180/255], [255/255,127/255,14/255], [44/255,160/255,44/255], [214/255,39/255,40/255], [148/255,103/255,189/255], [140/255,86/255,75/255], [227/255,119/255,194/255]]

    for s, color in zip(strokes, colors):
        # s = s[::105]  # 45
        plt.plot(s[:, 0], -s[:, 1], color=color)

        if show_dot:
            plt.scatter(s[:, 0], -s[:, 1], s=80, color=[31/255,119/255,180/255])

        # if not show_axis:
        #     plt.axis('off')
        # plt.show()

    if not show_axis:
        plt.axis('off')
    plt.show()


def vis_quickdraw(npz_file):
    sketch_all = fr.npz_read(npz_file)[0]
    for c_sketch in sketch_all:
        vis_sketch_data(c_sketch)


def vis_tensor_map(cuda_tensor, title=None, save_root=None, is_show=True):
    m, n = cuda_tensor.size()

    # 1. 将 CUDA Tensor 转换为 CPU 上的 NumPy 数组
    cpu_array = cuda_tensor.cpu().numpy()  # 关键步骤：数据从 GPU → CPU

    # 2. 绘制矩阵热力图
    plt.figure(figsize=(8, 4))  # 设置图像尺寸

    # 绘制热力图，cmap 指定颜色映射（如 'viridis'、'coolwarm' 等）
    plt.imshow(cpu_array, cmap='viridis', interpolation='nearest', aspect='auto')

    # 3. 自定义图像样式
    plt.title(title)
    plt.xlabel("Columns", fontsize=12)
    plt.ylabel("Rows", fontsize=12)
    plt.xticks(range(n))
    plt.yticks(range(m))
    plt.colorbar()

    if save_root is not None:
        plt.savefig(save_root)

    if is_show:
        plt.show()

    plt.clf()
    plt.close()

def vis_pil_tensor(cuda_tensor, flip_rgb=False, title=None, save_root=None,
                   is_show=True):
    from torchvision.transforms.functional import to_pil_image
    cpu_tensor = cuda_tensor.cpu()

    if flip_rgb:
        cpu_tensor = cpu_tensor[[2, 1, 0], :, :]  # BGR → RGB

    pil_tensor = to_pil_image(cpu_tensor)
    plt.imshow(pil_tensor)
    plt.title(title)
    plt.axis('off')

    if save_root is not None:
        plt.savefig(save_root)
    if is_show:
        plt.show()

def vis_ret(sketch_filename, ret_file, image_dir, sketch_dir, top_k=100, cols=2):
    # 如果sketch_filename在sketch_dir中不存在
    if not os.path.exists(os.path.join(sketch_dir, sketch_filename)):
        raise FileNotFoundError(f'{sketch_filename} not found in {sketch_dir}')

    # 将草图列表、图片列表和距离列表从.ret文件中读取
    sketches, images, dists = load_ret_file(ret_file)
    # 获取当前sketch在sketches中的索引
    sketch_dix = np.where(sketches == sketch_filename)[0].item()
    # 获取当前sketch相对于images的距离
    sketch_to_images_dist = dists[sketch_dix, :]
    # 获取当前sketch最接近的top_k个图片索引
    top_k_image_indices = np.argsort(sketch_to_images_dist)[:top_k]
    # 获取最接近的top_k个图片名称的列表
    top_k_image_filenames = images[top_k_image_indices].tolist()
    top_k_image_filepaths = [os.path.join(image_dir, filename) for filename in top_k_image_filenames]

    class ImageGallery(QWidget):
        def __init__(self, image_paths, columns):
            super().__init__()
            self.setWindowTitle("Image Gallery with Scroll")
            self.resize(1920, 1080)

            scroll = QScrollArea(self)
            scroll.setWidgetResizable(True)

            widget = QWidget()
            self.grid_layout = QGridLayout()
            widget.setLayout(self.grid_layout)

            scroll.setWidget(widget)

            layout = QVBoxLayout()
            layout.addWidget(scroll)
            self.setLayout(layout)

            self.load_images(image_paths, columns)

        def load_images(self, image_paths, columns):
            row = 0
            col = 0
            for i, img_path in enumerate(image_paths):
                label = QLabel()
                label.setPixmap(QPixmap(img_path).scaled(
                    224, 224, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                label.setAlignment(Qt.AlignLeft | Qt.AlignTop)  # 左上角对齐
                self.grid_layout.addWidget(label, row, col)

                col += 1
                if col >= columns:
                    col = 0
                    row += 1

    app = QApplication(sys.argv)
    gallery = ImageGallery(top_k_image_filepaths, cols)
    gallery.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    """
    sketch_data = np.loadtxt(r"E:\Dataset\Sketchy\sketches_s5\alarm_clock\n02694662_92-2.txt", delimiter=',')
    sketch_data = torch.from_numpy(sketch_data)
    print(sketch_data)
    vis_s5_data(sketch_data, coor_mode="REL")
    """
    """
    sketch_data = s5_read(
        r"E:\Dataset\sketches\ZSE-SBIR\Sketchy_s5\256x256"
        r"\sketch\tx_000000000000_ready\bee\n02206856_55-1.txt")
    vis_s5_data(sketch_data, coor_mode="REL")
    """
    ret_file = r"E:\Code\ContrastiveSketchRetrieval\vis_test\test.ret"
    image_dir = r"E:\Code\ContrastiveSketchRetrieval\vis_test\images"
    sketch_dir = r"E:\Code\ContrastiveSketchRetrieval\vis_test\sketches"
    vis_ret("sketch1.svg", ret_file, image_dir, sketch_dir, top_k=10)


