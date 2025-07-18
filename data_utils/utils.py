import os
import re
import time
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import cv2
from torchvision.transforms import InterpolationMode
from pathlib import Path

from data_utils.sketch_file_read import s5_read


def get_all_train_file(args, skim):
    if skim != 'sketch' or skim != 'images':
        NameError(skim + ' not implemented!')

    if args.dataset == 'sketchy_extend':
        if args.test_class == "test_class_sketchy25":
            shot_dir = "zeroshot1"
        elif args.test_class == "test_class_sketchy21":
            shot_dir = "zeroshot0"

        cname_cid = args.data_path + f'/Sketchy/{shot_dir}/cname_cid.txt'
        if skim == 'sketch':
            file_ls_file = args.data_path + f'/Sketchy/{shot_dir}/sketch_tx_000000000000_ready_filelist_train.txt'
        elif skim == 'images':
            file_ls_file = args.data_path + f'/Sketchy/{shot_dir}/all_photo_filelist_train.txt'
        else:
            NameError(skim + ' not implemented!')

    elif args.dataset == 'tu_berlin':
        cname_cid = args.data_path + '/TUBerlin/zeroshot/cname_cid.txt'
        if skim == 'sketch':
            file_ls_file = args.data_path + '/TUBerlin/zeroshot/png_ready_filelist_train.txt'
        elif skim == 'images':
            file_ls_file = args.data_path + '/TUBerlin/zeroshot/ImageResized_ready_filelist_train.txt'
        else:
            NameError(skim + ' not implemented!')

    elif args.dataset == 'Quickdraw':
        cname_cid = args.data_path + '/QuickDraw/zeroshot/cname_cid.txt'
        if skim == 'sketch':
            file_ls_file = args.data_path + '/QuickDraw/zeroshot/sketch_train.txt'
        elif skim == 'images':
            file_ls_file = args.data_path + '/QuickDraw/zeroshot/all_photo_train.txt'
        else:
            NameError(skim + ' not implemented!')

    else:
        NameError(skim + ' not implemented! ')

    with open(file_ls_file, 'r') as fh:
        file_content = fh.readlines()

    # 图片相对路径
    file_ls = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
    # 图片的label,0,1,2...
    labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])

    # 所有的训练类
    with open(cname_cid, 'r') as ci:
        class_and_indx = ci.readlines()
    # 类名
    cname = np.array([' '.join(cc.strip().split()[:-1]) for cc in class_and_indx])

    return file_ls, labels, cname


def get_some_file_iccv(labels, rootpath, class_list, cname, number, file_ls):
    file_list = []
    for i in class_list:
        # 该类的label
        label = np.argwhere(cname == i)[0, 0]
        # 该类的所有样本
        ind = np.argwhere(labels == label)
        ind_rand = np.random.randint(1, len(ind), number)
        ind_ori = [ind[i] for i in ind_rand]
        files = [file_ls[i] for i in ind_ori]
        full_path = np.array([os.path.join(rootpath, f[0]) for f in files])
        file_list.append(full_path)
    return file_list


def get_file_iccv(labels, rootpath, class_name, cname, number, file_ls):
    # 该类的label
    label = np.argwhere(cname == class_name)[0, 0]
    # 该类的所有样本
    ind = np.argwhere(labels == label)
    ind_rand = np.random.randint(1, len(ind), number)
    ind_ori = ind[ind_rand]
    files = file_ls[ind_ori][0][0]
    full_path = os.path.join(rootpath, files)
    return full_path

# 用来读取草图文件
def get_file_iccv_sketch(labels, rootpath, class_name, cname, number, file_ls):
    pass

def get_file_list_iccv(args, rootpath, skim, split):

    if args.dataset == 'sketchy_extend':
        if args.test_class == "test_class_sketchy25":
            shot_dir = "zeroshot1"
        elif args.test_class == "test_class_sketchy21":
            shot_dir = "zeroshot0"
        else:
            NameError("zeroshot is invalid")

        if skim == 'sketch':
            file_ls_file = args.data_path + f'/Sketchy/{shot_dir}/sketch_tx_000000000000_ready_filelist_zero.txt'
        elif skim == 'images':
            file_ls_file = args.data_path + f'/Sketchy/{shot_dir}/all_photo_filelist_zero.txt'

    elif args.dataset == 'tu_berlin':
        if skim == 'sketch':
            file_ls_file = args.data_path + '/TUBerlin/zeroshot/png_ready_filelist_zero.txt'
        elif skim == 'images':
            file_ls_file = args.data_path + '/TUBerlin/zeroshot/ImageResized_ready_filelist_zero.txt'

    elif args.dataset == 'Quickdraw':
        if skim == 'sketch':
            file_ls_file = args.data_path + f'/QuickDraw/zeroshot/sketch_zero.txt'
        elif skim == 'images':
            file_ls_file = args.data_path + f'/QuickDraw/zeroshot/all_photo_zero.txt'

    else:
        NameError(args.dataset + 'is invalid')

    with open(file_ls_file, 'r') as fh:
        file_content = fh.readlines()
    file_ls = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
    labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])
    file_names = np.array([(rootpath + x) for x in file_ls])

    # 对验证的样本数量进行缩减
    # sketch 15229->762 images 17101->1711
    if args.dataset == 'sketchy_extend' and split == 'test' and skim == 'sketch':
        if args.testall:
            index = [i for i in range(0, file_names.shape[0], 1)]  # 15229
        else:
            index = [i for i in range(0, file_names.shape[0], 20)]   # 762
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    if args.dataset == 'sketchy_extend' and split == 'test' and skim == 'images':
        if args.testall:
            index = [i for i in range(0, file_names.shape[0], 1)]  # 17101
        else:
            index = [i for i in range(0, file_names.shape[0], 10)]  # 1711
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    # sketch 2400->800, images 27989->1400
    if args.dataset == "tu_berlin" and skim == "sketch" and split == "test":
        if args.testall:
            index = [i for i in range(0, file_names.shape[0], 1)]  # 2400
        else:
            index = [i for i in range(0, file_names.shape[0], 3)]  # 800
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    if args.dataset == "tu_berlin" and skim == "images" and split == "test":
        if args.testall:
            index = [i for i in range(0, file_names.shape[0], 1)]  # 27989
        else:
            index = [i for i in range(0, file_names.shape[0], 20)]  # 1400
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    # Quickdraw 92291->770, images 54151->1806
    if args.dataset == "Quickdraw" and skim == "sketch" and split == "test":
        if args.testall:
            index = [i for i in range(0, file_names.shape[0], 1)]  # 92291
        else:
            index = [i for i in range(0, file_names.shape[0], 120)]  # 770
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    if args.dataset == "Quickdraw" and skim == "images" and split == "test":
        if args.testall:
            index = [i for i in range(0, file_names.shape[0], 1)]  # 54151
        else:
            index = [i for i in range(0, file_names.shape[0], 30)]  # 1806
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    file_names_cls = labels
    return file_names, file_names_cls



def preprocess(image_path, img_type="im", max_len=1024):
    if img_type == 'im':
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711))
        ])

        return transform(Image.open(image_path).convert("RGB"))
    else:
        sketch_data = s5_read(image_path, max_len)
        return sketch_data


def remove_white_space_image(img_np: np.ndarray, padding: int):
    """
    获取白底图片中, 物体的bbox; 此处白底必须是纯白色.
    其中, 白底有两种表示方法, 分别是 1.0 以及 255; 在开始时进行检查并且匹配
    对最大值为255的图片进行操作.
    三通道的图无法直接使用255进行操作, 为了减小计算, 直接将三通道相加, 值为255*3的pix 认为是白底.
    :param img_np:
    :return:
    """
    # if np.max(img_np) <= 1.0:  # 1.0 <= 1.0 True
    #     img_np = (img_np * 255).astype("uint8")
    # else:
    #     img_np = img_np.astype("uint8")

    h, w, c = img_np.shape
    img_np_single = np.sum(img_np, axis=2)
    Y, X = np.where(img_np_single <= 300)  # max = 300
    ymin, ymax, xmin, xmax = np.min(Y), np.max(Y), np.min(X), np.max(X)
    img_cropped = img_np[max(0, ymin - padding):min(h, ymax + padding), max(0, xmin - padding):min(w, xmax + padding),
                  :]
    return img_cropped


def resize_image_by_ratio(img_np: np.ndarray, size: int):
    """
    按照比例resize
    :param img_np:
    :param size:
    :return:
    """
    # print(len(img_np.shape))
    if len(img_np.shape) == 2:
        h, w = img_np.shape
    elif len(img_np.shape) == 3:
        h, w, _ = img_np.shape
    else:
        assert 0

    ratio = h / w
    if h > w:
        new_img = cv2.resize(img_np, (int(size / ratio), size,))  # resize is w, h  (fx, fy...)
    else:
        new_img = cv2.resize(img_np, (size, int(size * ratio),))
    # new_img[np.where(new_img < 200)] = 0
    return new_img


def make_img_square(img_np: np.ndarray):
    if len(img_np.shape) == 2:
        h, w = img_np.shape
        if h > w:
            delta1 = (h - w) // 2
            delta2 = (h - w) - delta1

            white1 = np.ones((h, delta1)) * np.max(img_np)
            white2 = np.ones((h, delta2)) * np.max(img_np)

            new_img = np.hstack([white1, img_np, white2])
            return new_img
        else:
            delta1 = (w - h) // 2
            delta2 = (w - h) - delta1

            white1 = np.ones((delta1, w)) * np.max(img_np)
            white2 = np.ones((delta2, w)) * np.max(img_np)

            new_img = np.vstack([white1, img_np, white2])
            return new_img
    if len(img_np.shape) == 3:
        h, w, c = img_np.shape
        if h > w:
            delta1 = (h - w) // 2
            delta2 = (h - w) - delta1

            white1 = np.ones((h, delta1, c), dtype=img_np.dtype) * np.max(img_np)
            white2 = np.ones((h, delta2, c), dtype=img_np.dtype) * np.max(img_np)

            new_img = np.hstack([white1, img_np, white2])
            return new_img
        else:
            delta1 = (w - h) // 2
            delta2 = (w - h) - delta1

            white1 = np.ones((delta1, w, c), dtype=img_np.dtype) * np.max(img_np)
            white2 = np.ones((delta2, w, c), dtype=img_np.dtype) * np.max(img_np)

            new_img = np.vstack([white1, img_np, white2])
            return new_img



# 每个label，对应一个数字
def create_dict_texts(texts):
    texts = list(texts)
    dicts = {l: i for i, l in enumerate(texts)}
    return dicts

# 用来修改Sketchy(svg)中的内容
def refactor_sketchy_svg(sketchy_svg_path):
    # "--"合法存在的形式
    def line_legal(detected_line):
        condition1 = detected_line.count("<!--") + detected_line.count("-->") == detected_line.count("--")
        condition2 = detected_line.count("&amp;") + detected_line.count("&lt;") == detected_line.count("&")
        return condition1 and condition2

    try:
        with open(sketchy_svg_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if not lines[-1].strip().endswith('</svg>'):
            lines.append('</svg>\n')
        # 查看是否有特殊字符存在，如果有，则删除
        for line in lines:
            if not line_legal(line):
                print(f"Invalid line found in {sketchy_svg_path}: {line}")
                lines.remove(line)

        with open(sketchy_svg_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error processing {sketchy_svg_path}: {e}")

def refactor_sketchy_svg_batched(sketchy_svg_log):
    try:
        pattern = r"E:\\Dataset\\Sketchy\\sketches_svg\\(?:[^\\\n]+\\)*[^\\\n]+\.svg"
        with open(sketchy_svg_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                matches = re.findall(pattern, line)
                for match in matches:
                    refactor_sketchy_svg(match)
    except Exception as e:
        print(f"Error processing {sketchy_svg_log}: {e}")

def change_index_png_to_txt(file_path):
    # 读取所有行，替换 .png 为 .txt
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 替换后写回
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line.replace('.png', '.txt'))

def remove_non_existing_filename(file_path, refer_root):
    refer_root = Path(refer_root)
    filenames = []
    class_names = [f.name for f in refer_root.iterdir() if f.is_dir()]
    for class_name in class_names:
        folder_name = refer_root / class_name
        for filename in folder_name.iterdir():
            if filename.name.endswith('.txt'):
                filenames.append(filename.name)

    if len(filenames) == len(set(filenames)):
        print("该文件夹下没有同名文件")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    newlines = []
    for line in lines:
        filename = line.split("/")[-1].split(" ")[0]
        if filename not in filenames:
            print(f"{filename} is not in the refer_dir {refer_root}")
        else:
            newlines.append(line)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(newlines)


def load_ret_file(file_path):
    """
    用来读取.ret类型的文件:
    sketches:
        sketch_filename1;
        sketch_filename2;
        ...
        sketch_filenameN;
    images:
        image_filename1;
        image_filename2;
        ...
        image_filenameM;
    dist:
        dist[1, 1], dist[1, 2], ..., dist[1, M];
        .......................................;
        dist[N, 1], dist[N, 2], ..., dist[N, M];

    :param file_path:
    :return: image_list, sketch_list,
    """
    sketches, images, dist = [], [], []

    section = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('images:'):
                section = 'images'
                continue
            elif line.startswith('sketches:'):
                section = 'sketches'
                continue
            elif line.startswith('dist:'):
                section = 'dist'
                continue

            if section == 'images':
                if line.endswith(';'):
                    images.append(line[:-1])
            elif section == 'sketches':
                if line.endswith(';'):
                    sketches.append(line[:-1])
            elif section == 'dist':
                if line.endswith(';'):
                    row = line[:-1].split(',')  # 去除结尾分号，按逗号分隔
                    dist.append([float(x.strip()) for x in row])

    return np.array(sketches), np.array(images), np.array(dist)

def write_ret_file(ret_file, sketches, images, dist):
    with open(ret_file, 'w', encoding='utf-8') as f:
        f.write("sketches:\n")
        # 草图信息写入
        for sketch in sketches:
            f.write('\t' + sketch + '\n')
        f.write("images:\n")
        # 图片信息写入
        for image in images:
            f.write('\t' + image + '\n')

        f.write("dist:\n")
        # 距离矩阵信息写入
        for row in dist:
            f.write('\t' + ','.join([str(x) for x in row]) + '\n')




if __name__ == '__main__':
    """
    refactor_sketchy_svg_batched(r'E:\Code\ContrastiveSketchRetrieval\data_utils\logs\sketch_conversion_errors.log')
    """
    """
    change_index_png_to_txt(
        r"E:\Dataset\sketches\ZSE-SBIR\Sketchy_s5\zeroshot0\sketch_tx_000000000000_ready_filelist_train.txt")
    change_index_png_to_txt(
        r"E:\Dataset\sketches\ZSE-SBIR\Sketchy_s5\zeroshot0\sketch_tx_000000000000_ready_filelist_zero.txt")
    change_index_png_to_txt(
        r"E:\Dataset\sketches\ZSE-SBIR\Sketchy_s5\zeroshot1\sketch_tx_000000000000_ready_filelist_train.txt")
    change_index_png_to_txt(
        r"E:\Dataset\sketches\ZSE-SBIR\Sketchy_s5\zeroshot1\sketch_tx_000000000000_ready_filelist_zero.txt")
    """
    """ remove_non_existing_filename(
                r"E:\Dataset\sketches\ZSE-SBIR\Sketchy_s5\Sketchy\zeroshot0"
                r"\sketch_tx_000000000000_ready_filelist_zero.txt",
                r"E:\Dataset\sketches\ZSE-SBIR\Sketchy_s5\Sketchy\256x256\sketch\tx_000000000000_ready")

            remove_non_existing_filename(
                r"E:\Dataset\sketches\ZSE-SBIR\Sketchy_s5\Sketchy\zeroshot0"
                r"\sketch_tx_000000000000_ready_filelist_train.txt",
                r"E:\Dataset\sketches\ZSE-SBIR\Sketchy_s5\Sketchy\256x256\sketch\tx_000000000000_ready")

            remove_non_existing_filename(
                r"E:\Dataset\sketches\ZSE-SBIR\Sketchy_s5\Sketchy\zeroshot1"
                r"\sketch_tx_000000000000_ready_filelist_zero.txt",
                r"E:\Dataset\sketches\ZSE-SBIR\Sketchy_s5\Sketchy\256x256\sketch\tx_000000000000_ready")

            remove_non_existing_filename(
                r"E:\Dataset\sketches\ZSE-SBIR\Sketchy_s5\Sketchy\zeroshot1"
                r"\sketch_tx_000000000000_ready_filelist_train.txt",
                r"E:\Dataset\sketches\ZSE-SBIR\Sketchy_s5\Sketchy\256x256\sketch\tx_000000000000_ready")"""

    sketches, images, dists = load_ret_file("../vis_test/test.ret")

    write_ret_file("../vis_test/test1.ret", sketches, images, dists)