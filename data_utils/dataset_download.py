from datasets import load_dataset
from collections import Counter
import os
from pathlib import Path

from options import Option
from dataset_download_utils import *


# 设置代理的端口
os.environ["HTTP_PROXY"] = "http://127.0.0.1:33210"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:33210"

def download_sketchy_dataset(args):
	cache_dir = args.cache_dir + "/Sketchy"
	save_dir = Path(args.cache_dir + "/Sketchy_svg")
	classmap_dir = args.classmap_dir
	sketchy_svg_dataset = load_dataset("kmewhort/sketchy-svgs",
									   cache_dir=cache_dir)
	# 获取imagenet-id -> class的映射
	classmap = get_sketchy_classmap(classmap_dir)
	# 创建用于记录imagenet-id出现多少次的计数器
	image_id_counter = Counter()

	# 遍历数据集的train和test
	for split in sketchy_svg_dataset:
		# 遍历数据集中的每个草图文件
		for sketch_svg in sketchy_svg_dataset[split]['svg']:
			# 获取草图的信息
			sketch_info = get_sketchy_info(sketch_svg)
			# 获取草图的类和imagenet-id
			sketch_id = sketch_info["ImageNet ID"]
			sketch_class = classmap[sketch_id]
			# 计数器加一
			image_id_counter[sketch_id] += 1
			num = image_id_counter[sketch_id]
			save_folder = save_dir/sketch_class
			save_folder.mkdir(parents=True, exist_ok=True)

			with open(save_folder/f"{sketch_id}-{num}.svg",
					  "w", encoding="utf-8") as f:
				f.write(sketch_svg)


if __name__ == "__main__":
	args = Option().parse()
	download_sketchy_dataset(args)