from lxml import etree
import pathlib
import re

def get_sketchy_info(svg_str):
	"""
	:param svg_str: Sketchy svg文件的具体内容，通过其Annotation中的内容解析
	:return: svg文件的主要信息
	"""
	# 将svg字符串转为xml对象树
	svg_tree = etree.fromstring(svg_str)
	# 获取xml对象中所有的注释信息
	comments = [el for el in svg_tree.iter() if isinstance(el, etree._Comment)]
	# 一张图片中出现一段注释，否则认为出错
	assert len(comments) == 1

	# 获取唯一一段注释内容并转为文本模式
	annotation_text = comments[0].text
	# 清理掉开头的 Sketchy ANNOTATION 和结尾的 END ANNOTATION
	clean_text = re.sub(r'^.*Sketchy ANNOTATION', '', annotation_text)
	# 开始遍历注释信息并使用正则表达式进行匹配转为字段
	fields = re.findall(r'^\s*([\w\s]+):\s*(.*?)\s*$', clean_text, re.MULTILINE)
	# 转成字典
	annotation_dict = {k.strip(): v.strip() for k, v in fields}

	return annotation_dict

def get_sketchy_classmap(dataset_path):
	"""
	:param dataset_path: 有关png格式的dataset,是有关classes上面的目录
	:return: 获取imagenet_id -> class的映射
	"""
	folder_path = pathlib.Path(dataset_path)
	class_map = {}
	# 遍历下属的直接子目录获取类名称
	for class_folder in folder_path.iterdir():
		class_name = class_folder.name
		# 获取该类下面的所有草图
		for sketch_file in class_folder.iterdir():
			image_id = sketch_file.name.split('-')[0]
			# 将image_id -> class对加入映射中
			class_map[image_id] = class_name

	return class_map


if __name__ == "__main__":
	class_map = get_sketchy_classmap(
		r'E:\Dataset\sketches\Sketchy\Sketchy\Sketchy'
		r'\256x256\sketch\tx_000000000000_ready')
