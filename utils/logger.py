import logging
from pathlib import Path
import os


def create_logger(args):
	log_name = args.log_name
	if log_name is None:
		log_name = f"{args.dataset}-{args.test_class}-{args.epoch}"

	log_dir = Path(args.save)
	os.makedirs(log_dir, exist_ok=True)
	log_path = Path(args.save) / log_name

	logger = logging.getLogger(log_name)
	logger.setLevel(logging.INFO)

	formatter = logging.Formatter('%(asctime)s - %(message)s')

	silent = args.silent
	if not silent:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(logging.INFO)
		console_handler.setFormatter(formatter)
		logger.addHandler(console_handler)

	file_handler = logging.FileHandler(log_path, mode='a')
	file_handler.setLevel(logging.INFO)
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)

	return logger

