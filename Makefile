.PHONY: train_debug train_all

train_debug:
	python src/train.py -m arch=debug epochs=3

train_all:
	python src/train.py -m arch=mobilenetv2,efficientnet_b0
