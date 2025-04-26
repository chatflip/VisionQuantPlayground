.PHONY: train_debug train_all

train_debug:
	python src/train.py -m epochs=2 arch=mobilenetv2 arch.batch_size=128

train_all:
	python src/train.py -m arch=mobilenetv2,efficientnet_b0