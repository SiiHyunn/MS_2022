import cv2
import glob
import os
import random
import shutil

all_categories = os.listdir("./data")


def create_train_val_test_folder() :
    os.makedirs("./dataset/train/" , exist_ok=True)
    for category in sorted(all_categories) :
        os.makedirs("./dataset/train/" + category, exist_ok=True)
        all_images = os.listdir("./data/" + category + "/")
        for image in random.sample(all_images, int(0.8 * len(all_images))) :
            shutil.move("./data/" + category + "/" + image,
                        "./dataset/train/" + category + "/")

    os.makedirs("./dataset/val/", exist_ok=True)
    for category in sorted(all_categories) :
        os.makedirs("./dataset/val/" + category, exist_ok=True)
        all_images = os.listdir("./data/" + category + "/")
        for image in random.sample(all_images, int(0.5 * len(all_images))) :
            shutil.move("./data/" + category + "/" + image,
                        "./dataset/val/" + category + "/")

    # 테스트는 옮길 자료만 남았기 때문에 그냥 이동시킨다.
    os.makedirs("./dataset/test/", exist_ok=True)
    for category in sorted(all_categories) :
        os.makedirs("./dataset/test/" + category, exist_ok=True)
        all_images = os.listdir("./data/" + category + "/")
        for image in all_images:
            shutil.move("./data/" + category + "/" + image,
                        "./dataset/test/" + category + "/")

create_train_val_test_folder()