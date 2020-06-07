import os
import glob
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import csv

BASE_FILE = ""


def visualize_samples(split, output_dir, n=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_dir = os.path.join(BASE_FILE + f"WIDER_{split}", "images")
    label_file = BASE_FILE + f"wider_face_split/wider_face_{split}_bbx_gt.txt"
    labels = load_labels(label_file, split)

    random.shuffle(labels)
    for i in range(n):
        image = cv2.imread(os.path.join(image_dir, labels[i]["image_path"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for bbox in labels[i]["bboxes"]:
            x1, y1 = bbox["x1"], bbox["y1"]
            x2, y2 = x1 + bbox["w"], y1 + bbox["h"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        plt.imshow(image)
        print("[INFO] Save image at: ", os.path.join(
            output_dir, f"vis-{i}.png"))
        plt.savefig(os.path.join(output_dir, f"vis-{i}.png"))


def load_labels(file, split):
    with open(file) as f:
        content = [line.strip() for line in f.readlines()]

    annotations = []
    for i in range(len(content)):
        if "--" not in content[i]:
            continue
        annotation = {}
        annotation["image_path"] = content[i]
        n_bboxes = int(content[i+1])
        annotation["bboxes"] = []
        for i in range(i+2, i+2+n_bboxes):
            bbox = {}
            data = [int(x) for x in content[i].split()]
            bbox["x1"] = data[0]
            bbox["y1"] = data[1]
            bbox["w"] = data[2]
            bbox["h"] = data[3]
            annotation["bboxes"].append(bbox)
        annotations.append(annotation)
    return annotations


def convert2csv(split, output_dir):

    with open('csv_val.csv', mode='w') as employee_file:
        employee_writer = csv.writer(
            employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        image_dir = os.path.join(f"WIDER_{split}", "images")
        label_file = f"wider_face_split/wider_face_{split}_bbx_gt.txt"
        labels = load_labels(label_file, split)
        random.shuffle(labels)
        # print("[INFO] Len labels: ", len(labels))
        for i in range(len(labels)):
            for bbox in labels[i]["bboxes"]:
                x1, y1 = bbox["x1"], bbox["y1"]
                x2, y2 = x1 + bbox["w"], y1 + bbox["h"]
                employee_writer.writerow(
                    [image_dir + "/" + labels[i]["image_path"], x1, y1, x2, y2, "face"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI cardio pipeline")
    parser.add_argument("--split", type=str, choices=["train", "val"],
                        help="Split")
    parser.add_argument("--output_dir", type=str, default="visualization",
                        help="Output dir")
    args = parser.parse_args()

    # visualize_samples(args.split, args.output_dir)
    convert2csv(args.split, args.output_dir)
