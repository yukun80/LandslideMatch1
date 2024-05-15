import os


def generate_label_files(image_dir, label_dir, output_train_txt, output_val_txt):
    train_lines = []
    val_lines = []

    for filename in os.listdir(image_dir):
        if filename.startswith("train"):
            base_name = os.path.splitext(filename)[0]
            image_path = os.path.join(os.path.basename(image_dir), f"{base_name}.tif")
            label_path = os.path.join(os.path.basename(label_dir), f"{base_name}.png")
            train_lines.append(f"{image_path} {label_path}")
        elif filename.startswith("test"):
            base_name = os.path.splitext(filename)[0]
            image_path = os.path.join(os.path.basename(image_dir), f"{base_name}.tif")
            label_path = os.path.join(os.path.basename(label_dir), f"{base_name}.png")
            val_lines.append(f"{image_path} {label_path}")

    with open(output_train_txt, "w") as train_file:
        train_file.write("\n".join(train_lines))

    with open(output_val_txt, "w") as val_file:
        val_file.write("\n".join(val_lines))


# 使用示例
image_dir = r"D:\_codeProject\_Segmentation\Semi-supervised\UniMatch-fyk\dataset\HR-GLDD\Images"  # 替换为实际的图像文件夹路径
label_dir = r"D:\_codeProject\_Segmentation\Semi-supervised\UniMatch-fyk\dataset\HR-GLDD\SegmentationClass"  # 替换为实际的标签文件夹路径
output_train_txt = r"D:\_codeProject\_Segmentation\Semi-supervised\UniMatch-fyk\splits\HR-GLDD\1119\labeled.txt"  # 替换为实际的输出train.txt文件路径
output_val_txt = r"D:\_codeProject\_Segmentation\Semi-supervised\UniMatch-fyk\splits\HR-GLDD\val.txt"  # 替换为实际的输出val.txt文件路径

generate_label_files(image_dir, label_dir, output_train_txt, output_val_txt)
