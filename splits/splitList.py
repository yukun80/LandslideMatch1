import random


def split_labeled_file(input_txt, output_labeled_txt, output_unlabeled_txt, num):
    # 读取原始的 labeled.txt 文件内容
    with open(input_txt, "r") as file:
        lines = file.readlines()

    # 随机选择 num 行
    labeled_lines = random.sample(lines, num)

    # 其余行作为 unlabeled_lines
    unlabeled_lines = [line for line in lines if line not in labeled_lines]

    # 将选择的行写入新的 labeled.txt
    with open(output_labeled_txt, "w") as labeled_file:
        labeled_file.write("".join(labeled_lines))

    # 将其余的行写入 unlabeled.txt
    with open(output_unlabeled_txt, "w") as unlabeled_file:
        unlabeled_file.write("".join(unlabeled_lines))


# 使用示例
input_txt = r"D:\_codeProject\_Segmentation\Semi-supervised\UniMatch-fyk\splits\HR-GLDD\1119\labeled.txt"  # 替换为实际的原始 labeled.txt 文件路径
output_labeled_txt = (
    r"D:\_codeProject\_Segmentation\Semi-supervised\UniMatch-fyk\splits\HR-GLDD\600\labeled.txt"  # 替换为实际的输出新的 labeled.txt 文件路径
)
output_unlabeled_txt = (
    r"D:\_codeProject\_Segmentation\Semi-supervised\UniMatch-fyk\splits\HR-GLDD\600\unlabeled.txt"  # 替换为实际的输出 unlabeled.txt 文件路径
)
num = 560  # 替换为需要随机选择的行数

split_labeled_file(input_txt, output_labeled_txt, output_unlabeled_txt, num)
