import os
import cv2
import numpy as np


def load_img(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def save_img(path, img):
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def extract_mask_and_boxes(I, T, thresh=0.15):
    """
    I: blended
    T: transmission
    """

    # residual
    R = I - T
    R = np.maximum(R, 0.0)

    # -----------------------------
    # 关键：用“亮度指标”做mask
    # -----------------------------

    # 方法1：RGB最小值（你提的思路）
    score = np.min(R, axis=2)

    # 方法2（更稳定）：平均亮度
    # score = np.mean(R, axis=2)

    mask = (score > thresh).astype(np.uint8) * 255

    # 去噪（很重要）
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 画框
    vis = (I * 255).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:   # 去掉小噪声
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))

        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return mask, vis, R, boxes


if __name__ == "__main__":

    blended_dir = r"E:\ERRNet\real89\blended"
    trans_dir = r"E:\ERRNet\real89\transmission_layer"
    out_dir = r"E:\ERRNet\real89\reflection_mask"

    os.makedirs(out_dir, exist_ok=True)

    for name in os.listdir(blended_dir):

        I_path = os.path.join(blended_dir, name)
        T_path = os.path.join(trans_dir, name)

        if not os.path.exists(T_path):
            continue

        I = load_img(I_path)
        T = load_img(T_path)

        if I.shape != T.shape:
            T = cv2.resize(T, (I.shape[1], I.shape[0]))

        mask, vis, R, boxes = extract_mask_and_boxes(I, T, thresh=0.12)

        base = os.path.join(out_dir, name.split('.')[0])

        cv2.imwrite(base + "_mask.png", mask)
        cv2.imwrite(base + "_box.png", vis)
        save_img(base + "_residual.png", R)

        print(name, "boxes:", len(boxes))