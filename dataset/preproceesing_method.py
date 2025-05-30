import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def histogram_equal(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    l, a, b = cv2.split(img)

    lab_clahe = cv2.merge((cv2.equalizeHist(l), a, b))
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_YCrCb2BGR)

    return result

def clahe(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    l, a, b = cv2.split(img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_YCrCb2BGR)

    return result

def retinx(img, sigma_list=[15, 80, 250], gain=1.0, offset=0):
    img = img.astype(np.float32) + 1.0
    log_R = np.zeros_like(img)

    for sigma in sigma_list:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        log_R += np.log(img) - np.log(blur + 1.0)

    log_R /= len(sigma_list)

    sum_channels = np.sum(img, axis=2, keepdims=True)
    crf = np.log(img / (sum_channels + 1e-6) + 1.0)


    msrcr = gain * log_R * crf + offset

    msrcr = np.clip(msrcr, 0, None)
    msrcr = cv2.normalize(msrcr, None, 0, 255, cv2.NORM_MINMAX)
    msrcr = np.clip(msrcr, 0, 255).astype(np.uint8)

    return msrcr

def gammacorrection(img, gamma=0.5):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

if __name__ == '__main__':

    image = '/storage/sjpark/vehicle_data/Dataset2/test_image/16_193828_220929_16_193828_220929_02.jpg'
    img = Image.open(image)
    orig_img = img
    img = np.array(img, dtype=np.uint8)
    #
    img_CV = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #
    clahe_result = clahe(img_CV)
    #
    histogram_equal_result = histogram_equal(img_CV)
    #
    retinx_result = retinx(img)
    #
    gamma_reuslt = gammacorrection(img, gamma=0.5)
    # RGB
    # plt.imshow(orig_img)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    # img = img[:,:,0]
    # plt.hist(img.ravel(), bins=256, range=(0, 256), color='gray', density=True)
    # #
    # plt.imshow(cv2.cvtColor(clahe_result, cv2.COLOR_BGR2RGB))
    # clahe_result = cv2.cvtColor(clahe_result, cv2.COLOR_BGR2RGB)
    # img2 = cv2.cvtColor(clahe_result, cv2.COLOR_RGB2YCrCb)
    # img2 = img2[:,:,0]
    # plt.hist(img2.ravel(), bins=256, range=(0, 256), color='gray', density=True)
    # #
    # plt.imshow(cv2.cvtColor(histogram_equal_result, cv2.COLOR_BGR2RGB))
    # histogram_equal_result = cv2.cvtColor(histogram_equal_result, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(histogram_equal_result, cv2.COLOR_RGB2YCrCb)
    # img = img[:,:,0]
    # plt.hist(img.ravel(), bins=256, range=(0, 256), color='gray', density=True)
    #
    plt.imshow(retinx_result)
    retinx_result = cv2.cvtColor(retinx_result, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(retinx_result, cv2.COLOR_RGB2YCrCb)
    img = img[:,:,0]
    plt.hist(img.ravel(), bins=256, range=(0, 256), color='gray', density=True)
    #
    plt.imshow(gamma_reuslt)
    gamma_reuslt = cv2.cvtColor(gamma_reuslt, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(gamma_reuslt, cv2.COLOR_RGB2YCrCb)
    img = img[:,:,0]
    plt.hist(img.ravel(), bins=256, range=(0, 256), color='gray', density=True)
