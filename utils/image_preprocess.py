import numpy as np
from skimage.transform import resize
from configs import FLAGS

def upsample(img, img_size_target=FLAGS.img_size_target):
    if img.shape[0] == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)


def downsample(img, img_size_ori=FLAGS.img_size_ori, img_size_target=FLAGS.img_size_target):
    if img.shape[0] == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)


def random_crop_image(image):
    random_array = np.random.randint(0, 26, size=2)
    w = FLAGS.img_size_ori
    h = FLAGS.img_size_ori
    x = random_array[0]
    y = random_array[1]

    image_crop = image[y:h + y, x:w + x, 1]
    return image_crop


# Alternative resizing method by padding to (128,128,1) and reflecting the image to the padded areas
def pad_reflect(img, north=0, south=0, west=0, east=0):
    h = img.shape[0]
    w = img.shape[1]
    new_image = np.zeros((north + h + south, west + w + east))

    # Place the image in new_image
    new_image[north:north + h, west:west + w] = img

    new_image[north:north + h, 0:west] = np.fliplr(img[:, :west])
    new_image[north:north + h, west + w:] = np.fliplr(img[:, w - east:])

    new_image[0:north, :] = np.flipud(new_image[north:2 * north, :])
    new_image[north + h:, :] = np.flipud(new_image[north + h - south:north + h, :])

    return new_image


def unpad_reflect(img, north=0, west=0, height=0, width=0):
    return img[north:north + height, west:west + width]
