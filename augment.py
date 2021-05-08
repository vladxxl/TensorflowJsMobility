from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from skimage.transform import resize
from skimage import io

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

np.random.seed(44)
ia.seed(44)

def main():
    for i in range(1, 102):
        draw_single_sequential_images(str(i).zfill(3), "pictures/left", "augment/left-aug")
    for i in range(1, 102):
        draw_single_sequential_images(str(i).zfill(3), "pictures/right", "augment/right-aug")
    for i in range(1, 102):
        draw_single_sequential_images(str(i).zfill(3), "pictures/up", "augment/up-aug")
    for i in range(1, 102):
        draw_single_sequential_images(str(i).zfill(3), "pictures/down", "augment/down-aug")
    for i in range(1, 102):
        draw_single_sequential_images(str(i).zfill(3), "pictures/other", "augment/other-aug")

def draw_single_sequential_images(filename, path, aug_path):
    ia.seed(44)

    # image = misc.imresize(ndimage.imread(path + "/" + filename + ".jpg"), (56, 100))

    image = resize(io.imread(path + "/" + filename + ".jpg"), (320, 480))

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
           iaa.Fliplr(0.5), # horizontally flip 50% of all images
           # crop images by -5% to 10% of their height/width
           sometimes(iaa.CropAndPad(
               percent=(-0.05, 0.1),
               pad_mode=ia.ALL,
               pad_cval=(0, 255)
           )),
           sometimes(iaa.Affine(
               scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
               translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
               rotate=(-5, 5),
               shear=(-5, 5), # shear by -16 to +16 degrees
               order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
               cval=(0, 255), # if mode is constant, use a cval between 0 and 255
               mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
           )),
           iaa.Grayscale(alpha=(0.0, 1.0)),
           # execute 0 to 5 of the following (less important) augmenters per image
           # don't execute all of them, as that would often be way too strong
           iaa.SomeOf((0, 5),
               [
                   iaa.OneOf([
                       iaa.GaussianBlur((0, 1)), # blur images with a sigma between 0 and 3.0
                       iaa.AverageBlur(k=(1, 2)), # blur image using local means with kernel sizes between 2 and 5
                       iaa.MedianBlur(k=(1, 3)), # blur image using local medians with kernel sizes between 2 and 5
                   ]),
                   iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                   iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                   iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images
                   iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                   iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                   # either change the brightness of the whole image (sometimes
                   # per channel) or change the brightness of subareas
                   iaa.OneOf([
                       iaa.Multiply((0.9, 1.1), per_channel=0.5),
                       iaa.BlendAlphaFrequencyNoise(
                           exponent=(-2, 0),
                           foreground=iaa.Multiply((0.9, 1.1), per_channel=True),
                           background=iaa.contrast.LinearContrast((0.9, 1.1))
                       )
                   ]),
                   iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
               ],
               random_order=True
           )
        ],
        random_order=True
    )

    # make sure that values are between 0 and 255, i.e. within 8bit range
    image *= 255 / image.max()
    # cast to 8bit
    image = np.array(image, np.uint8)

    im = np.zeros((1, 320, 480, 3), dtype=np.uint8)
    for c in range(0, 1):
        im[c] = image

    grid = seq(images=im)
    for im in range(len(grid)):
        io.imsave(aug_path + "/" + filename + "_" + str(im) + ".jpg", grid[im])

if __name__ == "__main__":
    main()

