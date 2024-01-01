import numpy as np
import random
import torch
from PIL import ImageFilter, ImageEnhance
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


class RandomContrast(object):  # PIL
    def __init__(self, contrast=0.4):
        assert contrast >= 0.0
        assert contrast <= 1.0
        self.contrast = contrast

    def __call__(self, img):
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

        return img


class RandomSaturation(object):
    def __init__(self, saturation=0.4):
        assert saturation >= 0.0
        assert saturation <= 1.0
        self.saturation = saturation

    def __call__(self, img):
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)

        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_factor)
        return img


class RandomBrightness(object):
    def __init__(self, brightness=0.4):
        assert brightness >= 0.0
        assert brightness <= 1.0
        self.brightness = brightness

    def __call__(self, img):
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        return img


class GaussianBlur(object):
    def __init__(self, sigma=None, aug_blur=False):
        if sigma is None:
            sigma = [.1, 2.]
        self.sigma = sigma
        self.p = 0.5 if aug_blur else 0.

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.rand_brightness = RandomBrightness(brightness)
        self.rand_contrast = RandomContrast(contrast)
        self.rand_saturation = RandomSaturation(saturation)

    def __call__(self, img, targets, seg):
        if random.random() < 0.8:
            func_inds = list(np.random.permutation(3))
            for func_id in func_inds:
                if func_id == 0:
                    img = self.rand_brightness(img)
                elif func_id == 1:
                    img = self.rand_contrast(img)
                elif func_id == 2:
                    img = self.rand_saturation(img)

        return img, targets, seg


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, [0, 0, padw, padh], fill=fill)
    return img


def pad_if_smaller_targets(img, targets, seg, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, [0, 0, padw, padh], fill=fill)

        if seg is not None:
            seg = F.pad(seg, [0, 0, padw, padh], fill=fill)
        if targets is not None:
            for i in range(len(targets)):
                targets[i][:, 0] = targets[i][:, 0] * oh / (padh + oh) # height
                targets[i][:, 1] = targets[i][:, 1] * ow / (padw + ow) # width
    return img, targets, seg


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, targets):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if targets is not None:
                for i in range(len(targets)):
                    targets[i] = F.hflip(targets[i])
        return image, targets


class RandomSelect(object):
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, targets, seg):
        if random.random() < self.p:
            return self.transforms1(img, targets, seg)
        return self.transforms2(img, targets, seg)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, targets, seg):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, targets, seg


class Resize(object):
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, image, targets, seg):
        image = F.resize(image, [self.h, self.w])

        if seg is not None:
            seg = F.resize(seg, [self.h, self.w], interpolation=InterpolationMode.NEAREST)
        return image, targets, seg


class RandomResize(object):
    def __init__(self, sizes, resize_long_side=True):
        self.sizes = sizes
        self.resize_long_side = resize_long_side
        if resize_long_side:
            self.choose_size = max
        else:
            self.choose_size = min

    def __call__(self, image, targets, seg):
        # Return a random integer N such that a <= N <= b. Alias for randrange(a, b+1)
        size = random.choice(self.sizes)

        h, w = image.height, image.width
        ratio = float(size) / self.choose_size(h, w)
        new_h, new_w = round(h * ratio), round(w * ratio)
        image = F.resize(image, [new_h, new_w])

        if seg is not None:
            seg = F.resize(seg, [new_h, new_w], interpolation=InterpolationMode.NEAREST)
        return image, targets, seg


class RandomSizeCrop(object):  # customized for kps detection
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, targets, seg):
        size = random.randint(self.min_size, self.max_size)
        # image = pad_if_smaller(image, size)
        image, targets, seg = pad_if_smaller_targets(image, targets, seg, size)
        ow, oh = image.size
        crop_params = T.RandomCrop.get_params(image, (size, size))
        image = F.crop(image, *crop_params)

        if seg is not None:
            seg = F.crop(seg, *crop_params)

        if targets is not None:
            i_h, i_w, th, tw = crop_params  # i_h for height, i_w for width, use i is bug
            for i, target in enumerate(targets):
                target[:, 0] = target[:, 0] * oh   # height, to image scale
                target[:, 1] = target[:, 1] * ow   # width, to image scale
                target_mask = (target[:, 0] > i_h + 1) & (target[:, 0] < (i_h + th - 1)) & \
                              (target[:, 1] > i_w + 1) & (target[:, 1] < (i_w + tw - 1))
                target = target[target_mask]  # only keep the kps in cropped image
                target[:, 0] = (target[:, 0] - i_h) / th  # height, normalize to 0~1 scale
                target[:, 1] = (target[:, 1] - i_w) / tw
                targets[i] = target
        return image, targets, seg


class ToTensor(object):
    def __call__(self, image, targets, seg):
        image = F.to_tensor(image)
        if targets is not None:
            for i in range(len(targets)):
                targets[i] = torch.as_tensor(targets[i], dtype=torch.float32)   # still 0~1 scale
        if seg is not None:
            seg = torch.from_numpy(np.array(seg)).long().unsqueeze(0)
        return image, targets, seg


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, targets, seg=None):
        for t in self.transforms:
            image, targets, seg = t(image, targets, seg)
        return image, targets, seg
