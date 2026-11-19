
import PIL
import cv2
import numpy as np
import torch
import torchvision.transforms as tvf

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# Resize the longer image edge while preserving aspect ratio.
def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS 
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC 
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)



def resize_img(img, size, square_ok=True, return_transformation=False):

    # 1. Resize image
    assert size == 224 or size == 512
    # numpy to PIL format
    img = PIL.Image.fromarray(np.uint8(img * 255))
    W1, H1 = img.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
    else:
        # resize long side to 512
        img = _resize_pil_image(img, size)

    # Center-crop to multiples of 16 and adjust the principal point.
    # At size 512, use a square crop when allowed; otherwise use a 4:3 crop.
    W, H = img.size
    cx, cy = W // 2, H // 2
    if size == 224:
        # make w and h equals 224
        half = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not (square_ok) and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    if return_transformation:
        scale_w = W1 / W
        scale_h = H1 / H
        half_crop_w = (W - img.size[0]) / 2
        half_crop_h = (H - img.size[1]) / 2
        return ImgNorm(img), (scale_w, scale_h, half_crop_w, half_crop_h)

    return ImgNorm(img)



class PinholeCamera:

    def __init__(self, target_size_slam, downsample_map, W_original, H_original, calib_parameter,
                 center_force=True):
        self.target_size = target_size_slam
        self.calib = calib_parameter
        self.W_original = W_original
        self.H_original = H_original
        fx, fy, cx, cy = calib_parameter[:4]
        distortion = np.zeros(4)
        if len(calib_parameter) > 4:
            distortion = np.array(calib_parameter[4:])

        # 1. find best intrinsic
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        K_best, _ = cv2.getOptimalNewCameraMatrix(
            K, distortion,
            (W_original, H_original), 0,
            (W_original, H_original),
            centerPrincipalPoint=center_force
        )
        self.mapx , self.mapy = cv2.initUndistortRectifyMap(K, distortion, None,
                                                            K_best, (W_original, H_original), cv2.CV_32FC1)

        # 2. resize image to fit the input size of SLAM model
        _, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_img(
            np.zeros((H_original, W_original, 3)), target_size_slam, return_transformation=True
        )
        K_slam = K_best.copy()
        K_slam[0, 0] = K_slam[0, 0] / scale_w
        K_slam[1, 1] = K_slam[1, 1] / scale_h
        K_slam[0, 2] = K_slam[0, 2] / scale_w - half_crop_w
        K_slam[1, 2] = K_slam[1, 2] / scale_h - half_crop_h
        self.H_slam = _.shape[1]
        self.W_slam = _.shape[2]
        self.K_slam = torch.from_numpy(K_slam).to(torch.float32)

        # 3. find the intrinsic for map
        K_map = K_best.copy()
        K_map[0, 0] = K_map[0, 0] / downsample_map
        K_map[1, 1] = K_map[1, 1] / downsample_map
        K_map[0, 2] = K_map[0, 2] / downsample_map
        K_map[1, 2] = K_map[1, 2] / downsample_map
        self.K_map = torch.from_numpy(K_map).to(torch.float32)
        self.downsample_map = downsample_map
        _ = cv2.resize(
            np.zeros((H_original, W_original, 3)),
            (0, 0),
            fx=1 / self.downsample_map,
            fy=1 / self.downsample_map,
            interpolation=cv2.INTER_AREA,
        )
        self.H_map, self.W_map =  _.shape[0], _.shape[1]

    def to_slam(self, img, device="cuda:0"):
        """
            img HWC [0,255]

            return
            target_image CH'W' [-1, 1]
        """
        target_image = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR).astype(np.float32) / 255
        target_image = resize_img(target_image, self.target_size).to(device)

        return target_image

    def to_map(self, img, device="cuda:0"):
        target_image = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR).astype(np.float32)
        target_image = cv2.resize(
            target_image,
            (0, 0),
            fx=1 / self.downsample_map,
            fy=1 / self.downsample_map,
            interpolation=cv2.INTER_AREA,
        )
        image = torch.from_numpy(target_image).permute(2, 0, 1).to(device).float() / 255.0
        return image
