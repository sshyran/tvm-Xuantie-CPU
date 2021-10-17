# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument, too-many-lines, import-outside-toplevel
# pylint: disable=no-else-return, inconsistent-return-statements, no-else-raise
""" Preprocess data helper """
import os
import glob
import functools

import cv2
import numpy as np


class ModelKind(object):
    """Denote the kind of model framework. """

    NOMODEL = -1
    TENSORFLOW = 0
    CAFFE = 1


class PathType(object):
    """Denote the kind of path. """

    NOTEXISTS = 0
    FILE = 1
    DIR = 2


class PreprocessParams(object):
    """ Denote the preprocess parameters for data processing. """

    def __init__(
        self,
        mean=None,
        scale=None,
        resize=None,
        crop=None,
        pixel_format=None,
        disable=False,
        resize_base=None,
    ):
        self.mean = self.parse_mean(mean)
        self.scale = scale
        self.resize = resize
        self.crop = crop
        self.pixel_format = pixel_format
        self.disable = disable
        self.resize_base = resize_base

    def parse_mean(self, mean):
        """Parse the mean value for rgb.

        Parameters
        ----------
        mean : str or list
            The provided mean value

        Returns
        -------
        mean_list : list[int]
            The rgb mean list
        """
        if isinstance(mean, list):
            return mean
        if "," in mean:
            mean = mean.replace(",", " ")
        mean_list = mean.strip().split(" ")
        mean_list = list([float(n) for n in mean_list if n])
        # if len(mean_list) == 1:
        #     mean_list = mean_list * 3
        return mean_list


def check_data(func):
    """The wrapper for checking whether the data field is empty. """

    @functools.wraps(func)
    def wrapper_inner(*args, **kwargs):
        self = args[0]
        if len(self.data) == 0:
            raise ValueError("Please load images first.")
        value = func(*args, **kwargs)
        return value

    return wrapper_inner


class DataLayout(object):
    """Denote the kind of data layout."""

    NCHW = 0
    NHWC = 1
    CHW = 2
    HWC = 3

    @classmethod
    def get_channel_idx(cls, layout_type):
        channel_index = {cls.CHW: 0, cls.HWC: 2, cls.NCHW: 1, cls.NHWC: 3}
        return channel_index[layout_type]

    @classmethod
    def get_type2str(cls, layout_type):
        """ Convert layout type into str """
        if layout_type == cls.NCHW:
            return "NCHW"
        elif layout_type == cls.NHWC:
            return "NHWC"
        elif layout_type == cls.HWC:
            return "HWC"
        elif layout_type == cls.CHW:
            return "CHW"
        else:
            raise ValueError("Unsupport for the layout of data.")

    @classmethod
    def get_str2type(cls, layout_str):
        """ Convert str into layout type """
        if layout_str == "NCHW":
            return cls.NCHW
        elif layout_str == "NHWC":
            return cls.NHWC
        elif layout_str == "HWC":
            return cls.HWC
        elif layout_str == "CHW":
            return cls.CHW
        else:
            raise ValueError("Unsupport for the layout str of data.")

    @classmethod
    def get_data_layout(cls, shape):
        """ Get the data layout"""
        assert shape
        assert len(shape) == 4, "The dim of data should be 4."
        curr_shape = shape[1:]
        if 1 in curr_shape:
            index = curr_shape.index(1)
        elif 3 in curr_shape:
            index = curr_shape.index(3)
        else:
            raise ValueError("The dim of data channel is not 1 or 3.")
        if index == 0:
            return cls.NCHW
        elif index == 2:
            return cls.NHWC
        else:
            raise ValueError("Unable to infer data layout.")


class DataPreprocess(object):
    """Data preprocess helper"""

    def __init__(self):
        self.path = None
        self.data = list()  # list of ndarray
        self.layout = DataLayout.HWC
        self.origin_filenames = list()

    def _check_path(self, path):
        """check the type of path

        Parameters
        ----------
        path : str
            The path to be checked

        Returns
        -------
        type : PathType
            PathType.NOTEXISTS: file or dir not exists
            PathType.FILE: file type
            PathType.DIR: dir type
        """
        path = path.strip()
        if not os.path.exists(path):
            return PathType.NOTEXISTS
        elif os.path.isfile(path):
            return PathType.FILE
        elif os.path.isdir(path):
            return PathType.DIR

    def load_image(self, img_path, gray=False):
        """load single image. Only support for .jpg/.png/.JPEG file.

        Parameters
        ----------
        img_path : str
            The path of image
        gray : bool
            If gray is False, load image with rgb mode;
            If gray is True, load image with gray mode while it is gray image.
        """
        if self._check_path(img_path) in (PathType.NOTEXISTS, PathType.DIR):
            raise FileNotFoundError("File does not exist: {}".format(img_path))
        suffix = img_path.strip().split(".")[-1].lower()
        if suffix not in ("jpg", "png", "jpeg"):
            raise ValueError("Unsuport for image type:{}".format(suffix))
        self.origin_filenames.append(os.path.basename(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = np.array(img, dtype=np.float32)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
            if not gray:
                img = np.tile(img, (1, 1, 3))
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        self.data.append(img)

    def get_data(self):
        """obtain the list of loaded images data."""
        return self.data

    def get_file_list(self, path):
        """Get the file list from path(dir, .txt, .npz or images file)"""
        # support for directory .txt or .npz
        path_type = self._check_path(path)
        img_path_list = list()
        if path_type == PathType.DIR:
            img_path_list += glob.glob(os.path.join(path, "*.jpg"))
            img_path_list += glob.glob(os.path.join(path, "*.png"))
            img_path_list += glob.glob(os.path.join(path, "*.JPEG"))
        elif path_type == PathType.FILE:
            suffix = path.strip().split(".")[-1].lower()
            if suffix == "txt":
                with open(path, "r") as f:
                    for line in f.readlines():
                        line_s = line.strip()
                        if line_s and self._check_path(line_s) == PathType.FILE:
                            img_path_list.append(line_s)
            elif suffix in ("png", "jpg", "jpeg"):
                img_path_list.append(path.strip())
            elif suffix == "npz":
                img_path_list.append(path.strip())
            else:
                raise NotImplementedError(
                    "Unsupport for {}, "
                    "please pass valid files(.txt .npz .jpg .png .JPEG).".format(path)
                )
        else:
            raise FileNotFoundError("Please pass valid path.")
        return img_path_list

    def load_dataset(self, path, gray=False):
        """load multi-images. Only support for directory and .txt

        Parameters
        ----------
        path : str
            The path of images: only support for directory and .txt(include only
                     one image path in a line.)
        gray : bool
            If gray is False, load image with rgb mode;
            If gray is True, load image with gray mode while it is gray image.
        """
        # support for directory .txt or .npz
        img_path_list = self.get_file_list(path)
        if not img_path_list:
            raise ValueError("there is no data...")
        for img_path in img_path_list:
            self.load_image(img_path, gray)

    @check_data
    def sub_mean(self, mean_val=None):
        """Sub mean value for loaded dataset.

        Parameters
        ----------
        mean_val: int or float or list or tuple or numpy.ndarray
            The mean values
        Note
        ----
        If the dim of mean_val more than 1, then its shape either equals to dataset's
        or (H, W, C) or (C, H, W)
        """
        if mean_val is None:
            return
        if isinstance(mean_val, (int, float)):
            mean_val = np.array([mean_val], np.float32)
        elif isinstance(mean_val, (list, tuple)):
            for i in mean_val:
                if not isinstance(i, (int, float)):
                    raise TypeError("mean_val must be float or int.")
            mean_val = np.array(mean_val, dtype=np.float32)
        elif isinstance(mean_val, np.ndarray):
            try:
                mean_val = mean_val.astype(np.float32)
            except:
                raise TypeError("mean_val can't cantain not-float value.")
        else:
            raise TypeError("Unsupport for type:{} in sub_mean".format(type(mean_val)))
        if mean_val.ndim == 1:
            channel_idx = DataLayout.get_channel_idx(self.layout)
            data_channel_num = self.data[0].shape[channel_idx]
            if mean_val.shape[0] == 1:
                pass
            elif mean_val.shape[0] == data_channel_num:
                if self.layout == DataLayout.CHW:
                    mean_val = mean_val[:, np.newaxis, np.newaxis]
                elif self.layout == DataLayout.NCHW:
                    mean_val = mean_val[:, np.newaxis, np.newaxis, np.newaxis]
            else:
                raise ValueError(
                    "Can not broadcast: {}vs{}".format(mean_val.shape, self.data[0].shape)
                )
        else:
            if mean_val.shape != self.data[0].shape:
                if mean_val.ndim == self.data[0].ndim:
                    raise ValueError(
                        "Can not broadcast: {}vs{}".format(mean_val.shape, self.data[0].shape)
                    )
                else:
                    if mean_val.ndim == 3 or mean_val.ndim == 4:
                        if mean_val.shape == self.data[0].shape[1:]:
                            mean_val = np.expand_dims(mean_val, axis=0)
                        else:
                            raise ValueError(
                                "Can not broadcast: {}vs{}".format(
                                    mean_val.shape, self.data[0].shape
                                )
                            )
        for idx, _ in enumerate(self.data):
            self.data[idx] -= mean_val

    @check_data
    def data_scale(self, scale=1.0):
        """Scale the value of dataset by multiplying scale

        Parameters
        ----------
        scale : int or float
            The scale value
        """
        if not isinstance(scale, (int, float)):
            raise TypeError("scale must be float.")
        for d in self.data:
            d *= scale

    @check_data
    def img_resize(self, resize_shape=None):
        """Resize loaded images to specific shape. If resize_shape is int,
            resize shorter side of images to resize_shape and then resize longer
            side of images with the same ratio.

        Parameter
        ---------
        resize_shape : int or list/tuple
            The purpose shape.
        """
        # resize_shape: [height, with]
        if self.layout != DataLayout.HWC:
            raise ValueError("The layout before resizing must be HWC.")
        if resize_shape is None:
            return
        if isinstance(resize_shape, (list, tuple)):
            if len(resize_shape) > 2:
                raise ValueError("Invalid resize_shape.")
            for i in resize_shape:
                if not isinstance(i, int):
                    raise TypeError("resize_shape must be int value")
        elif isinstance(resize_shape, int):
            pass
        else:
            raise ValueError("Invalid resize_shape.")

        channel_dim = self.data[0].shape[2]
        for idx, d in enumerate(self.data):
            h, w, _ = d.shape
            if isinstance(resize_shape, int):
                new_h = h * resize_shape // min(h, w)
                new_w = w * resize_shape // min(h, w)
                resized_data = cv2.resize(d, (new_w, new_h))
            elif isinstance(resize_shape, (list, tuple)):
                new_shape = (resize_shape[1], resize_shape[0])
                resized_data = cv2.resize(d, new_shape)
            if channel_dim == 1:
                resized_data = resized_data[:, :, np.newaxis]
            self.data[idx] = resized_data

    @check_data
    def img_crop(self, crop_shape=None):
        """Crop the images with crop_shape.
            If crop_shape is int, final shape is (crop_shape, crop_shape);

        Parameters
        ----------
        crop_shape : int or list/tuple
            The purpose crop shape
        """
        if crop_shape is None:
            return
        if isinstance(crop_shape, (list, tuple)):
            if len(crop_shape) > 2:
                raise ValueError("Invalid crop_shape.")
            for i in crop_shape:
                if not isinstance(i, int):
                    raise TypeError("crop_shape must be int value")
        elif isinstance(crop_shape, int):
            pass
        else:
            raise ValueError("Invalid crop_shape")
        for idx, d in enumerate(self.data):
            if isinstance(crop_shape, int):
                new_h, new_w = crop_shape, crop_shape
            elif isinstance(crop_shape, (list, tuple)):
                new_h, new_w = crop_shape

            if self.layout == DataLayout.HWC:
                h, w, _ = d.shape
                starth = h // 2 - new_h // 2
                startw = w // 2 - new_w // 2
                self.data[idx] = d[starth : starth + new_h, startw : startw + new_w, :]
            elif self.layout == DataLayout.CHW:
                _, h, w = d.shape
                starth = h // 2 - new_h // 2
                startw = w // 2 - new_w // 2
                self.data[idx] = d[:, starth : starth + new_h, startw : startw + new_w]
            elif self.layout == DataLayout.NCHW:
                _, _, h, w = d.shape
                starth = h // 2 - new_h // 2
                startw = w // 2 - new_w // 2
                self.data[idx] = d[:, :, starth : starth + new_h, startw : startw + new_w]
            elif self.layout == DataLayout.NHWC:
                _, h, w, _ = d.shape
                starth = h // 2 - new_h // 2
                startw = w // 2 - new_w // 2
                self.data[idx] = d[:, starth : starth + new_h, startw : startw + new_w, :]
            else:
                raise ValueError("Error layout in img_crop.")

    @check_data
    def channel_swap(self, new_order=None):
        """Reorder the channel dimention. Mainly convert bgr to rgb

        Parameters
        ----------
        new_order : list or tuple
            The length of new_order must be 3, and is the permutation of (0,1,2)
        """
        if new_order is None:
            return
        if (
            (not isinstance(new_order, (list, tuple)))
            or len(new_order) != 3
            or (0 not in new_order or 1 not in new_order or 2 not in new_order)
        ):
            raise ValueError("new_order should be [r', g' ,b'] belong to {0, 1, 2}")
        channel_idx = DataLayout.get_channel_idx(self.layout)
        channel_num = self.data[0].shape[channel_idx]
        if channel_num == 3:
            for idx, d in enumerate(self.data):
                if self.layout == DataLayout.HWC:
                    self.data[idx] = d[:, :, new_order]
                elif self.layout == DataLayout.CHW:
                    self.data[idx] = d[new_order, :, :]
                elif self.layout == DataLayout.NCHW:
                    self.data[idx] = d[:, new_order, :, :]
                elif self.layout == DataLayout.NHWC:
                    self.data[idx] = d[:, :, :, new_order]

    @check_data
    def data_transpose(self, trans=None):
        """Transpose the loaded images.

        Parameters
        ----------
        trans : list/tuple
            The new order of axis, which number must be the same with
                loaded images.
        """
        if trans is None:
            return
        if not isinstance(trans, (list, tuple)) or len(trans) != self.data[0].ndim:
            raise ValueError("trans should be list or tuple and the length should match data")
        for idx, d in enumerate(self.data):
            self.data[idx] = np.transpose(d, trans)
        layout_str = DataLayout.get_type2str(self.layout)
        layout_list = np.array(list(layout_str))
        layout_list = layout_list[list(trans)]
        layout_list = list(layout_list)
        layout_str = "".join(layout_list)
        self.layout = DataLayout.get_str2type(layout_str)

    @check_data
    def data_expand_dim(self):
        """Expand the dim of loaded images. Normally convert
        HWC(CHW) into NCHW(NCHW)
        """
        if self.layout == DataLayout.HWC:
            self.layout = DataLayout.NHWC
        elif self.layout == DataLayout.CHW:
            self.layout = DataLayout.NCHW
        else:
            raise ValueError("Can't expand dim while the layout is not HWC or CHW.")
        for idx, d in enumerate(self.data):
            self.data[idx] = np.expand_dims(d, axis=0)

    @check_data
    def data_clear(self):
        self.data.clear()
        self.layout = DataLayout.HWC


class DatasetLoader(object):
    """
    load preprocessed dataset by generator
    """

    def __init__(
        self, data_path, input_shape_dict=None, preprocess_params=None, batch_size=1, num_workers=1
    ):
        self.data_path = data_path
        self.input_shape_dict = input_shape_dict
        self.preprocess_params = preprocess_params
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dp = DataPreprocess()
        self.file_path = self.dp.get_file_list(self.data_path)

    def get_data(self):
        """Generator for get preprocessed data"""
        input_name_list = list(self.input_shape_dict.keys())
        input_name_list = list([str(name) for name in input_name_list])
        suffix = self.data_path.split(".")[-1]
        if suffix == "npz":
            npz_data = np.load(self.data_path)
            for i_name in input_name_list:
                if i_name not in npz_data.keys():
                    raise ValueError(
                        "The input data {} is not in the passed .npz file, "
                        "Please rebuild the input file."
                    )
            split_npz_data = {}
            assert self.batch_size == 1, "Only support for batch_size=1 while loading npz data."
            epoch = list(npz_data.values())[0].shape[0]
            for name, value in npz_data:
                split_dataset = np.split(value, epoch, axis=0)
                split_npz_data[name] = split_dataset
            for index in range(epoch):
                tmp_d = {}
                for k, v in split_npz_data.items():
                    tmp_d[k] = v[index]
                yield tmp_d

            self.preprocess_params.disable = True
        else:
            if len(input_name_list) > 1:
                raise ValueError(
                    "The number of model input is more than 1, please "
                    "use .npz file by '--input-dataset'(numpy data format) instead."
                )
            input_shape = list(self.input_shape_dict.values())[0]
            input_shape = list([int(i) for i in input_shape])
            data_layout = DataLayout.get_data_layout(input_shape)
            gray = False
            if 1 in input_shape[1:]:
                gray = True
            data_h_w = tuple()
            if data_layout == DataLayout.NCHW:
                data_h_w = (input_shape[2], input_shape[3])
            else:
                data_h_w = (input_shape[1], input_shape[2])

            self.preprocess_params.crop = list(data_h_w)
            if self.preprocess_params.resize_base:
                self.preprocess_params.resize = [int(self.preprocess_params.resize_base), 0]
            else:
                self.preprocess_params.resize = list(data_h_w)

            for img_file in self.file_path:
                self.dp.load_image(img_file, gray=gray)
                if len(self.dp.get_data()) != self.batch_size:
                    continue
                if self.preprocess_params.resize_base:
                    self.dp.img_resize(resize_shape=int(self.preprocess_params.resize_base))
                    self.dp.img_crop(crop_shape=data_h_w)
                else:
                    self.dp.img_resize(resize_shape=data_h_w)

                self.dp.sub_mean(mean_val=self.preprocess_params.mean)
                self.dp.data_scale(self.preprocess_params.scale)

                if data_layout == DataLayout.NCHW:
                    if self.preprocess_params.pixel_format == "RGB" and input_shape[1] == 3:
                        self.dp.channel_swap((2, 1, 0))
                self.dp.data_expand_dim()
                self.dp.data_transpose((0, 3, 1, 2))
                dataset = self.dp.get_data()
                dataset = np.concatenate(dataset, axis=0)
                yield {input_name_list[0]: dataset}
                self.dp.data_clear()
