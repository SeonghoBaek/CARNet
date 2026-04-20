# CARNet: Coordinate Attention Residual Block Network for HR Image Segmentation
# Author: Seongho Baek
# e-mail: seongho.baek@sk.com

USE_TF_2 = True

if USE_TF_2 is True:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
else:
    import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.utils import shuffle
import layers
import argparse
import time
import util

# scope
SEGMENT_Encoder_Init_scope = 'input_preprocess'
SEGMENT_Encoder_scope = 'segment_encoder'
SEGMENT_Decoder_scope = 'segment_decoder'
SEGMENT_Decoder_Out_scope = 'final_map'
SEGMENT_Refinement_scope = 'segment_refinement'


def load_images(file_name_list, base_dir=None, noise_mask=None, cutout=False, cutout_mask=None,
                add_eps=False, rotate=False, flip=False, gray_scale=False, shift=False, contrast=False,
                color_shift=False, blur=False, use_min_max_norm=False):
    try:
        images = []
        gt_images = []
        seg_images = []

        for file_name in file_name_list:
            fullname = file_name
            if base_dir is not None:
                fullname = os.path.join(base_dir, file_name).replace("\\", "/")
            img = cv2.imread(fullname)

            if img is None:
                print('Load failed: ' + fullname)
                return None

            h, w, c = img.shape

            if contrast is True:
                if np.random.randint(0, 10) < 5:
                    img = cv2.resize(img, dsize=(input_width // 2, input_height // 2), interpolation=cv2.INTER_AREA)

            '''    
            if h < 3000: # DSP Only
                target_width = np.int32(input_width * extend_ratio)
                target_height = np.int32(input_height * extend_ratio)
                img = cv2.resize(img, dsize=(target_width, target_height), interpolation=cv2.INTER_AREA)
                center_x = target_width // 2
                center_y = target_height // 2
                img = img[center_x - (input_width // 2): center_x + (input_width // 2), center_y - (input_height // 2): center_y + (input_height // 2)]
            else:
                img = cv2.resize(img, dsize=(input_width, input_height), interpolation=cv2.INTER_AREA)
            '''
            img = cv2.resize(img, dsize=(input_width, input_height), interpolation=cv2.INTER_AREA)
            img = np.float32(img)

            if gray_scale is True:
                min_x = np.min(img, keepdims=True)
                max_x = np.max(img, keepdims=True)
            else:
                min_x = np.min(img, axis=(0, 1), keepdims=True)
                max_x = np.max(img, axis=(0, 1), keepdims=True)

            if gray_scale is True:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if img is not None:
                img = np.array(img) * 1.0

                if rotate is True:
                    rot = np.random.randint(-45, 45)
                    if np.random.randint(0, 10) < 5:
                        img = util.rotate_image(img, rot)

                if flip is True:
                    if np.random.rand() < 0.75:
                        code = np.random.randint(low=-1, high=2)
                        img = cv2.flip(img, code)

                if shift is True:
                    x_shift = np.random.randint(-25, 25)
                    y_shift = np.random.randint(-25, 25)
                    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
                    img = cv2.warpAffine(img, M, (input_width, input_height))

                gt_img = img.copy()
                gt_img = np.array(gt_img)

                if gray_scale is True:
                    img = np.expand_dims(img, axis=-1)
                    gt_img = np.expand_dims(gt_img, axis=-1)
                    blur = False
                elif add_eps is True:
                    # img = img + np.random.uniform(low=-2.55, high=2.55, size=img.shape)
                    e = np.random.randint(low=-2, high=3)
                    channel = np.random.randint(0, 3)
                    img[:, :, channel] = img[:, :, channel] + e

                cut_mask = np.zeros_like(img)
                seg_img = cut_mask

                if cutout is True:
                    # square cut out
                    co_w = np.random.randint(low=1, high=input_width // 2)
                    co_h = np.random.randint(low=1, high=input_height // 2)
                    num_cutout = np.random.randint(low=5, high=20)

                    if np.random.randint(low=0, high=10) < 5:
                        if co_w <= co_h:
                            co_w = np.random.randint(low=1, high=5)
                        else:
                            co_h = np.random.randint(low=1, high=5)
                        noise_mask = None

                    cut_offset = input_width // 20

                    for _ in range(num_cutout):
                        r_x = np.random.randint(low=55, high=input_width - co_w - cut_offset)
                        r_y = np.random.randint(low=55, high=input_height - co_h - cut_offset)
                        # img[r_x:r_x + co_w, r_y:r_y + co_h] = 0.0
                        cut_mask[r_x:r_x + co_w, r_y:r_y + co_h, :] = 1.0

                    if noise_mask is not None:
                        # cut_mask = mask_noise + cut_mask
                        cut_mask = noise_mask * cut_mask
                        cut_mask = np.where(cut_mask > 0.5, 1.0, 0)

                    rot = np.random.randint(-90, 90)
                    cut_mask = util.rotate_image(cut_mask, rot)
                    if cutout_mask is not None:
                        cut_mask = cut_mask * cutout_mask

                    # Segmentation Mode
                    seg_img = cut_mask
                    bg_img = (1.0 - seg_img) * img
                    fg_img = seg_img * img
                    alpha = np.random.uniform(low=0.3, high=0.9)

                    if np.random.randint(low=0, high=10) < 5:
                        random_pixels = 2 * (0.1 + np.random.rand())
                        structural_noise = util.rotate_image(img, np.random.randint(0, 360))
                        cut_mask = np.abs(cut_mask * (structural_noise * random_pixels))
                    else:
                        cut_mask = 128.0 * cut_mask

                    cut_mask = np.where(cut_mask > 255, 255, cut_mask)
                    img = bg_img + (alpha * fg_img + (1 - alpha) * cut_mask)
                else:
                    if noise_mask is not None:
                        cut_mask = noise_mask

                        if color_shift is True:
                            seg_img = cut_mask
                            random_pixel = np.random.uniform(low=0, high=255, size=(3))
                            color_mask = [cut_mask, cut_mask, cut_mask]
                            color_mask = np.stack(color_mask, axis=2)
                            color_mask = np.squeeze(color_mask)
                            color_mask = random_pixel * color_mask
                            alpha = np.random.uniform(low=0.3, high=0.7)
                            img = alpha * img + (1 - alpha) * color_mask
                        else:
                            # Segmentation Mode
                            seg_img = cut_mask
                            bg_img = (1.0 - seg_img) * img
                            fg_img = seg_img * img
                            alpha = np.random.uniform(low=0.5, high=0.7)

                            random_pixels = 2 * (0.1 + np.random.rand())
                            structural_noise = util.rotate_image(img, np.random.randint(5, 355))
                            # structural_noise = cv2.GaussianBlur(structural_noise, (0, 0), 3)
                            # structural_noise = np.float32(structural_noise)
                            cut_mask = np.abs(cut_mask * (structural_noise * random_pixels))
                            cut_mask = np.abs(np.where(cut_mask > 255, 255, cut_mask))
                            img = bg_img + (alpha * fg_img + (1 - alpha) * cut_mask)
                            # img = (1.0 - cut_mask) * img

                seg_img = np.average(seg_img, axis=-1)
                seg_img = np.expand_dims(seg_img, axis=-1)
                seg_images.append(seg_img)

                if blur is True:
                    img_blur = cv2.GaussianBlur(img, (0, 0), 1.5)
                    img = np.float32(img_blur)

                if use_min_max_norm is True:
                    img = np.where(img > max_x, max_x, img)
                    img = np.where(img < min_x, min_x, img)
                    n_img = (img - min_x) / (max_x - min_x)
                    n_gt_img = (gt_img - min_x) / (max_x - min_x)
                else:
                    n_img = (img * 1.0) / 255.0
                    n_gt_img = (gt_img * 1.0) / 255.0
                    n_img = np.where(n_img > 1.0, 1.0, n_img)
                    n_img = np.where(n_img < 0, 0, n_img)

                if use_domain_std is True:
                    if gray_scale is False:
                        # Wafer Image
                        n_mean = np.array([0.487, 0.511, 0.533])
                        n_std = np.array([0.19, 0.200, 0.209])
                        # Real World Image mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
                        n_img = (n_img - n_mean) / n_std
                        n_gt_img = (n_gt_img - n_mean) / n_std

                images.append(n_img)
                gt_images.append(n_gt_img)

    except cv2.error as e:
        print(e)
        return None

    return np.array(images), np.array(gt_images), np.array(seg_images)


def get_color_loss(img1, img2):
    img1_hsv = tf.image.rgb_to_hsv(img1)
    img2_hsv = tf.image.rgb_to_hsv(img2)
    img2_h, _, _ = tf.split(img2_hsv, 3, axis=-1)
    img1_h, _, _ = tf.split(img1_hsv, 3, axis=-1)
    # print('hsv : ' + str(img1_hsv.get_shape().as_list()))
    diff1 = tf.abs(img2_h - img1_h)
    diff2 = 180 - diff1
    h_mean_loss = tf.reduce_mean(tf.minimum(diff1, diff2))

    return h_mean_loss


def get_gradient_loss(img1, img2):
    # Laplacian second derivation
    image_a = img1  # tf.expand_dims(img1, axis=0)
    image_b = img2  # tf.expand_dims(img2, axis=0)

    dx_a, dy_a = tf.image.image_gradients(image_a)
    dx_b, dy_b = tf.image.image_gradients(image_b)

    # d2x_ax, d2y_ax = tf.image.image_gradients(dx_a)
    # d2x_bx, d2y_bx = tf.image.image_gradients(dx_b)
    # d2x_ay, d2y_ay = tf.image.image_gradients(dy_a)
    # d2x_by, d2y_by = tf.image.image_gradients(dy_b)

    # loss1 = tf.reduce_mean(tf.abs(tf.subtract(d2x_ax, d2x_bx))) + tf.reduce_mean(tf.abs(tf.subtract(d2y_ax, d2y_bx)))
    # loss2 = tf.reduce_mean(tf.abs(tf.subtract(d2x_ay, d2x_by))) + tf.reduce_mean(tf.abs(tf.subtract(d2y_ay, d2y_by)))

    loss1 = tf.reduce_mean(tf.square(tf.subtract(dx_a, dx_b)))
    loss2 = tf.reduce_mean(tf.square(tf.subtract(dy_a, dy_b)))

    return loss1 + loss2


def get_residual_loss(value, target, type='l1', alpha=1.0):
    if type == 'mse':
        loss = alpha * tf.reduce_mean(tf.square(tf.subtract(target, value)))
    elif type == 'ce':
        eps = 1e-10
        loss = alpha * tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=value, labels=target))
    elif type == 'l1':
        loss = alpha * tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        loss = alpha * tf.reduce_mean(tf.square(tf.subtract(target, value)))
    elif type == 'entropy':
        eps = 1e-10
        loss = alpha * tf.reduce_mean(-1 * value * tf.log(value + eps))
    elif type == 'focal':
        eps = 1e-10
        q = tf.math.maximum(1 - value, eps)
        p = tf.math.maximum(value, eps)
        pos_loss = -(q ** 4) * tf.math.log(p)
        neg_loss = -(p ** 4) * tf.math.log(q)
        loss = alpha * tf.reduce_mean(target * pos_loss + (1 - target) * neg_loss)
    elif type == 'l1_focal':
        eps = 1e-10
        q = tf.math.maximum(1 - value, eps)
        p = tf.math.maximum(value, eps)
        pos_loss = -(q ** 4) * tf.math.log(p)
        neg_loss = -(p ** 4) * tf.math.log(q)
        f_loss = tf.reduce_mean(target * pos_loss + (1 - target) * neg_loss)
        l1_loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
        loss = f_loss + l1_loss
    elif type == 'ssim_focal':
        eps = 1e-10
        q = tf.math.maximum(1 - value, eps)
        p = tf.math.maximum(value, eps)
        pos_loss = -(q ** 2) * tf.math.log(p)
        neg_loss = -(p ** 2) * tf.math.log(q)
        f_loss = tf.reduce_mean(target * pos_loss + (1 - target) * neg_loss)
        loss = tf.reduce_mean(1 - tf.image.ssim_multiscale(value, target, max_val=1.0)) + \
               f_loss
    elif type == 'ssim_l1':
        m = tf.reduce_mean(tf.abs(tf.subtract(value, target)))
        # num_patches = 16
        # img1 = tf.reshape(value, shape=[-1, num_patches, input_height // 4, input_width // 4, num_channel])
        # img2 = tf.reshape(target, shape=[-1, num_patches, input_height // 4, input_width // 4, num_channel])
        loss = alpha * tf.reduce_mean(1 - tf.image.ssim_multiscale(value, target, max_val=1.0)) + (1 - alpha) * m
    elif type == 'dice':
        y = tf.layers.flatten(target)
        y_pred = tf.layers.flatten(value)
        eps = 1e-10
        nominator = 2 * tf.reduce_sum(tf.multiply(y_pred, y))
        denominator = tf.reduce_sum(y_pred) + tf.reduce_sum(y) + eps
        loss = 1 - tf.divide(nominator, denominator)
    elif type == 'ft':
        # Focal Tversky
        y = tf.layers.flatten(target)
        y_pred = tf.layers.flatten(value)

        tp = tf.reduce_sum(y * y_pred)
        fn = tf.reduce_sum(y * (1 - y_pred))
        fp = tf.reduce_sum((1 - y) * y_pred)
        recall = alpha
        precision = 1 - alpha
        eps = 1e-10
        tv = tp / (tp + recall * fn + precision * fp + eps)

        loss = (1 - tv + eps) ** 0.75
    elif type == 'combo':
        y = tf.layers.flatten(target)
        y_pred = tf.layers.flatten(value)
        eps = 1e-10

        tp = tf.reduce_sum(y * y_pred) + eps
        fn = tf.reduce_sum(y * (1 - y_pred)) + eps
        fp = tf.reduce_sum((1 - y) * y_pred) + eps
        recall = 0.7
        precision = 1 - recall
        tv = (tp + 1) / (tp + recall * fn + precision * fp + 1)
        tv = (1 - tv) ** 2

        ce_weight = 0.5
        ce = tf.reduce_mean(
            -ce_weight * y * tf.log(y_pred + eps) - (1 - ce_weight) * (1 - y) * tf.log(1 - y_pred + eps))
        ce_ratio = alpha

        loss = (ce_ratio * ce) + ((1 - ce_ratio) * tv)
    elif type == 'uf':
        # Unified Focal Loss
        eps = 1e-10
        y = tf.layers.flatten(target)
        y_pred = tf.layers.flatten(value)
        sigma = alpha
        gamma = 0.6

        gamma1 = 1 - gamma
        gamma2 = gamma

        q = tf.math.maximum(1 - y_pred, eps)
        p = tf.math.maximum(y_pred, eps)
        pos_loss = -(q ** gamma1) * tf.math.log(p)
        neg_loss = -(p ** gamma1) * tf.math.log(q)
        f_loss = sigma * tf.reduce_mean(y * pos_loss + (1 - y) * neg_loss)

        tp = tf.reduce_sum(y * y_pred)
        fn = tf.reduce_sum(y * (1 - y_pred))
        fp = tf.reduce_sum((1 - y) * y_pred)
        recall = sigma
        precision = 1 - sigma
        eps = 1e-10
        ft_index = tp / (tp + recall * fn + precision * fp + eps)
        ft_loss = (1 - ft_index) ** gamma2

        w = 0.5
        loss = w * f_loss + (1 - w) * ft_loss
    else:
        loss = 0.0

    return loss


def segment_encoder_init(in_tensor, activation=tf.nn.relu, norm='instance', scope='segment_encoder_init',
                         b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print('Segment encoder input tensor: ' + str(in_tensor.get_shape().as_list()))
        batch_mean, batch_var = tf.nn.moments(in_tensor, [1, 2], keep_dims=True)
        batch_std = tf.sqrt(batch_var)
        in_tensor_z_std = (in_tensor - batch_mean) / batch_std
        
        l = in_tensor_z_std
        _, _, _, c = l.get_shape().as_list()
      
        print('Segment encoder init 1024 expansion: ' + str(l.get_shape().as_list()))
        l = layers.conv(l, scope='init_1024', filter_dims=[7, 7, 4 * segment_unit_block_depth],
                        stride_dims=[2, 2], non_linear_fn=activation)
        print('Segment encoder init 512 transform: ' + str(l.get_shape().as_list()))
        l = layers.conv(l, scope='init_512_squeeze', filter_dims=[3, 3, segment_unit_block_depth],
                        stride_dims=[1, 1], non_linear_fn=activation)
        print('Segment encoder init 512 squeeze: ' + str(l.get_shape().as_list()))
        l = layers.conv(l, scope='init_512_base', filter_dims=[3, 3, segment_unit_block_depth],
                        stride_dims=[1, 1], non_linear_fn=None)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='init_norm_512')
        l = activation(l)
        print('Segment encoder init base latent: ' + str(l.get_shape().as_list()))

        return l


def segment_encoder(in_tensor, activation=tf.nn.relu, norm='instance', scope='segment_encoder', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print('Segment encoder input: ' + str(in_tensor.get_shape().as_list()))
        block_depth = segment_unit_block_depth
        l = in_tensor
        lateral_layers = []
        feature_layers = []

        print(' Add Lateral: ' + str(l.get_shape().as_list()))
        lateral_layers.append(l)
        print(' Add skip layer: ' + str(l.get_shape().as_list()))
        feature_layers.append(l)

        for i in range(segmentation_downsample_num):
            block_depth = block_depth * 2
            l = layers.conv(l, scope='downsapmple_' + str(i), filter_dims=[5, 5, 2 * block_depth], stride_dims=[2, 2],
                            non_linear_fn=activation)
            l = layers.conv(l, scope='downsample_squeeze_' + str(i), filter_dims=[3, 3, block_depth],
                            stride_dims=[1, 1], non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='downsample_feature_norm_' + str(i))
            l = activation(l)

            print(' Add Lateral: ' + str(l.get_shape().as_list()))
            lateral_layers.append(l)

        for n_loop in range(num_shuffle_car):
            for i in range(len(lateral_layers)):
                # Cross Stage Partial Network
                lateral_head1, lateral_head2 = tf.split(lateral_layers[i], 2, axis=-1)
                _, h, w, c = lateral_head2.get_shape().as_list()

                for num_rblock in range(num_car):
                    print(' CAR Residual Layer: ' + str(lateral_layers[i].get_shape().as_list()))
                    lateral_head2 = layers.add_residual_block(lateral_head2, filter_dims=[3, 3, c], norm=norm,
                                                              b_train=b_train, act_func=activation,
                                                              scope='loop_sqblock_' + str(n_loop) + str(i) + str(
                                                                  num_rblock))
                lateral_head2 = layers.ca_block(lateral_head2, act_func=activation, scope='car_' + str(n_loop) + str(i))

                lateral_layers[i] = tf.concat([lateral_head1, lateral_head2], axis=-1)

            print('CAR Shuffling')
            fused_layers = []

            for i in range(len(lateral_layers)):
                _, l_h, l_w, l_c = lateral_layers[i].get_shape().as_list()
                mixed_layer = lateral_layers[i]
                print(' Shuffle to: ' + str(lateral_layers[i].get_shape().as_list()))

                for j in range(len(lateral_layers)):
                    if i != j:
                        guest_lat = lateral_layers[j]
                        _, h, w, c = guest_lat.get_shape().as_list()

                        if l_h > h:
                            print('  Upsample ' + str(guest_lat.get_shape().as_list()))
                            # Upsample
                            guest_lat = layers.conv(guest_lat, scope='shuffling_' + str(n_loop) + str(i) + str(j),
                                                    filter_dims=[3, 3, l_c],
                                                    stride_dims=[1, 1], non_linear_fn=None)
                            guest_lat = layers.conv_normalize(guest_lat, norm=norm, b_train=b_train,
                                                              scope='shuffling_norm' + str(n_loop) + str(i) + str(j))
                            guest_lat = activation(guest_lat) 
                            guest_lat = tf.image.resize_images(guest_lat, size=[l_h, l_w])
                        elif l_h < h:
                            # Downsample
                            print('  Downsample ' + str(guest_lat.get_shape().as_list()))
                            ratio = h // l_h
                            num_downsample = int(np.log2(ratio))

                            for k in range(num_downsample):
                                guest_lat = layers.conv(guest_lat,
                                                        scope='shuffling_dn_' + str(n_loop) + str(i) + str(j) + str(k),
                                                        filter_dims=[5, 5, l_c], stride_dims=[2, 2], non_linear_fn=None)
                                guest_lat = layers.conv_normalize(guest_lat, norm=norm, b_train=b_train,
                                                                  scope='shuffling_dn_norm' + str(n_loop) + str(i) +
                                                                        str(j) + str(k))
                                guest_lat = activation(guest_lat)

                        mixed_layer = tf.add(mixed_layer, guest_lat)

                fused_layers.append(mixed_layer)

            lateral_layers = fused_layers

        lateral_layers.reverse()
        feature_layers.reverse()

        return lateral_layers, feature_layers


def segment_decoder_out(latent, activation=tf.nn.relu, norm='instance', scope='segment_decoder_out', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        _, h, w, c = latent.get_shape().as_list()
        latent = layers.ca_block(latent, act_func=activation, scope='segment_attention')

        latent = layers.conv(latent, scope='segment_latent_conv', filter_dims=[3, 3, segment_unit_block_depth],
                             stride_dims=[1, 1], non_linear_fn=None)
        latent = layers.conv_normalize(latent, norm=norm, b_train=b_train, scope='segment_latent_norm')
        latent = activation(latent)
        print('Segment latent: ' + str(latent.get_shape().as_list()))

        small_map_latent = layers.conv(latent, scope='segment_output_small_base', filter_dims=[3, 3, 8],
                                       stride_dims=[1, 1], non_linear_fn=activation)
        small_map = layers.conv(small_map_latent, scope='segment_output_small', filter_dims=[3, 3, 1],
                                stride_dims=[1, 1], non_linear_fn=tf.nn.sigmoid)

        r = input_height // h
        segment_high = layers.conv(latent, scope='upscale_latent_map',
                                   filter_dims=[3, 3, r * r * segment_unit_block_depth],
                                   stride_dims=[1, 1], non_linear_fn=None)
        segment_high = tf.nn.depth_to_space(segment_high, r)
        segment_high = layers.conv_normalize(segment_high, norm=norm, b_train=b_train, scope='segment_high_map_norm')
        segment_high = activation(segment_high)
        print('High map upscale: ' + str(segment_high.get_shape().as_list()))

        segment_map = layers.conv(segment_high, scope='segment_high_out_base', filter_dims=[3, 3, 8],
                                  stride_dims=[1, 1], non_linear_fn=activation)
        segment_map = layers.conv(segment_map, scope='segment_high_out', filter_dims=[3, 3, 1],
                                  stride_dims=[1, 1], non_linear_fn=tf.nn.sigmoid)

        return segment_map, small_map


def segment_decoder(lateral, feature, activation=tf.nn.relu, norm='instance', scope='segment_decoder',
                    b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        _, h, w, c = lateral[-1].get_shape().as_list()

        l0 = layers.conv(lateral[0], scope='lateral0_concat', filter_dims=[1, 1, c],
                         stride_dims=[1, 1], non_linear_fn=None)
        l0 = layers.conv_normalize(l0, norm=norm, b_train=b_train,
                                   scope='lateral_0_concat_norm')
        l0 = activation(l0)
        l0 = tf.image.resize_images(l0, size=[h, w])

        l1 = layers.conv(lateral[1], scope='lateral1_concat', filter_dims=[1, 1, c],
                         stride_dims=[1, 1], non_linear_fn=None)
        l1 = layers.conv_normalize(l1, norm=norm, b_train=b_train,
                                   scope='lateral_1_concat_norm')
        l1 = activation(l1)
        l1 = tf.image.resize_images(l1, size=[h, w])
        l2 = lateral[-1]

        skip_map = feature

        lateral = tf.concat([l0, l1, l2], axis=-1)
        print('Concat all lateral layers: ' + str(lateral.get_shape().as_list()))
        lateral_map = layers.conv(lateral, scope='lateral_concat', filter_dims=[3, 3, c],
                                  stride_dims=[1, 1], non_linear_fn=None)
        lateral_map = layers.conv_normalize(lateral_map, norm=norm, b_train=b_train,
                                            scope='lateral_map_feature_concat_norm')
        lateral_map = activation(lateral_map)
        print('Squeeze lateral layer: ' + str(lateral_map.get_shape().as_list()))

        ada_lat_map = layers.conv(lateral_map, scope='ada_lat_map', filter_dims=[1, 1, c // 2],
                                  stride_dims=[1, 1], non_linear_fn=activation)
        ada_skip_map = layers.conv(skip_map, scope='ada_skip_map', filter_dims=[1, 1, c // 2],
                                   stride_dims=[1, 1], non_linear_fn=activation)
        ada_map = tf.concat([ada_lat_map, ada_skip_map], axis=-1)
        ada_map = layers.conv(ada_map, scope='ada_alpha', filter_dims=[1, 1, 2],
                              stride_dims=[1, 1], non_linear_fn=activation)
        ada_map = tf.nn.softmax(ada_map, axis=-1)
        alpha, beta = tf.split(ada_map, 2, axis=-1)

        lateral_map = alpha * lateral_map + beta * skip_map
        print('Adaptive add lateral and skip layer: ' + str(lateral_map.get_shape().as_list()))

        _, h, w, c = lateral_map.get_shape().as_list()

        k = np.log2(h) // 2
        k = k + ((k + 1) % 2)
        MP = tf.nn.max_pool(lateral_map, ksize=[1, k, k, 1], strides=[1, 1, 1, 1], padding='SAME')

        p1 = layers.avg_pool(MP, filter_dims=[h, w], stride_dims=[h, w])
        p1 = tf.image.resize_images(p1, size=[h, w])

        p2 = layers.avg_pool(MP, filter_dims=[h // 4, w // 4], stride_dims=[h // 4, w // 4])
        p2 = tf.image.resize_images(p2, size=[h, w])

        p3 = layers.avg_pool(MP, filter_dims=[h // 16, w // 16], stride_dims=[h // 16, w // 16])
        p3 = tf.image.resize_images(p3, size=[h, w])

        p4 = layers.avg_pool(MP, filter_dims=[h // 64, w // 64], stride_dims=[h // 64, w // 64])
        p4 = tf.image.resize_images(p4, size=[h, w])

        p5 = layers.avg_pool(MP, filter_dims=[h // 256, w // 256], stride_dims=[h // 256, w // 256])
        p5 = tf.image.resize_images(p5, size=[h, w])

        psp_map = tf.concat([p1, p2, p3, p4, p5], axis=-1)

        print('Concat pyramid pool layers: ' + str(psp_map.get_shape().as_list()))
        psp_lateral = tf.concat([lateral_map, psp_map], axis=-1)
        print('Concat lateral and pyramid pool layers: ' + str(psp_lateral.get_shape().as_list()))
        psp_lateral = layers.conv(psp_lateral, scope='psp_lateral_concat_conv',
                                  filter_dims=[3, 3, segment_unit_block_depth],
                                  stride_dims=[1, 1], non_linear_fn=None)
        psp_lateral = layers.conv_normalize(psp_lateral, norm=norm, scope='psp_lateral_map_norm_concat')
        psp_lateral = activation(psp_lateral)
        print('Decoder latent: ' + str(psp_lateral.get_shape().as_list()))

        return psp_lateral

