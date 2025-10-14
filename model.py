# CARNet: Coordinate Attention Residual Block Network for HR Image Segmentation
# Author: Seongho Baek
# e-mail: seonghobaek@gmail.com

USE_TF_2 = False

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
RECONSTRUCTION_Decoder_scope = 'reconstruction_decoder'
SEGMENT_Decoder_Out_scope = 'final_map'
SPARTIAL_Memory_scope = 'structure_spatial_mem'


def load_images(file_name_list, base_dir=None, noise_mask=None, cutout=False, cutout_mask=None,
                add_eps=False, rotate=False, rotate_degree=0, flip=False, gray_scale=False, shift=False, contrast=False,
                color_shift=False, extend_ratio=0.0, make_border=False, blur=False):
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

            if extend_ratio > 0:
                target_width = np.int32(w * extend_ratio)
                target_height = np.int32(h * extend_ratio)
                offset = np.abs((target_width - w) // 2)
                img = cv2.resize(img, dsize=(target_width, target_height), interpolation=cv2.INTER_AREA)
                img = img[offset: offset + w, offset: offset + h]

            if contrast is True:
                if np.random.randint(0, 10) < 5:
                    img = cv2.resize(img, dsize=(w // 2, h // 2), interpolation=cv2.INTER_AREA)

            img = cv2.resize(img, dsize=(input_width, input_height), interpolation=cv2.INTER_AREA)

            if make_border is True:
                safe_boundary = input_height // 20
                if gray_scale is True:
                    img = cv2.copyMakeBorder(img, safe_boundary // 2, safe_boundary // 2, safe_boundary // 2,
                                             safe_boundary // 2,
                                             borderType=cv2.BORDER_CONSTANT,
                                             value=0)
                else:
                    border_color = img[input_height // 10, input_width // 10]
                    img = cv2.copyMakeBorder(img, safe_boundary // 2, safe_boundary // 2, safe_boundary // 2,
                                             safe_boundary // 2,
                                             borderType=cv2.BORDER_CONSTANT,
                                             value=(int(border_color[0]), int(border_color[1]), int(border_color[2])))

                img = cv2.resize(img, dsize=(input_width, input_height), interpolation=cv2.INTER_AREA)

            if gray_scale is True:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if img is not None:
                img = np.array(img) * 1.0

                if rotate is True:
                    r_d = rotate_degree
                    l_d = -1 * rotate_degree
                    rot = np.random.randint(l_d, r_d)
                    if np.random.randint(0, 10) < 5:
                        img = util.rotate_image(img, rot)

                if flip is True:
                    if np.random.rand() < 0.75:
                        code = np.random.randint(low=-1, high=2)
                        img = cv2.flip(img, code)

                if shift is True:
                    x_shift = 0
                    y_shift = 0
                    if np.random.randint(0, 10) < 5:
                        x_shift = np.random.randint(-96, 96)
                    else:
                        y_shift = np.random.randint(-96, 96)
                    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
                    img = cv2.warpAffine(img, M, (input_width, input_height))

                gt_img = img.copy()
                gt_img = np.array(gt_img)

                if add_eps is True:
                    img = img + np.random.uniform(low=-2.55, high=2.55, size=img.shape)

                if gray_scale is True:
                    img = np.expand_dims(img, axis=-1)
                    gt_img = np.expand_dims(gt_img, axis=-1)
                    blur = False

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
                            if np.random.randint(0, 10) < 2:
                                img = (1.0 - cut_mask) * img
                                seg_img = cut_mask
                            else:
                                seg_img = cut_mask
                                bg_img = (1.0 - seg_img) * img
                                fg_img = seg_img * img
                                alpha = np.random.uniform(low=0.5, high=0.7)

                                random_pixels = 2 * (0.1 + np.random.rand())
                                structural_noise = util.rotate_image(img, np.random.randint(5, 355))
                                cut_mask = np.abs(cut_mask * (structural_noise * random_pixels))
                                cut_mask = np.abs(np.where(cut_mask > 255, 255, cut_mask))
                                img = bg_img + (alpha * fg_img + (1 - alpha) * cut_mask)

                if blur is True:
                    img_blur = cv2.GaussianBlur(img, (0, 0), 1.5)
                    img = np.float32(img_blur)

                seg_img = np.average(seg_img, axis=-1)
                seg_img = np.expand_dims(seg_img, axis=-1)
                seg_images.append(seg_img)

                n_img = (img * 1.0) / 255.0
                n_gt_img = (gt_img * 1.0) / 255.0

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
        nominator = 2 * tf.reduce_sum(tf.multiply(y_pred, y)) + 1e-6
        denominator = tf.reduce_sum(y_pred) + tf.reduce_sum(y) + 1e-6
        loss = 1 - tf.divide(nominator, denominator)
    elif type == 'ft':
        # Focal Tversky
        y = tf.layers.flatten(target)
        y_pred = tf.layers.flatten(value)

        tp = tf.reduce_sum(y * y_pred) + 1e-6
        fn = tf.reduce_sum(y * (1 - y_pred)) + 1e-6
        fp = tf.reduce_sum((1 - y) * y_pred) + 1e-6
        recall = alpha
        precision = 1 - alpha
        tv = (tp + 1) / (tp + recall * fn + precision * fp + 1)

        loss = (1 - tv) ** 2
    else:
        loss = 0.0

    return loss


def segment_encoder_init(in_tensor, activation=tf.nn.relu, norm='instance', scope='segment_encoder_init',
                         b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print('Segment encoder init: ' + str(in_tensor.get_shape().as_list()))

        b, g, r = tf.split(in_tensor, 3, axis=-1)
        img_rgb = tf.concat([r, g, b], axis=-1)
        in_tensor_hsv = tf.image.rgb_to_hsv(img_rgb)

        in_tensor = tf.concat([in_tensor, in_tensor_hsv], axis=-1)
        l = layers.conv(in_tensor, scope='init_1024', filter_dims=[7, 7, 4 * segment_unit_block_depth],
                        stride_dims=[2, 2], non_linear_fn=activation)
        l = layers.conv(l, scope='init_1024_squeeze1', filter_dims=[3, 3, segment_unit_block_depth],
                        stride_dims=[1, 1], non_linear_fn=activation)
        # l = layers.ca_block(l, act_func=activation)
        l = layers.conv(l, scope='init_512_squeeze2', filter_dims=[3, 3, segment_unit_block_depth],
                        stride_dims=[1, 1], non_linear_fn=None)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='init_norm_512')
        l = activation(l)

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
            # feature_layers.append(l)

        for n_loop in range(num_shuffle_car):
            for i in range(len(lateral_layers)):
                # Cross Stage Partial Network
                lateral_head1, lateral_head2 = tf.split(lateral_layers[i], 2, axis=-1)
                _, h, w, c = lateral_head2.get_shape().as_list()

                for num_rblock in range(num_car):
                    print(' Conv Layer: ' + str(lateral_layers[i].get_shape().as_list()))
                    lateral_head2 = layers.add_residual_block(lateral_head2, filter_dims=[3, 3, c], norm=norm,
                                                              b_train=b_train, act_func=activation,
                                                              scope='loop_sqblock_' + str(n_loop) + str(i) + str(
                                                                  num_rblock))
                lateral_head2 = layers.ca_block(lateral_head2, act_func=activation, scope='car_' + str(n_loop) + str(i))

                lateral_layers[i] = tf.concat([lateral_head1, lateral_head2], axis=-1)
                # lateral_layers[i] = layers.conv(lateral_layers[i], scope='csp_merge_' + str(n_loop) + str(i),
                #                                filter_dims=[1, 1, 2 * c], stride_dims=[1, 1],
                #                                non_linear_fn=activation)

            print('Shuffling...')
            fused_layers = []

            for i in range(len(lateral_layers)):
                _, l_h, l_w, l_c = lateral_layers[i].get_shape().as_list()
                mixed_layer = lateral_layers[i]
                print(' Shuffle to: ' + str(lateral_layers[i].get_shape().as_list()))

                for j in range(len(lateral_layers)):
                    if i != j:
                        l_lat = lateral_layers[j]
                        _, h, w, c = l_lat.get_shape().as_list()

                        if l_h > h:
                            # Resize
                            l_lat = layers.conv(l_lat, scope='shuffling_' + str(n_loop) + str(i) + str(j),
                                                filter_dims=[3, 3, l_c],
                                                stride_dims=[1, 1], non_linear_fn=None)
                            l_lat = layers.conv_normalize(l_lat, norm=norm, b_train=b_train,
                                                          scope='shuffling_norm' + str(n_loop) + str(i) + str(j))
                            l_lat = activation(l_lat)
                            l_lat = tf.image.resize_images(l_lat, size=[l_h, l_w])
                        elif l_h < h:
                            # Downsample
                            ratio = h // l_h
                            num_downsample = int(np.log2(ratio))

                            for k in range(num_downsample):
                                l_lat = layers.conv(l_lat,
                                                    scope='shuffling_dn_' + str(n_loop) + str(i) + str(j) + str(k),
                                                    filter_dims=[5, 5, l_c], stride_dims=[2, 2], non_linear_fn=None)
                                l_lat = layers.conv_normalize(l_lat, norm=norm, b_train=b_train,
                                                              scope='shuffling_dn_norm' + str(n_loop) + str(i) +
                                                                    str(j) + str(k))
                            l_lat = activation(l_lat)

                        mixed_layer = tf.add(mixed_layer, l_lat)

                fused_layers.append(mixed_layer)

                # if n_loop == (num_shuffle_car - 1):
                #    break

            lateral_layers = fused_layers

        lateral_layers.reverse()
        feature_layers.reverse()

        return lateral_layers, feature_layers


def segment_decoder_out(latent, activation=tf.nn.relu, norm='instance', scope='segment_decoder_out', b_train=False,
                        b_smoothing=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print(scope)
        _, h, w, c = latent.get_shape().as_list()
        latent = layers.ca_block(latent, act_func=activation, scope='segment_attention')

        small_map_latent = layers.conv(latent, scope='segment_output_small_1', filter_dims=[3, 3, c],
                                       stride_dims=[1, 1], non_linear_fn=None)
        small_map_latent = layers.conv_normalize(small_map_latent, norm=norm, b_train=b_train, scope='small_map_latent_norm')
        small_map_latent = activation(small_map_latent)

        small_map_latent = layers.conv(small_map_latent, scope='segment_output_small_2', filter_dims=[1, 1, 8],
                                       stride_dims=[1, 1], non_linear_fn=activation)
        small_map = layers.conv(small_map_latent, scope='segment_output_small_3', filter_dims=[1, 1, 1],
                                stride_dims=[1, 1], non_linear_fn=tf.nn.sigmoid)

        r = input_height // h
        segment_high = layers.conv(latent, scope='upsacle_conv_average_map',
                                   filter_dims=[3, 3, r * r * segment_unit_block_depth],
                                   stride_dims=[1, 1], non_linear_fn=None)
        segment_high = tf.nn.depth_to_space(segment_high, r)
        segment_high = layers.conv_normalize(segment_high, norm=norm, b_train=b_train, scope='segment_high_map_norm')
        segment_high = activation(segment_high)

        segment_high = layers.conv(segment_high, scope='segment_high1', filter_dims=[3, 3, segment_unit_block_depth],
                                   stride_dims=[1, 1], non_linear_fn=None)
        segment_high = layers.conv_normalize(segment_high, norm=norm, b_train=b_train, scope='segment_high1_norm')
        segment_high = activation(segment_high)

        segment_high = layers.conv(segment_high, scope='segment_high_out1', filter_dims=[1, 1, 8],
                                   stride_dims=[1, 1], non_linear_fn=activation)

        segment_low = tf.image.resize_images(small_map_latent, size=[input_width, input_height])

        ada_seg_map_low = layers.conv(segment_low, scope='ada_seg_map_low', filter_dims=[1, 1, 4],
                                      stride_dims=[1, 1], non_linear_fn=activation)
        ada_seg_map_high = layers.conv(segment_high, scope='ada_seg_map_high', filter_dims=[1, 1, 4],
                                       stride_dims=[1, 1], non_linear_fn=activation)
        ada_seg_map = tf.concat([ada_seg_map_low, ada_seg_map_high], axis=-1)
        ada_seg_map = layers.conv(ada_seg_map, scope='ada_seg_map', filter_dims=[1, 1, 2],
                                  stride_dims=[1, 1], non_linear_fn=activation)
        ada_seg_map = tf.nn.softmax(ada_seg_map, axis=-1)
        alpha, beta = tf.split(ada_seg_map, 2, axis=-1)

        segment_map = alpha * segment_low + beta * segment_high

        segment_map = layers.conv(segment_map, scope='segment_ada_concat', filter_dims=[3, 3, 8],
                                  stride_dims=[1, 1], non_linear_fn=None)
        segment_map = layers.conv_normalize(segment_map, norm=norm, b_train=b_train, scope='segment_ada_concat_norm')
        segment_map = activation(segment_map)

        segment_map = layers.conv(segment_map, scope='segment_high_out2', filter_dims=[1, 1, 4],
                                  stride_dims=[1, 1], non_linear_fn=activation)
        segment_map = layers.conv(segment_map, scope='segment_high_out3', filter_dims=[1, 1, 1],
                                  stride_dims=[1, 1], non_linear_fn=tf.nn.sigmoid)

        # Make Less Sensitive
        if b_smoothing is True:
            segment_map = util.gaussian_smoothing(segment_map)

        return segment_map, small_map


def segment_decoder(lateral, feature, activation=tf.nn.relu, norm='instance', scope='segment_decoder',
                    b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print(scope)

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
        lateral_map = layers.conv(lateral, scope='lateral_concat', filter_dims=[1, 1, c],
                                  stride_dims=[1, 1], non_linear_fn=None)
        lateral_map = layers.conv_normalize(lateral_map, norm=norm, b_train=b_train,
                                            scope='lateral_map_feature_concat_norm')
        lateral_map = activation(lateral_map)

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

        # lateral_map = lateral
        _, h, w, c = lateral_map.get_shape().as_list()

        k = np.log2(h) // 2
        k = k + ((k + 1) % 2)
        MP = tf.nn.max_pool(lateral_map, ksize=[1, k, k, 1], strides=[1, 1, 1, 1], padding='SAME')
        # MP = lateral_map
        p1 = layers.avg_pool(MP, filter_dims=[h, w], stride_dims=[h, w])
        # p1 = layers.avg_pool(MP, filter_dims=[h, w], stride_dims=[h, w])
        p1 = layers.conv(p1, scope='p1_bt', filter_dims=[1, 1, c // 5],
                         stride_dims=[1, 1], non_linear_fn=activation)
        p1 = tf.image.resize_images(p1, size=[h, w])
        p2 = layers.avg_pool(MP, filter_dims=[h // 4, w // 4], stride_dims=[h // 4, w // 4])
        p2 = layers.conv(p2, scope='p2_bt', filter_dims=[1, 1, c // 5],
                         stride_dims=[1, 1], non_linear_fn=activation)
        p2 = tf.image.resize_images(p2, size=[h, w])
        p3 = layers.avg_pool(MP, filter_dims=[h // 7, w // 7], stride_dims=[h // 7, w // 7])
        p3 = layers.conv(p3, scope='p3_bt', filter_dims=[1, 1, c // 5],
                         stride_dims=[1, 1], non_linear_fn=activation)
        p3 = tf.image.resize_images(p3, size=[h, w])
        # p4 = layers.avg_pool(MP, filter_dims=[h // 32, w // 32], stride_dims=[h // 32, w // 32])
        # p4 = layers.conv(p4, scope='p4_bt', filter_dims=[1, 1, c // 4],
        #                 stride_dims=[1, 1], non_linear_fn=activation)
        # p4 = tf.image.resize_images(p4, size=[h, w])

        p5 = layers.avg_pool(MP, filter_dims=[h // 13, w // 13], stride_dims=[h // 13, w // 13])
        p5 = layers.conv(p5, scope='p5_bt', filter_dims=[1, 1, c // 5],
                         stride_dims=[1, 1], non_linear_fn=activation)
        p5 = tf.image.resize_images(p5, size=[h, w])
        p6 = layers.avg_pool(MP, filter_dims=[h // 25, w // 25], stride_dims=[h // 25, w // 25])
        p6 = layers.conv(p6, scope='p6_bt', filter_dims=[1, 1, c // 5],
                         stride_dims=[1, 1], non_linear_fn=activation)
        p6 = tf.image.resize_images(p6, size=[h, w])

        psp_map = tf.concat([p1, p2, p3, p5, p6], axis=-1)

        # lateral_map = layers.conv(lateral_map, scope='ada_lateral_map', filter_dims=[1, 1, c // 8],
        #                          stride_dims=[1, 1], non_linear_fn=activation)

        # Adaptive Pyramid Pooling
        ada_psp_map = layers.conv(psp_map, scope='ada_psp', filter_dims=[1, 1, 5],
                                  stride_dims=[1, 1], non_linear_fn=activation)
        ada_psp_map = tf.nn.softmax(ada_psp_map, axis=-1)
        w1, w2, w3, w5, w6 = tf.split(ada_psp_map, 5, axis=-1)
        psp_map = w1 * p1 + w2 * p2 + w3 * p3 + w5 * p5 + w6 * p6
        psp_lateral = tf.concat([lateral_map, psp_map], axis=-1)
        psp_lateral = layers.conv(psp_lateral, scope='psp_lateral_concat_conv',
                                  filter_dims=[1, 1, segment_unit_block_depth],
                                  stride_dims=[1, 1], non_linear_fn=None)
        psp_lateral = layers.conv_normalize(psp_lateral, norm=norm, scope='psp_lateral_map_norm_concat')
        psp_lateral = activation(psp_lateral)

        return psp_lateral


def reconstruction_decoder(structual_latent, activation=tf.nn.relu, norm='instance',
                           scope='reconstruction_decoder'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        lateral_map = structual_latent
        _, h, w, c = structual_latent.get_shape().as_list()
        reconstruction_block_depth = segment_unit_block_depth // 2

        k = np.log2(h) // 2
        k = k + ((k + 1) % 2)
        MP = tf.nn.max_pool(lateral_map, ksize=[1, k, k, 1], strides=[1, 1, 1, 1], padding='SAME')

        p1 = layers.avg_pool(MP, filter_dims=[h, w], stride_dims=[h, w])
        p1 = layers.conv(p1, scope='p1_bt', filter_dims=[1, 1, reconstruction_block_depth // 4],
                         stride_dims=[1, 1], non_linear_fn=activation)
        p1 = tf.image.resize_images(p1, size=[h, w])
        p2 = layers.avg_pool(MP, filter_dims=[h // 3, w // 3], stride_dims=[h // 3, w // 3])
        p2 = layers.conv(p2, scope='p2_bt', filter_dims=[1, 1, reconstruction_block_depth // 4],
                         stride_dims=[1, 1], non_linear_fn=activation)
        p2 = tf.image.resize_images(p2, size=[h, w])
        p3 = layers.avg_pool(MP, filter_dims=[h // 5, w // 5], stride_dims=[h // 5, w // 5])
        p3 = layers.conv(p3, scope='p3_bt', filter_dims=[1, 1, reconstruction_block_depth // 4],
                         stride_dims=[1, 1], non_linear_fn=activation)
        p3 = tf.image.resize_images(p3, size=[h, w])
        p4 = layers.avg_pool(MP, filter_dims=[h // 9, w // 9], stride_dims=[h // 9, w // 9])
        p4 = layers.conv(p4, scope='p4_bt', filter_dims=[1, 1, reconstruction_block_depth // 4],
                         stride_dims=[1, 1], non_linear_fn=activation)
        p4 = tf.image.resize_images(p4, size=[h, w])
        psp_map = tf.concat([p1, p2, p3, p4], axis=-1)

        lateral_map = tf.concat([lateral_map, psp_map], axis=-1)
        lateral_map = layers.conv(lateral_map, scope='pyramid_lateral_map',
                                  filter_dims=[1, 1, reconstruction_block_depth],
                                  stride_dims=[1, 1], non_linear_fn=None)
        lateral_map = layers.conv_normalize(lateral_map, norm=norm, scope='lateral_map_map_norm_concat')
        lateral_map = activation(lateral_map)

        sup_small_map = layers.conv(lateral_map, scope='sup_small_map_output_last', filter_dims=[3, 3, 3],
                                    stride_dims=[1, 1], non_linear_fn=tf.nn.sigmoid)

        print(' sup_small_map: ' + str(sup_small_map.get_shape().as_list()))

        small_map = layers.conv(lateral_map, scope='upsacle_small_map',
                                filter_dims=[3, 3, 2 * 2 * reconstruction_block_depth],
                                stride_dims=[1, 1], non_linear_fn=None)

        small_map = tf.nn.depth_to_space(small_map, 2)
        small_map = layers.conv_normalize(small_map, norm=norm, scope='small_map_norm_last')
        small_map = activation(small_map)

        small_map = layers.conv(small_map, scope='feature_upscale', filter_dims=[3, 3, reconstruction_block_depth],
                                stride_dims=[1, 1], non_linear_fn=None)
        small_map = layers.conv_normalize(small_map, norm=norm, scope='feature_upscale_norm')
        small_map = activation(small_map)

        # Reconstruct
        small_map = layers.conv(small_map, scope='small_map_output_last', filter_dims=[3, 3, 3],
                                stride_dims=[1, 1], non_linear_fn=tf.nn.sigmoid)
        print(' small_map: ' + str(small_map.get_shape().as_list()))

        return small_map, sup_small_map


def categorical_sample(logits):
    u = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)


def create_roi_mask(width, height, offset=60):
    empty_img = np.ones((width, height, num_channel), dtype=np.uint8) * 1.0
    center_x = width // 2
    center_y = height // 2
    center = np.array((center_x, center_y))
    radius = 1.0 * (width // 2) - offset

    for i in range(width):
        for j in range(height):
            p = np.array((i, j))
            d = np.linalg.norm(p - center)
            if d > radius:
                empty_img[i, j] = 0

    mask_img = empty_img

    return mask_img


def train(model_path='None', mode='train'):
    print('Please wait. Preparing to start training...')
    train_start_time = time.time()
    outlier_files = []
    if use_outlier_samples is True:
        # Classes
        raw_aug_files = os.listdir(aug_data)
        print('Load augmentation samples, Total Num of Samples: ' + str(len(raw_aug_files)))

        for a_file in raw_aug_files:
            a_file_path = os.path.join(aug_data, a_file).replace("\\", "/")
            outlier_files.append(a_file_path)
        outlier_files = shuffle(outlier_files)
        outlier_files = np.array(outlier_files)

    man_outliler_files = []
    if use_outlier_samples is True:
        raw_aug_files = os.listdir(man_aug_data)
        print('Load augmentation samples, Total Num of Samples: ' + str(len(raw_aug_files)))

        for a_file in raw_aug_files:
            a_file_path = os.path.join(man_aug_data, a_file).replace("\\", "/")
            man_outliler_files.append(a_file_path)
        man_outliler_files = shuffle(man_outliler_files)
        man_outliler_files = np.array(man_outliler_files)

    # Batch size : Learning rate = 500: 1
    learning_rate = 5e-4
    warmup_step = 0
    global_step = 0
    lr = 0.5 * learning_rate

    tf.reset_default_graph()

    X_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
    S_IN = tf.placeholder(tf.float32, [None, input_height, input_width, 1])
    GT_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
    B_TRAIN = True
    LR = tf.placeholder(tf.float32, None)

    S_IN_Small = tf.image.resize_images(S_IN, size=[input_height // 2, input_width // 2])
    input_feature = segment_encoder_init(X_IN, norm='instance', scope=SEGMENT_Encoder_Init_scope,
                                         activation=layers.swish, b_train=B_TRAIN)
    laterals, features = segment_encoder(input_feature, norm='instance', scope=SEGMENT_Encoder_scope,
                                         activation=layers.swish, b_train=B_TRAIN)

    segment_feature = segment_decoder(laterals, features[-1], norm='instance', activation=layers.swish,
                                      scope=SEGMENT_Decoder_scope, b_train=B_TRAIN)
    U_G_X, U_G_X_Small = segment_decoder_out(segment_feature, norm='instance', activation=layers.swish,
                                             scope=SEGMENT_Decoder_Out_scope, b_train=B_TRAIN)
    Segment_out_map, _ = segment_decoder_out(segment_feature, norm='instance', activation=layers.swish,
                                             scope=SEGMENT_Decoder_Out_scope, b_train=False, b_smoothing=False)
    print('Segment decoder images: ' + str(U_G_X.get_shape().as_list()))
    segment_residual_loss = 0.0
    segment_residual_loss += get_residual_loss(U_G_X_Small, S_IN_Small, type='l1_focal')
    segment_residual_loss += get_residual_loss(U_G_X, S_IN, type='l1_focal')
    segment_residual_loss += get_residual_loss(U_G_X_Small, S_IN_Small, type='ft', alpha=0.7)
    segment_residual_loss += get_residual_loss(U_G_X, S_IN, type='ft', alpha=0.7)

    segment_encoder_init_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=SEGMENT_Encoder_Init_scope)
    segment_encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=SEGMENT_Encoder_scope)
    segment_decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=SEGMENT_Decoder_scope)
    segment_decoder_out_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=SEGMENT_Decoder_Out_scope)

    segment_generator_vars = segment_encoder_init_vars + segment_encoder_vars + segment_decoder_vars + segment_decoder_out_vars
    # Optimizer
    l2_lambda = 0
    l2_weight_decay = tf.reduce_mean([tf.nn.l2_loss(v) for v in segment_generator_vars])

    segment_loss = segment_residual_loss + l2_lambda * l2_weight_decay
    segmentation_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(segment_loss,
                                                                               var_list=segment_generator_vars)
    supervised_segmentation_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(segment_loss,
                                                                                          var_list=segment_generator_vars)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    segment_model_path = os.path.join(model_path, 'm.chpt').replace("\\", "/")
    segment_encoder_init_saver = tf.train.Saver(segment_encoder_init_vars)
    segment_encoder_saver = tf.train.Saver(segment_encoder_vars)
    segment_decoder_saver = tf.train.Saver(segment_decoder_vars)
    segment_decoder_out_saver = tf.train.Saver(segment_decoder_out_vars)
    segment_generator_saver = tf.train.Saver(segment_generator_vars)

    tf.add_to_collection('input', X_IN)
    tf.add_to_collection('output', Segment_out_map)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        try:
            print('Loading segment encoder init...')
            segment_encoder_init_saver.restore(sess, segment_model_path)
            print('Success to load segment encoder init.')
        except Exception as e:
            print('Fail to load segment encoder init.')
            print(e)

        try:
            print('Loading segment encoder...')
            segment_encoder_saver.restore(sess, segment_model_path)
            print('Success to load segment encoder.')
        except Exception as e:
            print('Fail to load segment encoder.')
            print(e)

        try:
            print('Loading segment decoder...')
            segment_decoder_saver.restore(sess, segment_model_path)
            print('Success to load segment decoder.')
        except Exception as e:
            print('Fail to load segment decoder.')
            print(e)

        try:
            print('Loading segment decoder map...')
            segment_decoder_out_saver.restore(sess, segment_model_path)
            print('Success to load segment decoder map.')
        except Exception as e:
            print('Fail to load segment decoder map.')
            print(e)

        tr_dir = train_data
        up_dir = update_data

        # Supervised Settings
        labeled_list_X = os.listdir('data/supervised/X')
        labeled_list_Y = os.listdir('data/supervised/Y')

        labeled_X = []
        labeled_Y = []
        for file_x in labeled_list_X:
            labeled_file_X = 'data/supervised/X/' + file_x
            labeled_file_Y = 'data/supervised/Y/' + file_x.split('.')[0] + '.png'
            labeled_X.append(labeled_file_X)
            labeled_Y.append(labeled_file_Y)

        labeled_X = np.array(labeled_X)
        labeled_Y = np.array(labeled_Y)

        te_dir = test_data
        te_files = os.listdir(te_dir)
        warm_lr_epoch = 1  # num_epoch // 10

        sample_dic = {}

        classes = os.listdir(tr_dir)
        print(' Train classes: ' + str(len(classes)))

        for cls in classes:
            class_path = os.path.join(tr_dir, cls).replace("\\", "/")
            samples = os.listdir(class_path)
            sample_dic[class_path] = samples

        if mode == 'update':
            update_classes = os.listdir(up_dir)
            print(' Update classes: ' + str(len(update_classes)))

            for cls in update_classes:
                class_path = os.path.join(up_dir, cls).replace("\\", "/")
                samples = os.listdir(class_path)
                sample_dic[class_path] = samples

        for e in range(num_epoch):
            itr = 0

            tr_files = []
            for cls in classes:
                class_path = os.path.join(tr_dir, cls).replace("\\", "/")
                # samples = os.listdir(class_path)
                samples = sample_dic[class_path]
                if mode == 'update':
                    samples = np.random.choice(samples, size=1, replace=False)
                else:
                    replace = False
                    if len(samples) < num_samples_per_class:
                        replace = True
                    samples = np.random.choice(samples, size=num_samples_per_class,
                                               replace=replace)  # (1000//len(classes)))
                for s in samples:
                    sample_path = os.path.join(class_path, s).replace("\\", "/")
                    tr_files.append(sample_path)
            if mode == 'update':
                for cls in update_classes:
                    class_path = os.path.join(up_dir, cls).replace("\\", "/")
                    # samples = os.listdir(class_path)
                    samples = sample_dic[class_path]
                    replace = False
                    num_samples_per_class_for_update = 20
                    if len(samples) < num_samples_per_class_for_update:
                        replace = True
                    samples = np.random.choice(samples, size=num_samples_per_class_for_update,
                                               replace=replace)  # (1000//len(classes)))
                    for s in samples:
                        sample_path = os.path.join(class_path, s).replace("\\", "/")
                        tr_files.append(sample_path)

            tr_files = shuffle(tr_files)
            total_input_size = len(tr_files)
            tr_files = np.array(tr_files)
            warm_lr_steps = warm_lr_epoch * (total_input_size // batch_size)
            total_steps = num_epoch * (total_input_size // batch_size) - warm_lr_steps

            print(' Num samples per epoch: ' + str(len(tr_files)))

            training_batch = zip(range(0, total_input_size, batch_size),
                                 range(batch_size, total_input_size + 1, batch_size))

            for start, end in training_batch:
                if e >= warm_lr_epoch:
                    # lr = 0.5 * learning_rate * (1.0 + np.cos(np.pi * (e / num_epoch)))
                    lr = 0.5 * learning_rate * (1.0 + np.cos(np.pi * (global_step / total_steps)))
                    global_step = global_step + 1
                else:
                    lr = 0.5 * learning_rate * (1 + (warmup_step / warm_lr_steps))
                    warmup_step = warmup_step + 1

                # lr = 0.5 * learning_rate * (1.0 + np.cos(np.pi * (cur_step / total_steps)))

                itr = itr + 1
                b_use_cutdout = True
                b_use_outlier_samples = use_outlier_samples
                b_use_bg_samples = use_bg_samples

                train_with_normal_sample = True

                #if np.random.randint(1, 10) < 2:
                #    b_use_outlier_samples = False

                b_use_color_shift = False
                if b_use_outlier_samples is True:
                    sample_outlier_files = np.random.choice(outlier_files, size=1, replace=True)
                    man_outlier_files = np.random.choice(man_outliler_files, size=1, replace=True)

                    sample_outlier_imgs, _, _ = load_images(sample_outlier_files, gray_scale=True, flip=True,
                                                            rotate=True, rotate_degree=90, shift=False,
                                                            extend_ratio=1.1)
                    man_sample_outlier_imgs, _, _ = load_images(man_outlier_files, gray_scale=True, flip=True,
                                                                rotate=True, rotate_degree=90, shift=False)
                    sample_outlier_imgs = np.append(sample_outlier_imgs, man_sample_outlier_imgs, axis=0)
                    sample_outlier_imgs = np.sum(sample_outlier_imgs, axis=0)
                    # sample_outlier_imgs = aug_noise + sample_outlier_imgs
                    aug_noise = sample_outlier_imgs
                    aug_noise = np.where(aug_noise > 0.5, 1.0, 0)
                    b_use_cutdout = False
                else:
                    '''
                    # Color Shift
                    aug_noise = cutout_mask
                    b_use_color_shift = True
                    b_use_cutdout = False
                    b_use_bg_samples = False
                    '''
                    # Perlin Noise
                    perlin_res = int(np.random.choice([16, 32, 64], size=1))  # 1024 x 1024
                    # perlin_res = int(np.random.choice([8, 16, 32], size=1)) # 512 x 512
                    # perlin_res = 2, perlin_octave = 4 : for large smooth object augmentation.
                    # perlin_octave = 5
                    # noise = util.generate_fractal_noise_2d((input_width, input_height), (perlin_res, perlin_res),
                    #                                       perlin_octave)
                    noise = util.generate_perlin_noise_2d((input_width, input_height), (perlin_res, perlin_res))
                    perlin_noise = np.where(noise > np.average(noise), 1.0, 0.0)
                    perlin_noise = np.expand_dims(perlin_noise, axis=-1)
                    aug_noise = perlin_noise
                    # aug_noise = None

                if train_with_normal_sample is True:
                    batch_imgs, gt_imgs, seg_imgs = load_images(tr_files[start + 1:end],
                                                                flip=True,
                                                                noise_mask=aug_noise, cutout=b_use_cutdout, rotate=True,
                                                                rotate_degree=45, color_shift=b_use_color_shift)
                else:
                    batch_imgs, gt_imgs, seg_imgs = load_images(tr_files[start:end],
                                                                flip=True,
                                                                noise_mask=aug_noise, cutout=b_use_cutdout, rotate=True,
                                                                rotate_degree=45, color_shift=b_use_color_shift)

                seg_imgs = np.where(seg_imgs > 0, 1.0, 0.0)

                if b_use_bg_samples is True:
                    # noise_samples = np.random.choice(tr_files, size=batch_size)
                    num_samples = batch_size
                    if train_with_normal_sample is True:
                        num_samples = num_samples - 1

                    random_index_X = np.random.choice(total_input_size, size=num_samples, replace=False)
                    noise_sample_files_X = tr_files[random_index_X]

                    if b_use_outlier_samples is True:
                        random_index_Y = np.random.choice(len(outlier_files), size=num_samples, replace=False)
                        noise_sample_files_Y = outlier_files[random_index_Y]
                    else:
                        random_index_Y = np.random.choice(len(labeled_Y), size=num_samples, replace=False)
                        noise_sample_files_Y = labeled_Y[random_index_Y]

                    noise_sample_images, _, _ = load_images(noise_sample_files_X)
                    noise_sample_segments, _, _ = load_images(noise_sample_files_Y, gray_scale=True, extend_ratio=1.1)
                    noise_sample_images = np.flip(noise_sample_images)
                    noise_sample_segments = np.flip(noise_sample_segments)

                    # flip_axis = np.random.random_integers(low=1, high=2)
                    # noise_sample_images = np.flip(noise_sample_images, axis=flip_axis)
                    # noise_sample_segments = np.flip(noise_sample_segments, axis=flip_axis)
                    # noise_sample_imgs, _, _ = load_images(noise_samples, rotate=True)
                    blending_a = np.random.uniform(low=0.1, high=0.5)
                    # noise_sample_images = noise_sample_images + np.random.random(3)
                    # noise_sample_images = np.where(noise_sample_images > 1.0, 1.0, noise_sample_images)
                    noise_sample_images = (1 - blending_a) * noise_sample_images + blending_a * batch_imgs
                    # fg = seg_imgs * noise_sample_imgs

                    noise_strength = 0
                    if np.random.randint(0, 10) < 3:
                        noise_strength = 0.1
                    noise_sample_images = (np.random.rand(input_width, input_height, 3) * noise_strength + (
                            1 - noise_strength)) * noise_sample_images
                    fg = noise_sample_segments * noise_sample_images
                    # bg = (1 - seg_imgs) * batch_imgs
                    bg = (1 - noise_sample_segments) * batch_imgs
                    batch_imgs = fg + bg
                    seg_imgs = seg_imgs + noise_sample_segments
                    seg_imgs = np.where(seg_imgs > 0, 1.0, 0.0)

                if train_with_normal_sample is True:
                    _, gt, seg = load_images([tr_files[start]], flip=True,
                                             noise_mask=None, cutout=False, rotate=True, rotate_degree=45)
                    batch_imgs = np.append(batch_imgs, gt, axis=0)
                    gt_imgs = np.append(gt_imgs, gt, axis=0)
                    seg_imgs = np.append(seg_imgs, seg, axis=0)

                # batch_imgs, gt_imgs, seg_imgs = shuffle(batch_imgs, gt_imgs, seg_imgs)

                _, segment_g_loss, u_g_x_imgs = sess.run([segmentation_optimizer, segment_loss, U_G_X],
                                                         feed_dict={X_IN: batch_imgs, S_IN: seg_imgs, GT_IN: gt_imgs,
                                                                    LR: lr})
                print('unsupervised epoch: ' + str(e) + ', segment loss: ' + str(segment_g_loss))

                if itr % 10 == 0:
                    cv2.imwrite(out_dir + '/' + str(itr) + '.jpg', 255 * batch_imgs[0])
                    cv2.imwrite(out_dir + '/' + str(itr) + '_gt.jpg', 255 * seg_imgs[0])
                    cv2.imwrite(out_dir + '/' + str(itr) + '_pred.jpg', 255 * u_g_x_imgs[0])

                    print('Elapsed Time at  ' + str(e) + '/' + str(num_epoch) + ' epochs, ' +
                          str(time.time() - train_start_time) + ' sec')

                if itr % 30 == 0:
                    try:
                        print('Saving model...')
                        segment_generator_saver.save(sess, segment_model_path)
                        print('Saved.')
                    except Exception as e:
                        print('Save failed')
                        print(e)

            if use_semisupervised is True:
                sampled_indexes = np.random.choice(np.arange(len(labeled_Y)), size=512, replace=False)
                sup_X = labeled_X[sampled_indexes]
                sup_Y = labeled_Y[sampled_indexes]
                sup_X, sup_Y = shuffle(sup_X, sup_Y)

                # sup_X = labeled_X
                # sup_Y = labeled_Y

                training_batch = zip(range(0, len(sup_X), batch_size),
                                     range(batch_size, len(sup_X) + 1, batch_size))

                for start, end in training_batch:
                    labeled_file_X = sup_X[start:end]
                    labeled_file_Y = sup_Y[start:end]

                    labeled_img_X, _, _ = load_images(labeled_file_X, make_border=False)
                    labeled_img_Y, _, _ = load_images(labeled_file_Y, gray_scale=True, make_border=False)
                    labeled_img_Y = np.where(labeled_img_Y > 0.8, 1.0, 0)

                    if np.random.random_integers(low=0, high=10) < 7:
                        flip_axis = np.random.random_integers(low=1, high=2)
                        labeled_img_X = np.flip(labeled_img_X, flip_axis)
                        labeled_img_Y = np.flip(labeled_img_Y, flip_axis)

                    _, u_loss, u_g_x_imgs = sess.run([supervised_segmentation_optimizer, segment_loss, U_G_X],
                                                     feed_dict={X_IN: labeled_img_X, S_IN: labeled_img_Y,
                                                                GT_IN: labeled_img_X, LR: lr})
                    itr += 1
                    print('supervised epoch: ' + str(e) + ', segment loss: ' + str(u_loss))
                    if itr % 10 == 0:
                        cv2.imwrite(out_dir + '/' + str(itr) + '.jpg', 255 * labeled_img_X[0])
                        cv2.imwrite(out_dir + '/' + str(itr) + '_gt.jpg', 255 * labeled_img_Y[0])
                        cv2.imwrite(out_dir + '/' + str(itr) + '_pred.jpg', 255 * u_g_x_imgs[0])

                    if itr % 30 == 0:
                        try:
                            print('Saving model...')
                            # segment_generator_saver.save(sess, segment_model_path, write_meta_graph=False)
                            segment_generator_saver.save(sess, segment_model_path)
                            print('Saved.')
                        except Exception as e:
                            print('Save failed')
                            print(e)

            te_batch = zip(range(0, len(te_files), batch_size),
                           range(batch_size, len(te_files) + 1, batch_size))

            for t_s, t_e in te_batch:
                test_imgs, _, _ = load_images(te_files[t_s:t_e], base_dir=te_dir)
                u_gx_imgs = sess.run(U_G_X, feed_dict={X_IN: test_imgs})

                for i in range(batch_size):
                    cv2.imwrite('out/' + te_files[t_s + i], 255 * u_gx_imgs[i])
            try:
                print('Saving model...')
                # segment_generator_saver.save(sess, segment_model_path, write_meta_graph=False)
                segment_generator_saver.save(sess, segment_model_path)
                print('Saved.')
            except Exception as e:
                print('Save failed')
                print(e)


def calculate_anomaly_score(img, width, height, filter_size=16):
    step = filter_size // 2
    score_list = [0.0]
    threshold = 0
    alpha = 0.8

    for p_y in range(0, height - step, step):
        for p_x in range(0, width - step, step):
            roi = img[p_x:p_x + filter_size, p_y:p_y + filter_size]
            score = np.mean(roi)

            if score > threshold:
                score_list.append(score)

    max_score = np.max(score_list)
    mean_score = np.mean(score_list)
    anomaly_score = (1 - alpha) * mean_score + alpha * max_score

    return anomaly_score


def spatial_memory(query, size, dims, scope='spatial_aug_mem'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        aug_mem = tf.get_variable('mem_vars', [size, dims], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer)
        _, h, w, dim = query.get_shape().as_list()
        norm_q = tf.reshape(query, shape=[-1, dim])
        norm_q = tf.nn.l2_normalize(norm_q, axis=0)
        norm_mem = tf.nn.l2_normalize(aug_mem, axis=0)

        distance = tf.matmul(norm_q, norm_mem, transpose_b=True)  # [B x H x W, Q]
        sm = tf.nn.softmax(distance, axis=-1)  # [B x H x W, Q]

        threashold = tf.constant(1 / aug_mem_size, dtype=tf.float32)
        attention = tf.multiply(tf.nn.relu(sm - threashold), sm)
        attention = attention / (1e-7 + tf.abs(sm - threashold))

        l1 = tf.expand_dims(tf.reduce_sum(attention, axis=-1), axis=-1)
        attention = attention / (l1 + 1e-7)

        latent = tf.matmul(attention, aug_mem)  # [B x H x W, Q] x [Q, R] = [B x H x W, R]
        latent = tf.reshape(latent, shape=[-1, h, w, dim])

        return attention, latent


def memory(query, size, dims, scope='aug_mem'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        aug_mem = tf.get_variable('mem_vars', [size, dims], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer)
        print(scope + ' mem size: ' + str(aug_mem.get_shape().as_list()))
        # query = [B, R], aug_mem = [Q, R]
        norm_q = tf.nn.l2_normalize(query, axis=0)
        print(scope + ' norm_q: ' + str(norm_q.get_shape().as_list()))
        norm_mem = tf.nn.l2_normalize(aug_mem, axis=0)
        print(scope + ' norm mem: ' + str(norm_mem.get_shape().as_list()))

        distance = tf.matmul(norm_q, norm_mem, transpose_b=True)  # [B, Q]

        # Simple Adaptive Scaling
        dist_scale = layers.fc(query, aug_mem_size // 8, non_linear_fn=tf.nn.relu, scope='mem_dist_scale')
        scale = layers.fc(dist_scale, aug_mem_size, non_linear_fn=None, scope='mem_linear_scale')
        shift = layers.fc(dist_scale, aug_mem_size, non_linear_fn=None, scope='mem_linear_shift')

        distance = scale * distance + shift  # [B, Q]

        sm = tf.nn.softmax(distance, axis=-1)  # [B, Q]
        print(scope + ' softmax: ' + str(sm.get_shape().as_list()))
        threashold = tf.constant(1 / aug_mem_size, dtype=tf.float32)
        attention = tf.multiply(tf.nn.relu(sm - threashold), sm)
        attention = attention / (1e-4 + tf.abs(sm - threashold))
        print(scope + ' attention: ' + str(attention.get_shape().as_list()))
        l1 = tf.expand_dims(tf.reduce_sum(attention, axis=-1), axis=-1)
        print(scope + ' l1: ' + str(l1.get_shape().as_list()))
        attention = attention / (l1 + 1e-7)

        latent = tf.matmul(attention, aug_mem)  # [B, Q] x [Q, R] = [B, R]
        print(scope + ' attention: ' + str(attention.get_shape().as_list()))

        return attention, latent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train/test/update', default='train')
    parser.add_argument('--model_path', type=str, help='model check point file path', default='model/m.ckpt')
    parser.add_argument('--train_data', type=str, help='training data directory', default='data/train')
    parser.add_argument('--test_data', type=str, help='test data directory', default='data/test')
    parser.add_argument('--update_data', type=str, help='update data directory', default='data/update')
    parser.add_argument('--aug_data', type=str, help='augmentation samples', default='data/augmentation')
    parser.add_argument('--man_aug_data', type=str, help='mandatory augmentation samples',
                        default='data/man_augmentation')
    parser.add_argument('--noise_data', type=str, help='specific noise data samples', default='data/noise')
    parser.add_argument('--out_dir', type=str, help='output directory', default='imgs')
    parser.add_argument('--bgmask_data', type=str, help='background mask sample director', default='bgmask')
    parser.add_argument('--img_size', type=int, help='training image size', default=512)
    parser.add_argument('--epoch', type=int, help='num epoch', default=1000)
    parser.add_argument('--batch_size', type=int, help='Training batch size', default=16)
    parser.add_argument('--alpha', type=int, help='AE loss weight', default=1)

    args = parser.parse_args()

    input_width = args.img_size
    input_height = args.img_size
    batch_size = args.batch_size
    mode = args.mode
    model_path = args.model_path
    train_data = args.train_data
    test_data = args.test_data
    update_data = args.update_data
    out_dir = args.out_dir
    num_epoch = args.epoch
    alpha = args.alpha
    aug_data = args.aug_data
    man_aug_data = args.man_aug_data
    bg_mask_data = args.bgmask_data
    noise_data = args.noise_data
    segment_unit_block_depth = 64
    segmentation_upsample_ratio = 4
    segmentation_downsample_num = int(np.log2(segmentation_upsample_ratio))
    segmentation_upsample_num = segmentation_downsample_num
    num_channel = 3
    num_shuffle_car = 4
    num_car = 2
    use_outlier_samples = True
    use_bg_samples = True
    num_samples_per_class = 4
    aug_mem_size = 512
    use_semisupervised = True
    use_domain_std = False

    train(model_path, mode)



