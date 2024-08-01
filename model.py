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
SEGMENT_Decoder_Out_scope = 'final_map'

def load_images(file_name_list, base_dir=None, noise_mask=None, cutout=False, cutout_mask=None,
                add_eps=False, rotate=False, flip=False, gray_scale=False, shift=False, contrast=False):
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

            if contrast is True:
                if np.random.randint(0, 10) < 5:
                    img = cv2.resize(img, dsize=(input_width // 2, input_height // 2), interpolation=cv2.INTER_AREA)

            img = cv2.resize(img, dsize=(input_width, input_height), interpolation=cv2.INTER_AREA)

            if gray_scale is True:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if img is not None:
                img = np.array(img) * 1.0

                if rotate is True:
                    rot = np.random.randint(-10, 10)
                    img = util.rotate_image(img, rot)

                if flip is True:
                    code = np.random.randint(low=-1, high=2)
                    img = cv2.flip(img, code)

                if shift is True:
                    x_shift = np.random.randint(-25, 25)
                    y_shift = np.random.randint(-25, 25)
                    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
                    img = cv2.warpAffine(img, M, (input_width, input_height))

                gt_img = img.copy()
                gt_img = np.array(gt_img)

                if add_eps is True:
                    img = img + np.random.uniform(low=-2.55, high=2.55, size=img.shape)

                if gray_scale is True:
                    img = np.expand_dims(img, axis=-1)
                    gt_img = np.expand_dims(gt_img, axis=-1)

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
                    alpha = np.random.uniform(low=0.1, high=0.9)

                    if np.random.randint(low=0, high=10) < 5:
                        random_pixels = 2 * (0.1 + np.random.rand())
                        structural_noise = util.rotate_image(img, np.random.randint(0, 360))
                        cut_mask = np.abs(cut_mask * (structural_noise * random_pixels))
                    else:
                        cut_mask = 128.0 *  cut_mask

                    cut_mask = np.where(cut_mask > 255, 255, cut_mask)
                    img = bg_img + (alpha * fg_img + (1 - alpha) * cut_mask)
                else:
                    if noise_mask is not None:
                        cut_mask = noise_mask

                        # Segmentation Mode
                        seg_img = cut_mask
                        bg_img = (1.0 - seg_img) * img
                        fg_img = seg_img * img
                        alpha = np.random.uniform(low=0.5, high=0.7)

                        random_pixels = 2 * (0.1 + np.random.rand())
                        structural_noise = util.rotate_image(img, np.random.randint(0, 360))
                        cut_mask = np.abs(cut_mask * (structural_noise * random_pixels))
                        cut_mask = np.abs(np.where(cut_mask > 255, 255, cut_mask))
                        img = bg_img + (alpha * fg_img + (1 - alpha) * cut_mask)
                        # img = (1.0 - cut_mask) * img

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

    loss1 = tf.reduce_mean(tf.abs(tf.subtract(dx_a, dx_b)))
    loss2 = tf.reduce_mean(tf.abs(tf.subtract(dy_a, dy_b)))

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


def segment_encoder_init(in_tensor, activation=tf.nn.relu, norm='instance', scope='segment_encoder_init', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print('Segment encoder init: ' + str(in_tensor.get_shape().as_list()))
        l = layers.conv(in_tensor, scope='init_512', filter_dims=[7, 7, 4 * segment_unit_block_depth],
                        stride_dims=[2, 2], non_linear_fn=activation)
        l = layers.conv(l, scope='init_512_squeeze', filter_dims=[1, 1, segment_unit_block_depth],
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
            l = layers.conv(l, scope='downsample_squeeze_' + str(i), filter_dims=[1, 1, block_depth],
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
                    print(' Conv Layer: ' + str(lateral_layers[i].get_shape().as_list()))
                    lateral_head2 = layers.add_residual_block(lateral_head2, filter_dims=[3, 3, c], norm=norm,
                                                  b_train=b_train, act_func=activation,
                                                  scope='loop_sqblock_' + str(n_loop) + str(i) + str(num_rblock))
                lateral_head2 = layers.ca_block(lateral_head2, act_func=activation, scope='car_' + str(n_loop) + str(i))

                lateral_layers[i] = tf.concat([lateral_head1, lateral_head2], axis=-1)

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
                                                filter_dims=[1, 1, l_c],
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

                if n_loop == (num_shuffle_car - 1):
                    break

            lateral_layers = fused_layers

        lateral_layers.reverse()
        feature_layers.reverse()

        return lateral_layers, feature_layers


def segment_decoder_out(latent, activation=tf.nn.relu, norm='instance', scope='segment_decoder_out',  b_train=False, b_smoothing=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print(scope)
        _, h, w, c = latent.get_shape().as_list()
        segment_small = layers.conv(latent, scope='segment_output_small_1', filter_dims=[1, 1, 8],
                                    stride_dims=[1, 1], non_linear_fn=activation)
        segment_small_map = layers.conv(segment_small, scope='segment_output_small_2', filter_dims=[1, 1, 1],
                                        stride_dims=[1, 1], non_linear_fn=tf.nn.sigmoid)

        r = input_height // h
        segment_high = layers.conv(latent, scope='upsacle_conv_average_map',
                                   filter_dims=[3, 3, r * r * segment_unit_block_depth],
                                   stride_dims=[1, 1], non_linear_fn=activation)
        segment_high = tf.nn.depth_to_space(segment_high, r)

        segment_high = layers.conv(segment_high, scope='segment_high1', filter_dims=[1, 1, 8],
                                   stride_dims=[1, 1], non_linear_fn=activation)
        segment_map = layers.conv(segment_high, scope='segment_high2', filter_dims=[1, 1, 1],
                                  stride_dims=[1, 1], non_linear_fn=tf.nn.sigmoid)
        # Make Less Sensitive
        if b_smoothing is True:
            segment_map = util.gaussian_smoothing(segment_map)

        return segment_map, segment_small_map


def segment_decoder(lateral_layers, skip_layers, activation=tf.nn.relu, norm='instance', scope='segment_decoder',
                    b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print(scope)

        lateral_map = lateral_layers[-1]
        _, h, w, c = lateral_map.get_shape().as_list()

        # Adaptive Concat
        skip_map = skip_layers[-1]
        ada_lat_map = layers.conv(lateral_map, scope='ada_lat_map', filter_dims=[1, 1, c//8],
                                  stride_dims=[1, 1], non_linear_fn=activation)
        ada_skip_map = layers.conv(skip_map, scope='ada_skip_map', filter_dims=[1, 1, c//8],
                                  stride_dims=[1, 1], non_linear_fn=activation)
        ada_map = tf.concat([ada_lat_map, ada_skip_map], axis=-1)
        ada_map = layers.conv(ada_map, scope='ada_alpha', filter_dims=[1, 1, 2],
                            stride_dims=[1, 1], non_linear_fn=activation)
        ada_map = tf.nn.softmax(ada_map, axis=-1)
        alpha, beta = tf.split(ada_map, 2, axis=-1)

        lateral_map = alpha * lateral_map + beta * skip_map

        lateral_map = layers.conv(lateral_map, scope='feature_concat', filter_dims=[3, 3, segment_unit_block_depth],
                                  stride_dims=[1, 1], non_linear_fn=None)
        lateral_map = layers.conv_normalize(lateral_map, norm=norm, b_train=b_train,
                                            scope='feature_concat_norm')
        lateral_map = activation(lateral_map)

        k = np.log2(h) // 2
        k = k + ((k + 1) % 2)
        MP = tf.nn.max_pool(lateral_map, ksize=[1, k, k, 1], strides=[1, 1, 1, 1], padding='SAME')
        #MP = lateral_map
        #p1 = layers.avg_pool(MP, filter_dims=[h // 4, w // 4], stride_dims=[h // 4, w // 4])
        p1 = layers.avg_pool(MP, filter_dims=[h, w], stride_dims=[h, w])
        p1 = layers.conv(p1, scope='p1_bt', filter_dims=[1, 1, c // 8],
                         stride_dims=[1, 1], non_linear_fn=activation)
        p1 = tf.image.resize_images(p1, size=[h, w])
        p2 = layers.avg_pool(MP, filter_dims=[h // 6, w // 6], stride_dims=[h // 6, w // 6])
        p2 = layers.conv(p2, scope='p2_bt', filter_dims=[1, 1, c // 8],
                         stride_dims=[1, 1], non_linear_fn=activation)
        p2 = tf.image.resize_images(p2, size=[h, w])
        p3 = layers.avg_pool(MP, filter_dims=[h // 10, w // 10], stride_dims=[h // 10, w // 10])
        p3 = layers.conv(p3, scope='p3_bt', filter_dims=[1, 1, c // 8],
                         stride_dims=[1, 1], non_linear_fn=activation)
        p3 = tf.image.resize_images(p3, size=[h, w])
        p4 = layers.avg_pool(MP, filter_dims=[h // 16, w // 16], stride_dims=[h // 16, w // 16])
        p4 = layers.conv(p4, scope='p4_bt', filter_dims=[1, 1, c // 8],
                         stride_dims=[1, 1], non_linear_fn=activation)
        p4 = tf.image.resize_images(p4, size=[h, w])
        p5 = layers.avg_pool(MP, filter_dims=[h // 24, w // 24], stride_dims=[h // 24, w // 24])
        p5 = layers.conv(p5, scope='p5_bt', filter_dims=[1, 1, c // 8],
                         stride_dims=[1, 1], non_linear_fn=activation)
        p5 = tf.image.resize_images(p5, size=[h, w])
        p6 = layers.avg_pool(MP, filter_dims=[h // 36, w // 36], stride_dims=[h // 36, w // 36])
        p6 = layers.conv(p6, scope='p6_bt', filter_dims=[1, 1, c // 8],
                         stride_dims=[1, 1], non_linear_fn=activation)
        p6 = tf.image.resize_images(p6, size=[h, w])
        p7 = layers.avg_pool(MP, filter_dims=[h // 54, w // 54], stride_dims=[h // 54, w // 54])
        p7 = layers.conv(p7, scope='p7_bt', filter_dims=[1, 1, c // 8],
                         stride_dims=[1, 1], non_linear_fn=activation)
        p7 = tf.image.resize_images(p7, size=[h, w])
        p8 = layers.avg_pool(MP, filter_dims=[h // 80, w // 80], stride_dims=[h // 80, w // 80])
        p8 = layers.conv(p8, scope='p8_bt', filter_dims=[1, 1, c // 8],
                         stride_dims=[1, 1], non_linear_fn=activation)
        p8 = tf.image.resize_images(p8, size=[h, w])
        psp_map = tf.concat([p1, p2, p3, p4, p5, p6, p7, p8], axis=-1)

        #lateral_map = layers.conv(lateral_map, scope='ada_lateral_map', filter_dims=[1, 1, c // 8],
        #                          stride_dims=[1, 1], non_linear_fn=activation)

        # Adaptive Pyramid Pooling
        #ada_psp_map = layers.conv(psp_map, scope='ada_psp', filter_dims=[1, 1, 8],
         #                         stride_dims=[1, 1], non_linear_fn=activation)
        #ada_psp_map = tf.nn.softmax(ada_psp_map, axis=-1)
        #w1, w2, w3, w4, w5, w6, w7, w8 = tf.split(ada_psp_map, 8, axis=-1)
        #psp_map = w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4 + w5 * p5 + w6 * p6 + w7 * p7 + w8 * p8
        psp_lateral = tf.concat([lateral_map, psp_map], axis=-1)
        psp_lateral = layers.conv(psp_lateral, scope='psp_lateral_concat_conv', filter_dims=[1, 1, segment_unit_block_depth],
                                  stride_dims=[1, 1], non_linear_fn=activation)

        return psp_lateral

    
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

    #cutout_mask = create_roi_mask(input_width, input_height, offset=8)
    outlier_files = []
    if use_outlier_samples is True:
        # Classes
        raw_aug_files = os.listdir(aug_data)
        print('Load augmentation samples, Total Num of Samples: ' + str(len(raw_aug_files)))

        for a_file in raw_aug_files:
            a_file_path = os.path.join(aug_data, a_file).replace("\\", "/")
            outlier_files.append(a_file_path)
        outlier_files = shuffle(outlier_files)

    # Batch size : Learning rate = 500: 1
    learning_rate = 5e-4
    cur_step = 0
    lr = 0.5 * learning_rate

    tf.reset_default_graph()

    X_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
    S_IN = tf.placeholder(tf.float32, [None, input_height, input_width, 1])
    B_TRAIN = True
    LR = tf.placeholder(tf.float32, None)

    input_feature = segment_encoder_init(X_IN, norm='instance', scope=SEGMENT_Encoder_Init_scope,
                                         activation=layers.swish, b_train=B_TRAIN)
    laterals, features = segment_encoder(input_feature, norm='instance', scope=SEGMENT_Encoder_scope,
                                         activation=layers.swish, b_train=B_TRAIN)
    segment_feature = segment_decoder(laterals, features, norm='instance', activation=layers.swish,
                                      scope=SEGMENT_Decoder_scope, b_train=B_TRAIN)
    U_G_X, U_G_X_Small = segment_decoder_out(segment_feature, norm='instance', activation=layers.swish,
                                             scope=SEGMENT_Decoder_Out_scope, b_train=B_TRAIN)
    print('Segment decoder images: ' + str(U_G_X.get_shape().as_list()))
    segment_residual_loss = 0.0
    S_IN_Small = tf.image.resize_images(S_IN, size=[input_height // 2, input_width // 2])

    segment_residual_loss += get_residual_loss(U_G_X_Small, S_IN_Small, type='focal')  # Deep Supervision
    #segment_residual_loss += get_residual_loss(U_G_X_Small, S_IN_Small, type='ft', alpha=0.7)
    segment_residual_loss += get_residual_loss(U_G_X, S_IN, type='l1_focal')
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
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    segment_model_path = os.path.join(model_path, 'm.chpt').replace("\\", "/")

    segment_encoder_init_saver = tf.train.Saver(segment_encoder_init_vars)
    segment_encoder_saver = tf.train.Saver(segment_encoder_vars)
    segment_decoder_saver = tf.train.Saver(segment_decoder_vars)
    segment_decoder_out_saver = tf.train.Saver(segment_decoder_out_vars)
    segment_generator_saver = tf.train.Saver(segment_generator_vars)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        try:
            print('Loading segment encoder init...')
            segment_encoder_init_saver.restore(sess, segment_model_path)
            print('Success to load segment encoder init.')
        except:
            print('Fail to load segment encoder init.')

        try:
            print('Loading segment encoder...')
            segment_encoder_saver.restore(sess, segment_model_path)
            print('Success to load segment encoder.')
        except:
            print('Fail to load segment encoder.')

        try:
            print('Loading segment decoder...')
            segment_decoder_saver.restore(sess, segment_model_path)
            print('Success to load segment decoder.')
        except:
            print('Fail to load segment decoder.')

        try:
            print('Loading segment decoder map...')
            segment_decoder_out_saver.restore(sess, segment_model_path)
            print('Success to load segment decoder map.')
        except:
            print('Fail to load segment decoder map.')

        tr_dir = train_data
        up_dir = update_data

        # Classes
        classes = os.listdir(tr_dir)
        print(' Train classes: ' + str(len(classes)))

        if mode == 'update':
            update_classes = os.listdir(up_dir)
            print(' Update classes: ' + str(len(update_classes)))

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
        warm_lr_epoch = num_epoch // 10

        for e in range(num_epoch):
            itr = 0

            if use_semisupervised is True:
                sup_X, sup_Y = shuffle(labeled_X, labeled_Y)

                training_batch = zip(range(0, len(sup_X), batch_size),
                                     range(batch_size, len(sup_X) + 1, batch_size))

                for start, end in training_batch:
                    labeled_file_X = sup_X[start:end]
                    labeled_file_Y = sup_Y[start:end]

                    labeled_img_X, _, _ = load_images(labeled_file_X)
                    labeled_img_Y, _, _ = load_images(labeled_file_Y, gray_scale=True)

                    if np.random.random_integers(low=0, high=10) < 7:
                        flip_axis = np.random.random_integers(low=1, high=2)
                        labeled_img_X = np.flip(labeled_img_X, flip_axis)
                        labeled_img_Y = np.flip(labeled_img_Y, flip_axis)

                    _, u_loss, u_g_x_imgs = sess.run([segmentation_optimizer, segment_loss, U_G_X],
                                                     feed_dict={X_IN: labeled_img_X, S_IN: labeled_img_Y, LR: lr})
                    itr += 1
                    print('supervised epoch: ' + str(e) + ', segment loss: ' + str(u_loss))
                    if itr % 10 == 0:
                        cv2.imwrite(out_dir + '/' + str(itr) + '.jpg', 255 * labeled_img_X[0])
                        cv2.imwrite(out_dir + '/' + str(itr) + '_gt.jpg', 255 * labeled_img_Y[0])
                        cv2.imwrite(out_dir + '/' + str(itr) + '_pred.jpg', 255 * u_g_x_imgs[0])

                    if itr % 30 == 0:
                        try:
                            print('Saving model...')
                            segment_generator_saver.save(sess, segment_model_path, write_meta_graph=False)
                            print('Saved.')
                        except:
                            print('Save failed')

            #tr_files = shuffle(tr_files)
            tr_files = []
            for cls in classes:
                class_path = os.path.join(tr_dir, cls).replace("\\", "/")
                samples = os.listdir(class_path)
                if mode == 'update':
                    samples = np.random.choice(samples, size=1)
                else:
                    samples = np.random.choice(samples, size=num_samples_per_class)  # (1000//len(classes)))
                for s in samples:
                    sample_path = os.path.join(class_path, s).replace("\\", "/")
                    tr_files.append(sample_path)
            if mode == 'update':
                for cls in update_classes:
                    class_path = os.path.join(up_dir, cls).replace("\\", "/")
                    samples = os.listdir(class_path)
                    samples = np.random.choice(samples, size=num_samples_per_class - 1)  # (1000//len(classes)))
                    for s in samples:
                        sample_path = os.path.join(class_path, s).replace("\\", "/")
                        tr_files.append(sample_path)

            total_input_size = len(tr_files)
            warm_lr_steps = warm_lr_epoch * (total_input_size // batch_size)
            total_steps = num_epoch * (total_input_size // batch_size)

            print(' Num samples per epoch: ' + str(len(tr_files)))

            training_batch = zip(range(0, total_input_size, batch_size),
                                 range(batch_size, total_input_size + 1, batch_size))

            for start, end in training_batch:
                cur_step = cur_step + 1
                if e >= warm_lr_epoch:
                    #lr = 0.5 * learning_rate * (1.0 + np.cos(np.pi * (cur_step / warm_lr_steps)))
                    lr = 0.5 * learning_rate * (1.0 + np.cos(np.pi * (cur_step /total_steps)))
                else:
                    lr = 0.5 * learning_rate * (1 + (cur_step / warm_lr_steps))

                itr = itr + 1
                b_use_cutdout = True
                b_use_outlier_samples = use_outlier_samples
                b_use_bg_samples = use_bg_samples
                train_with_normal_sample = True

                if np.random.randint(1, 10) < 2:
                    b_use_outlier_samples = False

                #if np.random.randint(1, 10) < 3:
                #    b_use_bg_samples = False
                #    train_with_normal_sample = True

                if b_use_outlier_samples is True:
                    sample_outlier_files = np.random.choice(outlier_files, size=1)
                    sample_outlier_imgs, _, _ = load_images(sample_outlier_files, flip=True)
                    sample_outlier_imgs = np.sum(sample_outlier_imgs, axis=0)
                    # sample_outlier_imgs = aug_noise + sample_outlier_imgs
                    aug_noise = sample_outlier_imgs
                    aug_noise = np.where(aug_noise > 0.9, 1.0, 0.0)
                    b_use_cutdout = False
                else:
                    # Perlin Noise
                    perlin_res = int(np.random.choice([16, 32, 64], size=1))  # 1024 x 1024
                    # perlin_res = int(np.random.choice([8, 16, 32], size=1)) # 512 x 512
                    # perlin_res = 2, perlin_octave = 4 : for large smooth object augmentation.
                    #perlin_octave = 5
                    #noise = util.generate_fractal_noise_2d((input_width, input_height), (perlin_res, perlin_res),
                    #                                       perlin_octave)
                    noise = util.generate_perlin_noise_2d((input_width, input_height), (perlin_res, perlin_res))
                    perlin_noise = np.where(noise > np.average(noise), 1.0, 0.0)
                    perlin_noise = np.expand_dims(perlin_noise, axis=-1)
                    aug_noise = perlin_noise

                    #aug_noise = None

                if train_with_normal_sample is True:
                    batch_imgs, gt_imgs, seg_imgs = load_images(tr_files[start + 1:end],
                                                                flip=True,
                                                                noise_mask=aug_noise, cutout=b_use_cutdout)
                else:
                    batch_imgs, gt_imgs, seg_imgs = load_images(tr_files[start:end],
                                                                flip=True,
                                                                noise_mask=aug_noise, cutout=b_use_cutdout)
                seg_imgs = np.where(seg_imgs > 0, 1.0, 0.0)

                if b_use_bg_samples is True:
                    # noise_samples = np.random.choice(tr_files, size=batch_size)
                    num_samples = batch_size
                    if train_with_normal_sample is True:
                        num_samples = num_samples - 1
                    random_index = np.random.choice(len(labeled_X), size=num_samples, replace=False)
                    noise_sample_files_X = labeled_X[random_index]
                    noise_sample_files_Y = labeled_Y[random_index]
                    noise_sample_images, _, _ = load_images(noise_sample_files_X)
                    noise_sample_segments, _, _ = load_images(noise_sample_files_Y, gray_scale=True)
                    flip_axis = np.random.random_integers(low=1, high=2)
                    noise_sample_images = np.flip(noise_sample_images, axis=flip_axis)
                    noise_sample_segments = np.flip(noise_sample_segments, axis=flip_axis)
                    # noise_sample_imgs, _, _ = load_images(noise_samples, rotate=True)
                    blending_a = np.random.uniform(low=0.1, high=0.9)
                    #noise_sample_images = noise_sample_images + np.random.random(3)
                    #noise_sample_images = np.where(noise_sample_images > 1.0, 1.0, noise_sample_images)
                    noise_sample_images = (1 - blending_a) * noise_sample_images + blending_a * batch_imgs
                    # fg = seg_imgs * noise_sample_imgs
                    noise_strength = 0.1
                    noise_sample_images = (np.random.rand(input_width, input_height, 3) * noise_strength + (1 - noise_strength)) * noise_sample_images
                    fg = noise_sample_segments * noise_sample_images
                    # bg = (1 - seg_imgs) * batch_imgs
                    bg = (1 - noise_sample_segments) * batch_imgs
                    batch_imgs = fg + bg
                    seg_imgs = seg_imgs + noise_sample_segments
                    seg_imgs = np.where(seg_imgs > 0, 1.0, 0.0)

                if train_with_normal_sample is True:
                    b_img, gt, seg = load_images([tr_files[start]], flip=True,
                                                 noise_mask=None, cutout=False)
                    batch_imgs = np.append(batch_imgs, b_img, axis=0)
                    gt_imgs = np.append(gt_imgs, gt, axis=0)
                    seg_imgs = np.append(seg_imgs, seg, axis=0)

                batch_imgs, gt_imgs, seg_imgs = shuffle(batch_imgs, gt_imgs, seg_imgs)

                _, segment_g_loss, u_g_x_imgs = sess.run([segmentation_optimizer, segment_loss, U_G_X],
                                                         feed_dict={X_IN: batch_imgs, S_IN: seg_imgs, LR: lr})
                print('unsupervised epoch: ' + str(e) + ', segment loss: ' + str(segment_g_loss))

                if itr % 10 == 0:
                    cv2.imwrite(out_dir + '/' + str(itr) + '.jpg', 255 * batch_imgs[0])
                    cv2.imwrite(out_dir + '/' + str(itr) + '_gt.jpg', 255 * seg_imgs[0])
                    cv2.imwrite(out_dir + '/' + str(itr) + '_pred.jpg', 255 * u_g_x_imgs[0])
                    # cv2.imwrite(out_dir + '/' + str(itr) + '_recon.jpg', 255 * g_x_imgs[0])
                    print('Elapsed Time at  ' + str(e) + '/' + str(num_epoch) + ' epochs, ' +
                          str(time.time() - train_start_time) + ' sec')

            te_batch = zip(range(0, len(te_files), batch_size),
                           range(batch_size, len(te_files) + 1, batch_size))

            for t_s, t_e in te_batch:
                test_imgs, _, _ = load_images(te_files[t_s:t_e], base_dir=te_dir)
                u_gx_imgs = sess.run(U_G_X, feed_dict={X_IN: test_imgs})

                for i in range(batch_size):
                    cv2.imwrite('out/' + te_files[t_s + i], 255 * u_gx_imgs[i])
                    s = np.sum(u_gx_imgs[i])
                    print('anomaly score of ' + te_files[t_s + i] + ': ' + str(s))
            try:
                print('Saving model...')
                segment_generator_saver.save(sess, segment_model_path, write_meta_graph=False)
                print('Saved.')
            except:
                print('Save failed')


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


def test(model_path):
    print('Please wait. Preparing to test...')

    tf.reset_default_graph()

    X_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
    B_TRAIN = False

    input_feature = segment_encoder_init(X_IN, norm='instance', scope=SEGMENT_Encoder_Init_scope,
                                        activation=layers.swish, b_train=B_TRAIN)
    laterals, features = segment_encoder(input_feature, norm='instance', scope=SEGMENT_Encoder_scope,
                                         activation=layers.swish, b_train=B_TRAIN)
    segment_feature = segment_decoder(laterals, features, norm='instance', activation=layers.swish,
                                      scope=SEGMENT_Decoder_scope, b_train=B_TRAIN)
    U_G_X, _ = segment_decoder_out(segment_feature, norm='instance', activation=layers.swish,
                                             scope=SEGMENT_Decoder_Out_scope, b_train=B_TRAIN, b_smoothing=True)

    segment_encoder_init_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=SEGMENT_Encoder_Init_scope)
    segment_decoder_out_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=SEGMENT_Decoder_Out_scope)
    segment_encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=SEGMENT_Encoder_scope)
    segment_decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=SEGMENT_Decoder_scope)
    segment_generator_vars = segment_encoder_init_vars + segment_encoder_vars + segment_decoder_vars + segment_decoder_out_vars

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    segment_model_path = os.path.join(model_path, 'm.chpt').replace("\\", "/")
    segment_generator_saver = tf.train.Saver(segment_generator_vars)

    with tf.Session(config=config) as sess:
        try:
            print('Loading model...')
            segment_generator_saver.restore(sess, segment_model_path)
            print('Success to load.')
        except:
            print('Fail to load.' )
            return
        test_start_time = time.time()
        te_dir = test_data
        te_files = os.listdir(te_dir)
        te_batch = zip(range(0, len(te_files), batch_size),
                       range(batch_size, len(te_files) + 1, batch_size))

        for t_s, t_e in te_batch:
            test_imgs, _, _ = load_images(te_files[t_s:t_e], base_dir=te_dir)
            u_gx_imgs = sess.run(U_G_X, feed_dict={X_IN: test_imgs})
            u_gx_imgs = np.where(u_gx_imgs > 0.5, u_gx_imgs, 0.0)
            for i in range(batch_size):
                cv2.imwrite('out/' + te_files[t_s + i], 255 * u_gx_imgs[i])
                s = np.sum(u_gx_imgs[i])
                print('anomaly score of ' + te_files[t_s + i] + ': ' + str(s))

        print('Total ' + str(len(te_files)) + ' Samples. Elapsed Time:  ' + str(time.time() - test_start_time) + ' sec')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train/test/update', default='train')
    parser.add_argument('--model_path', type=str, help='model check point file path', default='model/m.ckpt')
    parser.add_argument('--train_data', type=str, help='training data directory', default='data/train')
    parser.add_argument('--test_data', type=str, help='test data directory', default='data/test')
    parser.add_argument('--update_data', type=str, help='update data directory', default='data/update')
    parser.add_argument('--aug_data', type=str, help='augmentation samples', default='data/augmentation')
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
    num_samples_per_class = 8
    use_semisupervised = True
    use_domain_std = False

    if mode == 'test':
        test(model_path)
    else:
        train(model_path, mode)
        
