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
import argparse
import time


# file_name_list: List of image file name string
# gray_scale: Load image in gray scale.
# return: List of normalized(0.0~1.0) image object. None(Failed)
def load_images(file_name_list, gray_scale=False):
    try:
        images = []

        for file_name in file_name_list:
            fullname = file_name

            img = cv2.imread(fullname)

            if img is None:
                print('Load failed: ' + fullname)
                return None

            h, w, c = img.shape

            if h != input_width:
                img = cv2.resize(img, dsize=(input_width, input_height), interpolation=cv2.INTER_AREA)

            if gray_scale is True:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img is not None:
                img = np.array(img) * 1.0
                n_img = (img * 1.0) / 255.0
                images.append(n_img)

    except cv2.error as e:
        print(e)
        return None

    return np.array(images)


# model_directory_path: Model directory path. Model directory should contain meta file, chpt file.
# return: Opened tensorflow session object, None(Failed)
def carnet_open(model_directory_path):
    tf.reset_default_graph()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    carnet_meta = os.path.join(model_directory_path, 'm.chpt.meta').replace("\\", "/")
    print('Loading graph...')
    carnet = tf.train.import_meta_graph(carnet_meta)
    print('Success to load.')
    carnet_data = os.path.join(model_directory_path, 'm.chpt').replace("\\", "/")

    try:
        p_session = tf.Session(config=config)
        print('Loading model...')
        carnet.restore(p_session, carnet_data)
        print('Success to load.')
    except:
        print('Fail to load.')
        return None

    return p_session


# p_session: Opened Tensorflow Session Object
# return: True(Success), False(Fail)
def carnet_close(p_session):
    if p_session is not None:
        try:
            p_session.close()
        except:
            return False

        return True
    return False


# p_session: Opened Tensorflow Session Object
# input_file_path_list: List of image input file name.
# batch_size: Inference batch size(Default 1)
# out_directory: If not None, result file will be saved in out_directory.
# return: Normalized segment map image list(Success), None(Failed)
def inference(p_session, input_file_path_list, batch_size=1, out_direcoty=None, threshold=10):
    try:
        input = tf.get_collection('input')[0]
        output = tf.get_collection('output')[0]
        reconstruction = tf.get_collection('reconstruction')[0]

        input_batch = zip(range(0, len(input_file_path_list), batch_size),
                       range(batch_size, len(input_file_path_list) + 1, batch_size))

        segment_map_list = None

        for t_s, t_e in input_batch:
            test_imgs = load_images(input_file_path_list[t_s:t_e])

            segment_map_list,  reconstruction_map_list = p_session.run([output, reconstruction], feed_dict={input: test_imgs})
            segment_map_list = np.where(segment_map_list > 0.5, segment_map_list, 0.0)

            if out_direcoty is not None:
                for i in range(batch_size):
                    src_rgb = cv2.resize(255 * test_imgs[i], dsize=(512, 512), interpolation=cv2.INTER_AREA)
                    src_rgb = src_rgb[128:128 + 256, 128:128 + 256]
                    src_rgb = np.uint8(src_rgb)
                    src_hsv = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2HSV)
                    src_hsv = np.float32(src_hsv)
                    src_h, src_s, src_v = cv2.split(src_hsv)

                    res_rgb = 255 * reconstruction_map_list[i]
                    res_rgb = np.uint8(res_rgb)
                    file_name = input_file_path_list[t_s + i].split('/')[-1]
                    #cv2.imwrite(out_direcoty + '/color_' + str(t_s + i) + '.jpg', res_bgr)
                    res_bgr = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)
                    #cv2.imwrite(out_direcoty + '/' + file_name, res_bgr)

                    res_rgb = res_rgb[128:128 + 256, 128:128 + 256]
                    res_hsv = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2HSV)
                    res_hsv = np.float32(res_hsv)
                    res_h, res_s, res_v = cv2.split(res_hsv)

                    canvas_size = 256
                    window_size = 128
                    stride = window_size // 2
                    num_window = 2 * (canvas_size // window_size) - 1

                    max_diff = 0
                    for row in range(num_window):
                        for col in range(num_window):
                            r = row * stride
                            c = col * stride
                            res_window_crop = res_h[r: r + window_size, c: c + window_size]
                            src_window_crop = src_h[r: r + window_size, c: c + window_size]
                            #print(str(r) + ':' + str(c) + ',' + str(r + window_size) + ':' + str(c + window_size))
                            dist_a = np.abs(res_window_crop - src_window_crop)
                            dist_b = 180 - dist_a
                            dist = np.array([dist_a, dist_b])  # (2, window_size, window_size)
                            dist = np.transpose(dist, (1, 2, 0))  # (window_size, window_size, 2)
                            dist = np.min(dist, axis=-1)  # (window_size, window_size)
                            dist_mean = np.mean(dist)
                            #dist = np.where(dist < dist_mean, 0, dist)
                            #dist_mean = 2 * np.mean(dist)
                            if dist_mean > max_diff:
                                max_diff = dist_mean

                    degree_dist = max_diff

                    test_img_rgb = np.uint8(255 * test_imgs[i])
                    test_img_bgr = cv2.cvtColor(test_img_rgb, cv2.COLOR_RGB2BGR)

                    test_seg_bgr = 255 * segment_map_list[i]

                    dist_a = np.abs(res_h - src_h)
                    dist_b = 180 - dist_a
                    dist = np.array([dist_a, dist_b]) # (2, 512, 512)
                    dist = np.transpose(dist, (1, 2, 0))  # (512, 512, 2)
                    dist = np.min(dist, axis=-1)  # (32, 32)
                    dist_mean = np.mean(dist)
                    dist_std = np.std(dist)
                    #dist = np.where(dist < dist_mean + 0.1 * dist_std, 0, dist)
                    dist = np.where(dist < dist_mean, 0, dist)
                    degree_dist += np.mean(dist)
                    #degree_dist = dist_mean

                    if degree_dist > threshold:
                        # Abnormal
                        cv2.imwrite(out_direcoty + '/abnormal/' + file_name, test_img_bgr)
                    else:
                        # Normal
                        cv2.imwrite(out_direcoty + '/normal/' + file_name, test_img_bgr)
                        cv2.imwrite(out_direcoty + '/segment/' + file_name, test_seg_bgr)

                    '''
                    # 10 bins
                    src_r_hist = cv2.calcHist(images=[src_rgb], channels=[0], mask=None, histSize=[10], ranges=[0, 256])
                    src_r_hist = np.float32(src_r_hist / 65536)
                    src_g_hist = cv2.calcHist(images=[src_rgb], channels=[1], mask=None, histSize=[10], ranges=[0, 256])
                    src_g_hist = np.float32(src_g_hist / 65536)
                    src_b_hist = cv2.calcHist(images=[src_rgb], channels=[2], mask=None, histSize=[10], ranges=[0, 256])
                    src_b_hist = np.float32(src_b_hist / 65536)

                    res_r_hist = cv2.calcHist(images=[res_rgb], channels=[0], mask=None, histSize=[10], ranges=[0, 256])
                    res_r_hist = np.float32(res_r_hist / 65536)
                    res_g_hist = cv2.calcHist(images=[res_rgb], channels=[1], mask=None, histSize=[10], ranges=[0, 256])
                    res_g_hist = np.float32(res_g_hist / 65536)
                    res_b_hist = cv2.calcHist(images=[res_rgb], channels=[2], mask=None, histSize=[10], ranges=[0, 256])
                    res_b_hist = np.float32(res_b_hist / 65536)

                    kld_r = (cv2.compareHist(res_r_hist, src_r_hist, cv2.HISTCMP_KL_DIV) +
                             cv2.compareHist(src_r_hist, res_r_hist, cv2.HISTCMP_KL_DIV)) / 2
                    kld_g = (cv2.compareHist(res_g_hist, src_g_hist, cv2.HISTCMP_KL_DIV) +
                             cv2.compareHist(src_g_hist, res_g_hist, cv2.HISTCMP_KL_DIV)) / 2
                    kld_b = (cv2.compareHist(res_b_hist, src_b_hist, cv2.HISTCMP_KL_DIV) +
                             cv2.compareHist(src_b_hist, res_b_hist, cv2.HISTCMP_KL_DIV)) / 2
                    kld = np.mean([kld_r, kld_g, kld_b])
                    '''
                    s = degree_dist

                    print('anomaly score of ' + file_name + ': ' + str(s))
    except Exception as e:
        print(e)
        return None

    return segment_map_list


# p_session: Opened Tensorflow Session Object
# input_file_path_list: List of image input file name.
# ensemble_file_path_list: List of ensemble result file name. Each image should be averaged and normalized.
# batch_size: Inference batch size(Default 1)
# out_directory: If not None, result file will be saved in out_directory.
# return: Normalized segment map image list(Success), None(Failed)
def ensemble_inference(p_session, input_file_path_list, ensemble_file_path_list=None, batch_size=1, out_direcoty=None):
    try:
        input = tf.get_collection('input')[0]
        output = tf.get_collection('output')[0]
        hint = tf.get_collection('hint')[0]

        input_batch = zip(range(0, len(input_file_path_list), batch_size),
                       range(batch_size, len(input_file_path_list) + 1, batch_size))

        segment_map_list = None

        for t_s, t_e in input_batch:
            test_imgs = load_images(input_file_path_list[t_s:t_e])

            if ensemble_file_path_list is None:
                (b, w, h, c) = np.shape(test_imgs)
                hint_imgs = np.zeros((b, w, h, 1))
            else:
                hint_imgs = load_images(ensemble_file_path_list[t_s:t_e], gray_scale=True)

            segment_map_list = p_session.run(output, feed_dict={input: test_imgs, hint: hint_imgs})
            segment_map_list = np.where(segment_map_list > 0.5, segment_map_list, 0.0)

            if out_direcoty is not None:
                for i in range(batch_size):
                    cv2.imwrite(out_direcoty + '/' + str(t_s + i) + '.jpg', 255 * segment_map_list[i])
    except:
        return None

    return segment_map_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, help='model check point file path', default='model/m.ckpt')
    parser.add_argument('--test_data', type=str, help='test data directory', default='data/test')
    parser.add_argument('--out_dir', type=str, help='output directory', default='imgs')
    parser.add_argument('--img_size', type=int, help='training image size', default=2048)
    parser.add_argument('--batch_size', type=int, help='Training batch size', default=16)

    args = parser.parse_args()

    input_width = args.img_size
    input_height = args.img_size
    batch_size = args.batch_size
    model_path = args.model_path
    test_data = args.test_data
    out_dir = args.out_dir

    # Inference all files in directory
    test_dir = test_data
    test_files = [os.path.join(test_dir, dentry).replace("\\", "/") for dentry in os.listdir(test_dir)]
    test_files = sorted(test_files)

    p_carnet_session = carnet_open(model_path)
    if inference(p_carnet_session, test_files,  batch_size=batch_size, out_direcoty=out_dir) is None:
        print('Inference Error')
    #inference(p_carnet_session, test_files, batch_size=batch_size, out_direcoty=out_dir)
    #ensemble_inference(p_carnet_session, test_files, batch_size=batch_size, out_direcoty=out_dir)
    carnet_close(p_carnet_session)
