import time, os
import cv2, math
import numpy as np
import pathlib
import math




RADIAN_PER_DEGREE = 0.0174532
debug = False


def crop_image(input_img, bbox, bbox_ratio=1., offset_x=0, offset_y=0):
    left = int(bbox_ratio * bbox[0])
    top = int(bbox_ratio * bbox[1])
    width = int(bbox_ratio * bbox[2])
    height = int(bbox_ratio * bbox[3])
    crop_img = input_img[top + offset_y:top + height + offset_y,
               left + offset_x:left + width + offset_x]
    return crop_img


def resize_normalize(img, normalize_width=1654):
    w = img.shape[1]
    h = img.shape[0]
    resize_ratio = normalize_width / w
    normalize_height = round(h * resize_ratio)
    resize_img = cv2.resize(img, (normalize_width, normalize_height), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('resize img', resize_img)
    # cv2.waitKey(0)
    return resize_ratio, resize_img


def draw_bboxes(img, bboxes, window_name='draw bboxes'):
    # e.g: bboxes= [(0,0),(0,5),(5,5),(5,0)]
    if len(img.shape) != 3:
        img_RGB = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_RGB = img
    color_red = (0, 0, 255)
    for idx, bbox in enumerate(bboxes):
        cv2.line(img_RGB, bbox[0], bbox[1], color=color_red, thickness=2)
        cv2.line(img_RGB, bbox[1], bbox[2], color=color_red, thickness=2)
        cv2.line(img_RGB, bbox[2], bbox[3], color=color_red, thickness=2)
        cv2.line(img_RGB, bbox[3], bbox[0], color=color_red, thickness=2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (bbox[0][0], bbox[0][1] - 5)

        # fontScale
        fontScale = 1.5

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.putText() method
        img_RGB = cv2.putText(img_RGB, str(idx), org, font,
                              fontScale, color, thickness, cv2.LINE_AA)

    img_RGB = cv2.resize(img_RGB, (int(img_RGB.shape[1] / 2), int(img_RGB.shape[0] / 2)))
    cv2.imshow(window_name, img_RGB)
    cv2.waitKey(0)


class Template_info:
    def __init__(self, name, template_path, field_bboxes, field_rois_extend=(1.0, 1.0),
                 field_search_areas=None,
                 confidence=0.7, scales=(0.9, 1.1, 0.1), rotations=(-2, 2, 2), normalize_width=1654):  # 1654
        self.name = name
        self.template_img = cv2.imread(template_path, 0)
        self.normalize_width = normalize_width
        self.resize_ratio, self.template_img = resize_normalize(self.template_img, normalize_width)
        self.template_width = self.template_img.shape[1]
        self.template_height = self.template_img.shape[0]
        self.confidence = confidence
        self.field_bboxes = field_bboxes
        self.field_rois_extend = field_rois_extend
        self.field_search_areas = field_search_areas
        self.field_locs = []
        self.list_field_samples = []
        for idx, bbox in enumerate(self.field_bboxes):
            bbox = self.resize_bbox(bbox, self.resize_ratio)

            field = dict()
            field['name'] = str(idx)
            field['loc'] = (bbox[0] + (bbox[2] - 1) / 2, bbox[1] + (bbox[3] - 1) / 2)
            self.field_locs.append(field['loc'])
            field['search_area'] = None
            if field_search_areas is not None:
                field['search_area'] = self.resize_bbox(field_search_areas[idx], self.resize_ratio)
            else:
                field['data'] = self.crop_image(self.template_img, bbox)
                # cv2.imwrite(field['name']+'.jpg', field['data'])
                field_w = max(field['data'].shape[1], 50)
                field_h = max(field['data'].shape[0], 50)
                extend_x = int(self.field_rois_extend[0] * field_w)
                extend_y = int(self.field_rois_extend[1] * field_h)
                left = max(int(field['loc'][0] - field_w / 2 - extend_x), 0)
                top = max(int(field['loc'][1] - field_h / 2 - extend_y), 0)
                right = min(int(field['loc'][0] + field_w / 2 + extend_x), self.template_width)
                bottom = min(int(field['loc'][1] + field_h / 2 + extend_y), self.template_height)
                width = right - left
                height = bottom - top
                field['search_area'] = [left, top, width, height]

            self.createSamples(field, scales, rotations)
            self.list_field_samples.append(field)

    def resize_bbox(self, bbox, resize_ratio):
        for i in range(len(bbox)):
            bbox[i] = round(bbox[i] * resize_ratio)
        return bbox

    def crop_image(self, input_img, bbox, offset_x=0, offset_y=0):
        # logger.info('crop')
        crop_img = input_img[bbox[1] + offset_y:bbox[1] + bbox[3] + offset_y,
                   bbox[0] + offset_x:bbox[0] + bbox[2] + offset_x]
        return crop_img

    def createSamples(self, field, scales, rotations):
        # logger.info('Add_template', field['name'])
        list_scales = []
        list_rotations = []

        num_scales = round((scales[1] - scales[0]) / scales[2]) + 1
        num_rotations = round((rotations[1] - rotations[0]) / rotations[2]) + 1
        for i in range(num_scales):
            list_scales.append(round(scales[0] + i * scales[2], 4))
        for i in range(num_rotations):
            list_rotations.append(round(rotations[0] + i * rotations[2], 4))

        field['list_samples'] = []
        field_data = field['data']
        w = field_data.shape[1]
        h = field_data.shape[0]
        bgr_val = int((int(field_data[0][0]) + int(field_data[0][w - 1]) + int(
            field_data[h - 1][w - 1]) + int(field_data[h - 1][0])) / 4)
        for rotation in list_rotations:
            abs_rotation = abs(rotation)
            if (w < h):
                if (abs_rotation <= 45):
                    sa = math.sin(abs_rotation * RADIAN_PER_DEGREE)
                    ca = math.cos(abs_rotation * RADIAN_PER_DEGREE)
                    newHeight = (int)((h - w * sa) / ca)
                    # newHeight = newHeight - ((h - newHeight) % 2)
                    szOutput = (w, newHeight)
                else:
                    sa = math.sin((90 - abs_rotation) * RADIAN_PER_DEGREE)
                    ca = math.cos((90 - abs_rotation) * RADIAN_PER_DEGREE)
                    newWidth = (int)((h - w * sa) / ca)
                    # newWidth = newWidth - ((w - newWidth) % 2)
                    szOutput = (newWidth, w)
            else:
                if (abs_rotation <= 45):
                    sa = math.sin(abs_rotation * RADIAN_PER_DEGREE)
                    ca = math.cos(abs_rotation * RADIAN_PER_DEGREE)
                    newWidth = (int)((w - h * sa) / ca)
                    # newWidth = newWidth - ((w - newWidth) % 2)
                    szOutput = (newWidth, h)
                else:
                    sa = math.sin((90 - rotation) * RADIAN_PER_DEGREE)
                    ca = math.cos((90 - rotation) * RADIAN_PER_DEGREE)
                    newHeight = (int)((w - h * sa) / ca)
                    # newHeight = newHeight - ((h - newHeight) % 2)
                    szOutput = (h, newHeight)

            (h, w) = field_data.shape[:2]
            (cX, cY) = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D((cX, cY), -rotation, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY
            rotated = cv2.warpAffine(field_data, M, (nW, nH), borderValue=bgr_val)

            # (h_rot, w_rot) = rotated.shape[:2]
            # (cX_rot, cY_rot) = (w_rot // 2, h_rot // 2)
            # pt1=(int(cX_rot-3), int(cY_rot-3))
            # pt2=(int(cX_rot+3), int(cY_rot+3))
            # pt3=(int(cX_rot-3), int(cY_rot+3))
            # pt4=(int(cX_rot+3), int(cY_rot-3))
            # cv2.line(rotated,pt1,pt2,color=255)
            # cv2.line(rotated,pt3,pt4,color=255)

            offset_X = int((nW - szOutput[0]) / 2)
            offset_Y = int((nH - szOutput[1]) / 2)

            crop_rotated = rotated[offset_Y:nH - offset_Y - 1, offset_X:nW - offset_X - 1]
            crop_w = crop_rotated.shape[1]
            crop_h = crop_rotated.shape[0]
            # rint('origin size', crop_w, crop_h)

            for scale in list_scales:
                temp = dict()
                temp['rotation'] = rotation
                temp['scale'] = scale
                # logger.info('scale', scale, ', rotation', rotation)
                crop_rotate_resize = cv2.resize(crop_rotated, (int(scale * crop_w), int(scale * crop_h)))
                # logger.info('resize size', int(scale * crop_w), int(scale * crop_h))
                temp['data'] = crop_rotate_resize
                if debug:
                    cv2.imshow('result', crop_rotated)
                    cv2.imshow('result_crop', crop_rotate_resize)
                    ch = cv2.waitKey(0)
                    if ch == 27:
                        cv2.imwrite('result.jpg', crop_rotated)
                        break
                field['list_samples'].append(temp)

    def draw_template(self, src_img=None, crop=False, crop_dir=''):
        list_bboxes = []
        for idx, bbox in enumerate(self.field_bboxes):
            left = bbox[0]
            top = bbox[1]
            right = bbox[0] + bbox[2]
            bottom = bbox[1] + bbox[3]
            if crop:
                crop_img = crop_image(self.template_img, bbox)
                cv2.imwrite(os.path.join(crop_dir, self.name + '_field_' + str(idx) + '.jpg'), crop_img)
            bboxes = [(left, top), (right, top), (right, bottom), (left, bottom)]
            list_bboxes.append(bboxes)
        if src_img is None:
            draw_bboxes(self.template_img, list_bboxes)
        else:
            draw_bboxes(src_img, list_bboxes, window_name='new')

    def get_template_img(self):
        return self.template_img


class MatchingTemplate:
    def __init__(self, initTemplate=False):
        self.template_dir = ''
        self.template_names = []
        self.template_list = []
        self.template_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), 'templates')
        if initTemplate:
            self.initTemplate()
        self.matching_results = []
        self.activate_template = ''

    def initTemplate(self, template_dir=None, list_template_name=[]):
        kk=1

    def add_template(self, template_name, template_path, field_bboxes, field_rois_extend=(1.0, 1.0),
                     field_search_areas=None,
                     confidence=0.7, scales=(0.9, 1.1, 0.1), rotations=(-2, 2, 2), normalize_width=1654):
        if not os.path.exists(template_path):
            print('MatchingTemplate. No template path:', template_path)
            return
        print('MatchingTemplate. Init template','['+ str(template_name)+']')
        temp = Template_info(template_name, template_path, field_bboxes, field_rois_extend,
                             field_search_areas,
                             confidence, scales, rotations, normalize_width=normalize_width)
        self.template_list.append(temp)

    def clear_template(self):
        self.template_list.clear()

    def check_template(self, template_name):
        template_data = None
        for template in self.template_list:
            if template.name == template_name:
                self.activate_template = template_name
                template_data = template
                break
        if template_data is None:
            print('MatchingTemplate. No template name', template_name)
            # logger.info('Cannot find template', template_name, 'in database')
        return template_data

    def draw_template(self, template_name, src_img=None, crop=False, crop_dir=''):
        template_data = self.check_template(template_name)
        if template_data is None:
            return
        template_data.draw_template(src_img, crop=crop, crop_dir=crop_dir)

    def get_matching_result(self, final_locx, final_locy, final_sample):
        x0 = final_locx
        y0 = final_locy

        x1 = x0 - (final_sample['data'].shape[1] / 2) * final_sample['scale']
        y1 = y0 - (final_sample['data'].shape[0] / 2) * final_sample['scale']
        x2 = x0 + (final_sample['data'].shape[1] / 2) * final_sample['scale']
        y2 = y0 + (final_sample['data'].shape[0] / 2) * final_sample['scale']

        ##
        ca = math.cos(final_sample['rotation'] * RADIAN_PER_DEGREE)
        sa = math.sin(final_sample['rotation'] * RADIAN_PER_DEGREE)
        rx1 = round((x0 + (x1 - x0) * ca - (y1 - y0) * sa))
        ry1 = round((y0 + (x1 - x0) * sa + (y1 - y0) * ca))
        rx2 = round((x0 + (x2 - x0) * ca - (y1 - y0) * sa))
        ry2 = round((y0 + (x2 - x0) * sa + (y1 - y0) * ca))
        rx3 = round((x0 + (x2 - x0) * ca - (y2 - y0) * sa))
        ry3 = round((y0 + (x2 - x0) * sa + (y2 - y0) * ca))
        rx4 = round((x0 + (x1 - x0) * ca - (y2 - y0) * sa))
        ry4 = round((y0 + (x1 - x0) * sa + (y2 - y0) * ca))
        return [(rx1, ry1), (rx2, ry2), (rx3, ry3), (rx4, ry4)]

    def find_field(self, input_img, field, thres=0.3, fast=True, method='cv2.TM_CCORR_NORMED'):
        max_conf = 0
        final_locx, final_locy = -1, -1
        final_sample = None

        process_img = input_img.copy()
        if len(input_img.shape) == 3:  # BGR
            process_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        if fast:
            left = field['search_area'][0]
            top = field['search_area'][1]
            right = field['search_area'][0] + field['search_area'][2]
            bottom = field['search_area'][1] + field['search_area'][3]
            process_img = process_img[top:bottom, left:right]
        try:
            if not os.path.exists(os.path.join(self.template_dir, 'crop')):
                os.makedirs(os.path.join(self.template_dir, 'crop'))
            # print('MatchingTemplate. find_field. Write process image to',
            #       os.path.join(self.template_dir, 'crop', self.activate_template + '_' + field['name'] + '.jpg'))
            # cv2.imwrite(os.path.join(self.template_dir, 'crop', self.activate_template + '_' + field['name'] + '.jpg'),
            #             process_img)
        except:
            print("Except find field : make_dir func")
            pass
        for sample in field['list_samples']:
            sample_data = sample['data']
            res = cv2.matchTemplate(process_img, sample_data, 5)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            logger.info('Score:', round(max_val, 4), 'Scale:', sample['scale'], 'Angle:', sample['rotation'],
                        max_loc[0] + sample_data.shape[1] / 2, max_loc[1] + sample_data.shape[0] / 2)
            if max_val > max_conf:
                max_conf = max_val
                final_locx, final_locy = max_loc[0] + sample_data.shape[1] / 2, max_loc[1] + sample_data.shape[0] / 2
                final_sample = sample
        if fast:
            final_locx, final_locy = final_locx + field['search_area'][0], final_locy + field['search_area'][1]

        if max_conf >= thres:
            print('Score:', round(max_conf, 4), 'Scale:', final_sample['scale'], 'Angle:', final_sample['rotation'],
                  'Location:', final_locx, final_locy)
        else:  # cannot find field
            print('MatchingTemplate. find_field. Cannot find field! Max score:', round(max_conf, 4))
            return 0, -1, -1

        self.matching_results = self.get_matching_result(final_locx, final_locy, final_sample)
        # draw_bboxes(input_img, [self.matching_results], field['name'])

        return max_conf, final_locx, final_locy

    def find_template(self, template_name, src_img, fast=True, threshold=0.7):  # src_img is cv2 image
        # logger.info('\nCalib template', template_name)
        template_data = self.check_template(template_name)
        if template_data is None:
            return

        resize_ratio, src_img = resize_normalize(src_img, template_data.normalize_width)
        gray_img = src_img
        if len(src_img.shape) == 3:  # BGR
            gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        list_pts = []

        for idx, field in enumerate(template_data.list_field_samples):
            # logger.info(field['name'])
            conf, loc_x, loc_y = self.find_field(gray_img, field, fast=fast, thres=template_data.confidence)
            if conf > threshold:
                list_pts.append((loc_x, loc_y))
        return list_pts

    def calib_template(self, template_name, src_img, fast=True, simi_triangle_thres=4,
                       simi_line_thres=3):  # src_img is cv2 image
        template_data = self.check_template(template_name)
        if template_data is None:
            return False, None
        print('MatchingTemplate. Calib template', template_name, ', width', template_data.template_width, ', height',
              template_data.template_height)

        # src_img = cv2.resize(src_img, (template_data.template_width, template_data.template_height))
        resize_ratio, src_img = resize_normalize(src_img, template_data.normalize_width)
        gray_img = src_img
        if len(src_img.shape) == 3:  # BGR
            gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        list_pts = []

        for idx, field in enumerate(template_data.list_field_samples):
            # logger.info(field['name'])
            import time
            begin = time.time()
            conf, loc_x, loc_y = self.find_field(gray_img, field, fast=fast, thres=template_data.confidence)
            end = time.time()
            print('calib_template. find field time:', 1000 * (end - begin), 'ms')
            list_pts.append((loc_x, loc_y))

        src_pts = np.asarray(list_pts, dtype=np.float32)
        dst_pts = np.asarray(template_data.field_locs, dtype=np.float32)
        trans_img = src_img
        calib_success = True
        if len(src_pts) == 2:  # affine transformation with 1 synthetic point
            calib_success = check_angle_between_2_lines(template_data.field_locs, list_pts, diff_thres=simi_line_thres)
            inter_pts = (
                list_pts[0][0] + list_pts[0][1] - list_pts[1][1], list_pts[0][1] + list_pts[1][0] - list_pts[0][0])
            list_pts.append(inter_pts)
            inter_field_pts = [template_data.field_locs[0], template_data.field_locs[1]]
            inter_field_pts.append(
                (template_data.field_locs[0][0] + template_data.field_locs[0][1] - template_data.field_locs[1][1],
                 template_data.field_locs[0][1] + template_data.field_locs[1][0] - template_data.field_locs[0][0]))

            src_pts = np.asarray(list_pts, dtype=np.float32)
            dst_pts = np.asarray(inter_field_pts, dtype=np.float32)
            # print('dst_pts', dst_pts)
            affine_trans = cv2.getAffineTransform(src_pts, dst_pts)
            trans_img = cv2.warpAffine(src_img, affine_trans,
                                       (template_data.template_width, template_data.template_height))
        elif len(src_pts) == 3:  # affine transformation
            calib_success = check_similar_triangle(template_data.field_locs, list_pts, diff_thres=simi_triangle_thres)
            affine_trans = cv2.getAffineTransform(src_pts, dst_pts)
            trans_img = cv2.warpAffine(src_img, affine_trans,
                                       (template_data.template_width, template_data.template_height))
        elif len(src_pts) > 3:  # perspective transformation
            perspective_trans, status = cv2.findHomography(src_pts, dst_pts)
            w, h = template_data.template_width, template_data.template_height
            trans_img = cv2.warpPerspective(src_img, perspective_trans, (w, h))
        else:
            kk = 1
        return calib_success, trans_img

    def calib_template_2(self, template_name, src_img, fast=True, simi_triangle_thres=4,
                         simi_line_thres=3):  # src_img is cv2 image
        template_data = self.check_template(template_name)
        if template_data is None:
            return False, None, None
        print('MatchingTemplate. Calib template', template_name, ', width', template_data.template_width, ', height',
              template_data.template_height)

        # src_img = cv2.resize(src_img, (template_data.template_width, template_data.template_height))
        resize_ratio, src_img = resize_normalize(src_img, template_data.normalize_width)
        gray_img = src_img
        if len(src_img.shape) == 3:  # BGR
            gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        list_pts = []
        calib_success = True
        for idx, field in enumerate(template_data.list_field_samples):
            # logger.info(field['name'])
            import time
            begin = time.time()
            conf, loc_x, loc_y = self.find_field(gray_img, field, fast=fast, thres=template_data.confidence)
            end = time.time()
            print('calib_template. find field time:', 1000 * (end - begin), 'ms')
            list_pts.append((loc_x, loc_y))

        src_pts = np.asarray(list_pts, dtype=np.float32)
        dst_pts = np.asarray(template_data.field_locs, dtype=np.float32)
        trans_img = src_img

        if len(src_pts) == 2:  # affine transformation with 1 synthetic point
            calib_success = check_angle_between_2_lines(template_data.field_locs, list_pts, diff_thres=simi_line_thres)
            inter_pts = (
                list_pts[0][0] + list_pts[0][1] - list_pts[1][1], list_pts[0][1] + list_pts[1][0] - list_pts[0][0])
            list_pts.append(inter_pts)
            inter_field_pts = [template_data.field_locs[0], template_data.field_locs[1]]
            inter_field_pts.append(
                (template_data.field_locs[0][0] + template_data.field_locs[0][1] - template_data.field_locs[1][1],
                 template_data.field_locs[0][1] + template_data.field_locs[1][0] - template_data.field_locs[0][0]))

            src_pts = np.asarray(list_pts, dtype=np.float32)
            dst_pts = np.asarray(inter_field_pts, dtype=np.float32)
            # print('dst_pts', dst_pts)
            affine_trans = cv2.getAffineTransform(src_pts, dst_pts)
            trans_img = cv2.warpAffine(src_img, affine_trans,
                                       (template_data.template_width, template_data.template_height),
                                       borderValue=(255, 255, 255))
        elif len(src_pts) == 3:  # affine transformation
            calib_success = check_similar_triangle(template_data.field_locs, list_pts, diff_thres=simi_triangle_thres)
            affine_trans = cv2.getAffineTransform(src_pts, dst_pts)
            trans_img = cv2.warpAffine(src_img, affine_trans,
                                       (template_data.template_width, template_data.template_height))
        elif len(src_pts) > 3:  # perspective transformation
            perspective_trans, status = cv2.findHomography(src_pts, dst_pts)
            w, h = template_data.template_width, template_data.template_height
            trans_img = cv2.warpPerspective(src_img, perspective_trans, (w, h))
        else:
            kk = 1
        return calib_success, trans_img, dst_pts

    def crop_image(self, input_img, bbox, offset_x=0, offset_y=0):
        logger.info('crop')
        crop_img = input_img[bbox[1] + offset_y:bbox[1] + bbox[3] + offset_y,
                   bbox[0] + offset_x:bbox[0] + bbox[2] + offset_x]
        return crop_img


def test_calib_multi(template_name, src_img_dir):
    list_files = get_list_file_in_folder(src_img_dir)
    for idx, f in enumerate(list_files):
        print(idx, f)
        test_calib(template_name, os.path.join(src_img_dir, f))


def test_calib(template_name, src_img_path):
    src_img = cv2.imread(src_img_path)
    begin_init = time.time()

    match = MatchingTemplate(initTemplate=True)
    # match.add_template(template_name=template_name,
    #                    template_path='C:/Users/titik/Desktop/idcard_2June/test_MireaAsset/contract.JPG',
    #                    field_bboxes=[[184, 1256, 242, 142]],
    #                    field_rois_extend = (10.0,0.3),
    #                    field_search_areas=None,
    #                    # confidence=0.7,
    #                    # scales=(0.95, 1.05, 0.05),
    #                    # rotations=(-1, 1, 1))
    #                    confidence=0.2,
    #                    scales=(1.0, 1.0, 0.1),
    #                    rotations=(0, 0, 1))
    end_init = time.time()
    logger.info('Time init:', end_init - begin_init, 'seconds')
    # match.draw_template(template_name)
    begin = time.time()
    calib_success, calib_img = match.calib_template(template_name, src_img, fast=True)

    # base_name = os.path.basename(src_img_path)
    # cv2.imwrite(os.path.join(output_dir, base_name.replace('.jpg', '_trans.jpg')), calib_img)
    end = time.time()
    print('Time:', end - begin, 'seconds')
    logger.info('Time:', end - begin, 'seconds')

    debug = True
    if debug:
        # src_img_with_box = visualize_boxes('/home/aicr/cuongnd/text_recognition/data/SDV_invoices_mod/006.txt', src_img,
        #                                    debug=False, offset_x=-20, offset_y=-20)
        # src_img_with_box = cv2.resize(src_img, (int(src_img.shape[1] / 2), int(src_img.shape[0] / 2)))
        # cv2.imshow('src with boxes', src_img_with_box)
        # trans_img_with_box = visualize_boxes('/home/aicr/cuongnd/text_recognition/data/SDV_invoices_mod/006.txt',
        #                                      calib_img, debug=False, offset_x=-20, offset_y=-20)
        trans_img_with_box = cv2.resize(calib_img, (int(calib_img.shape[1] / 2), int(calib_img.shape[0] / 2)))
        trans_img_with_box = cv2.resize(calib_img, (calib_img.shape[1], calib_img.shape[0]))
        cv2.imshow('transform_with_boxes', trans_img_with_box)
        base_name = os.path.basename(src_img_path)
        # cv2.imwrite(src_img_path.replace(base_name, 'transform/' + base_name.replace('.jpg', '_trans.jpg')),
        #            trans_img_with_box)
        cv2.waitKey(0)
    return calib_img


def get_list_file_in_folder(dir, ext=['jpg', 'png', 'JPG', 'PNG']):
    included_extensions = ext
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names




def getAngle(a, b, c):
    ang = math.fabs(math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])))
    return ang + 360 if ang < 0 else ang


def simi_aaa(a1, a2, diff_thres):
    a1 = [float(i) for i in a1]
    a2 = [float(i) for i in a2]
    a1.sort()
    a2.sort()

    # Check for AAA
    diff_1 = math.fabs(a1[0] - a2[0])
    diff_2 = math.fabs(a1[1] - a2[1])
    diff_3 = math.fabs(a1[2] - a2[2])
    max_diff = max(diff_1, max(diff_2, diff_3))
    if diff_1 < diff_thres and diff_2 < diff_thres and diff_3 < diff_thres:
        return max_diff, True
    return max_diff, False


def check_similar_triangle(list_pts1, list_pts2, diff_thres =4):
    list_ang1 = [getAngle(list_pts1[0], list_pts1[1], list_pts1[2]),
                 getAngle(list_pts1[1], list_pts1[2], list_pts1[0]),
                 getAngle(list_pts1[2], list_pts1[0], list_pts1[1])]

    list_ang2 = [getAngle(list_pts2[0], list_pts2[1], list_pts2[2]),
                 getAngle(list_pts2[1], list_pts2[2], list_pts2[0]),
                 getAngle(list_pts2[2], list_pts2[0], list_pts2[1])]

    max_diff, is_similar=simi_aaa(list_ang1, list_ang2, diff_thres)
    # print('check_similar_triangle. max diff:',max_diff)
    return is_similar


def dot_product(vA, vB):
    return vA[0] * vB[0] + vA[1] * vB[1]


def check_angle_between_2_lines(lineA, lineB, diff_thres=2):
    # Get nicer vector form
    try:
        vA = [(lineA[0][0] - lineA[1][0]), (lineA[0][1] - lineA[1][1])]
        vB = [(lineB[0][0] - lineB[1][0]), (lineB[0][1] - lineB[1][1])]
        # Get dot prod
        dot_prod = dot_product(vA, vB)
        # Get magnitudes
        magA = dot_product(vA, vA) ** 0.5
        magB = dot_product(vB, vB) ** 0.5
        # Get cosine value
        cos_ = dot_prod / magA / magB
        # Get angle in radians and then convert to degrees
        angle = math.acos(dot_prod / magB / magA)
        # Basically doing angle <- angle mod 360
        ang_deg = math.degrees(angle) % 360

        if ang_deg >= 180:
            ang_deg = 360 - ang_deg
        if ang_deg > 90:
            ang_deg = 180 - ang_deg
        print ('check_angle_between_2_lines. angle:', ang_deg)

        if ang_deg<diff_thres:
            return True
        else:
            return False
    except:
        print ('check_angle_between_2_lines. something wrong')
        return False

