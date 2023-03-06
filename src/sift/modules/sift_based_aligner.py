import os
import numpy as np
import cv2
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.geometry import Polygon
import time

from ..utils.image_calib import check_similar_triangle, check_angle_between_2_lines
from ..utils.common import read_json


##
# @brief  Document classifier based on SIFT-based template matching
# @note This classifier can support varying illumination, varying out-of-plane rotation angles (up to roughly 60 degrees compared with the template image's orientation), and invariant to scale. These constraints are given in [this slides](http://vision.stanford.edu/teaching/cs231a_autumn1112/lecture/lecture12_SIFT_single_obj_recog_cs231a_marked.pdf)


class SIFTBasedAligner:
    ##
    # @brief  Initializer
    #
    # @param template_info (dict): Mapping from class names to template info lists. The format of this dict is given as
    #
    #   ```
    # 	{
    # 		"doc_type": {
    # 			"image_path": str, # path to template image
    # 			"anchors": [
    # 				{
    # 					"id": str, # anchor ID in template image
    # 					"position": {"top": int, "left": int},
    # 					"size": {"width": , "height": float}
    # 				},
    # 				{
    # 					"id": str, # anchor ID in template image
    # 					"position": {"top": int, "left": int},
    # 					"size": {"width": , "height": float}
    # 				},
    #				...
    # 			]
    # 		},
    # 		"doc_type": {
    # 			"image_path": str, # path to template image
    # 			"anchors": [
    # 				{
    # 					"id": str, # anchor ID in template image
    # 					"position": {"top": int, "left": int},
    # 					"size": {"width": , "height": float}
    # 				},
    # 				{
    # 					"id": str, # anchor ID in template image
    # 					"position": {"top": int, "left": int},
    # 					"size": {"width": , "height": float}
    # 				},
    #				...
    # 			]
    # 		},
    # 		...
    # 	}
    #
    #   ```
    #
    # @param min_match_count (int): Minimum number of matched points for a template to be considered "FOUND" in the input image
    # @param flann_based_matcher_config (dict): Configurations for cv2.FlannBasedMatcher, i.e. refer to the this class for more details
    # @param matching_topk (int): This must be 2
    # @param distance_threshold (float): Upper threshold for top-1-distance-over-top-2-distance ratio after kNN matching
    # @param ransac_threshold (float): RANSAC threshold for locating the template within the input image (if found)
    # @param valid_size_ratio_margin (float): Valid max-size-to-min-size ratio margin of the min-area surrounding box of the found template in the image, within which the found template is considered valid
    # @param valid_area_threshold (float): Valid area threhsold, above which the found template is considered valid
    # @param image_max_size (int): Maximum size for the input image. None if no limit
    # @param similar_triangle_threshold (float): Threshold, above which two triangles are considered non-similar
    # @param roi_to_template_box_ratio (float): Ratio to scale the template anchor to estimate the ROI, within which the template anchor is likely to exist
    def __init__(
        self, min_match_count, flann_based_matcher_config,
        matching_topk, distance_threshold, ransac_threshold,
        valid_size_ratio_margin, valid_area_threshold, image_max_size,
        similar_triangle_threshold, roi_to_template_box_ratio,
        default_image_size
    ):
        if matching_topk != 2:
            raise NotImplementedError("Only support matching_topk =2, found ", matching_topk)
        self._loaded_template = dict()  # store loaded template to improve efficiency

        # SIFT feature extractor
        # self.sift = cv2.xfeatures2d.SIFT_create()
        self.sift = cv2.SIFT_create()

        # kNN feature matcher
        self.matcher = cv2.FlannBasedMatcher(flann_based_matcher_config, {})

        # other arguments
        self.image_max_size = image_max_size
        self.min_match_count = min_match_count
        self.matching_topk = matching_topk
        self.distance_threshold = distance_threshold
        self.ransac_threshold = ransac_threshold
        self.roi_to_template_box_ratio = roi_to_template_box_ratio
        self.default_image_size = default_image_size

        # validity thresholds
        self.valid_size_ratio_interval = [
            1 - valid_size_ratio_margin, 1 + valid_size_ratio_margin
        ]
        self.valid_area_threshold = valid_area_threshold
        self.similar_triangle_threshold = similar_triangle_threshold

    def _extract_template_info(self, template_info):
        r"""Load template images from paths and extract features"""

        if template_info["name"] in self._loaded_template:
            return self._loaded_template[template_info["name"]]
        template_anchors = []
        template_features = []
        template_metadata = []
        if not os.path.exists(template_info["image_path"]):
            raise FileNotFoundError(template_info["image_path"])
        template_image = cv2.imread(template_info["image_path"])
        for anchor in template_info["anchors"]:
            # extract anchor
            x1, y1, x2, y2 = [int(a) for a in anchor]
            template_anchor = template_image[y1:y2, x1:x2]
            # print(template_image)
            # print(doc_type, anchor, template_image.shape, template_anchor.shape)
            # read template and extract features
            template_anchor = cv2.cvtColor(
                template_anchor, cv2.COLOR_BGR2GRAY
            )
            template_kpts, template_desc = self.sift.detectAndCompute(
                template_anchor, None
            )

            # append to dict
            max_size = np.max(template_anchor.shape[:2])
            min_size = np.min(template_anchor.shape[:2])

            template_anchors.append(template_anchor)
            template_features.append({
                "kpts": template_kpts,
                "desc": template_desc,
                "ratio": max_size / min_size
            })
            template_metadata.append({"box": [x1, y1, x2, y2]})
        self._loaded_template[template_info["name"]] = (
            template_image, template_anchors,
            template_features, template_metadata
        )
        return self._loaded_template[template_info["name"]]

    # def _load_template(self, template_info):
    #     r"""Load template images from paths and extract features"""
    #     template_images, template_anchors = {}, {}
    #     template_features, template_metadata = {}, {}
    #     for doc_type in template_info:
    #         template_anchors[doc_type] = []
    #         template_features[doc_type] = []
    #         template_metadata[doc_type] = []
    #         assert os.path.exists(template_info[doc_type]["image_path"]), print(template_info[doc_type]["image_path"])
    #         template_image = cv2.imread(template_info[doc_type]["image_path"])
    #         for anchor in template_info[doc_type]["anchors"]:
    #             # extract anchor
    #             x, y = anchor["position"]["left"], anchor["position"]["top"]
    #             w, h = anchor["size"]["width"], anchor["size"]["height"]
    #             x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    #             template_anchor = template_image[y1:y2, x1:x2]

    #             # read template and extract features
    #             template_anchor = cv2.cvtColor(
    #                 template_anchor, cv2.COLOR_BGR2GRAY
    #             )
    #             template_kpts, template_desc = self.sift.detectAndCompute(
    #                 template_anchor, None
    #             )

    #             # append to dict
    #             max_size = np.max(template_anchor.shape[:2])
    #             min_size = np.min(template_anchor.shape[:2])

    #             template_images[doc_type] = template_image
    #             template_anchors[doc_type].append(template_anchor)
    #             template_features[doc_type].append({
    #                 "kpts": template_kpts,
    #                 "desc": template_desc,
    #                 "ratio": max_size / min_size
    #             })
    #             template_metadata[doc_type].append({"box": [x1, y1, x2, y2]})

    #     return (
    #         template_images, template_anchors,
    #         template_features, template_metadata
    #     )

    ##
    # @brief  Run image transformer
    #
    # @param images: Refer to interface
    # @param metadata: Refer to interface. Each metadata dict has format
    #
    #   {
    #       "doc_type": str # document type of each image
    #   }
    #
    # @return   transformed_images: Refer to interface
    def run(self, image: np.ndarray, bboxes: list[list], template_infor: dict):
        # load templates
        (
            template_image, template_anchor,
            template_feature, template_metadata
        ) = self._extract_template_info(template_infor)
        gray_image = self._preprocess(image)
        # match against all templates of the given doc type
        anchor_centers, anchor_locations = [], []
        found_centers, found_locations = [], []
        for template_anchor_, template_feature_, template_metadata_ in zip(
                template_anchor, template_feature, template_metadata):
            (
                anchor_center, anchor_location,
                found_center, found_location
            ) = self._find_template(
                gray_image, template_image, template_anchor_,
                template_feature_, template_metadata_
            )
            if found_location is not None:
                anchor_centers.append(anchor_center)
                anchor_locations.append(anchor_location)
                found_centers.append(found_center)
                found_locations.append(found_location)

        if len(found_locations) == 0:
            print("len(found_locations) == 0")
            return None, None
        else:
            found_locations = np.concatenate(found_locations, axis=0)
            anchor_locations = np.concatenate(anchor_locations, axis=0)

        # check calibration ability
        calib_success = False
        if len(found_centers) < 3:
            # print("len(found_centers) < 3: ", len(found_centers), found_centers)
            # print("len(found_locations) < 3: ", len(found_locations), found_locations)
            return None, None
        # else:
            # print("len(found_centers) >= 3: ", len(found_centers), found_centers)
            # for center in found_centers:
            #     cv2.circle(image, tuple([int(x) for x in center]), 10, (255, 0, 0), -1)

            # cv2.imwrite("img_centers.jpg", image)
            # # print("len(found_locations) >= 3: ", len(found_locations), found_locations)
            # calib_success = check_similar_triangle(
            #     anchor_centers, found_centers,
            #     diff_thres=self.similar_triangle_threshold
            # )

        # align image
        # TODO: calib_success = False even when calib is successful
        calib_success = True
        if calib_success:
            perspective_trans, status = cv2.findHomography(found_locations, anchor_locations)
            transformed_image = cv2.warpPerspective(
                image, perspective_trans, (template_image.shape[1],
                                           template_image.shape[0]))
            transformed_boxes_in_img = self._transform_bboxes(bboxes, perspective_trans)
            return transformed_image, transformed_boxes_in_img
        else:
            print("calib_success is False")
            return None, None

    def _transform_bboxes(self, bboxes: list[list], perspective_trans) -> list[list]:
        bboxes = np.array(bboxes).astype(np.float32)
        bboxes, confs = (bboxes[:, : 4], bboxes[:, 4]) if bboxes.shape[1] == 5 else (bboxes, None)
        bboxes = bboxes.reshape(-1, 1, 2)
        transformed_boxes_in_img = cv2.perspectiveTransform(bboxes, perspective_trans)
        transformed_boxes_in_img = transformed_boxes_in_img.reshape(-1, 4)
        if confs is not None:
            transformed_boxes_in_img = np.hstack((transformed_boxes_in_img, confs.reshape(-1, 1)))
        return transformed_boxes_in_img.tolist()

    def _preprocess(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image

    def _find_template(
        self, gray_image, gray_template, gray_template_anchor,
        template_feature, template_metadata
    ):
        r"""Find templates and return template center"""
        # parser inputs
        template_kpts = template_feature["kpts"]
        template_desc = template_feature["desc"]
        template_box = template_metadata["box"]

        # resize ROI
        gray_image, shift, scale = self._crop_roi_and_resize(
            gray_image, gray_template, template_box
        )

        # extract features
        image_kpts, image_desc = self.sift.detectAndCompute(gray_image, None)
        # if image_desc is None:
        #     print("Error matching")
        #     return None, None, None, None
        # print(image_desc)
        try:
            # knnMatch to get top-K then sort by their distance
            matches = self.matcher.knnMatch(template_desc, image_desc, self.matching_topk)
        except Exception as err:
            print(err)
            return None, None, None, None
        matches = sorted(matches, key=lambda x: x[0].distance)

        # ratio test, to get good matches.
        # idea: good matches should uniquely match each other, i.e. top-1 and top-2 distances are much difference
        good = [
            m1 for (m1, m2) in matches
            if m1.distance < self.distance_threshold * m2.distance
        ]

        # find homography matrix
        if len(good) > self.min_match_count:
            # (queryIndex for the small object, trainIndex for the scene )
            src_pts = np.float32([
                template_kpts[m.queryIdx].pt for m in good
            ]).reshape(-1, 1, 2)
            dst_pts = np.float32([
                image_kpts[m.trainIdx].pt for m in good
            ]).reshape(-1, 1, 2)

            # find homography matrix in cv2.RANSAC using good match points
            M, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold
            )
            if M is not None:
                # get template center in original image
                h, w = gray_template_anchor.shape[:2]
                pts = np.float32([
                    [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]
                ]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                # get convex hull of the match and its min-area surrounding box
                hull = ConvexHull(dst[:, 0, :]).vertices
                hull = dst[hull][:, 0, :]
                hull_rect = cv2.minAreaRect(hull[:, None, :])
                hull_box = cv2.boxPoints(hull_rect)

                # compute sizes of the hull box
                hull_box_size = (
                    np.sqrt(np.sum((hull_box[0] - hull_box[1])**2, axis=-1)),
                    np.sqrt(np.sum((hull_box[1] - hull_box[2])**2, axis=-1))
                )

                # verify max-size-over-min-size ratio
                hull_box_ratio = np.max(hull_box_size) / np.min(hull_box_size)
                template_ratio = template_feature["ratio"]
                is_valid_ratio = (hull_box_ratio > self.valid_size_ratio_interval[0] * template_ratio) and (
                    hull_box_ratio < self.valid_size_ratio_interval[1] * template_ratio)

                # verify hull-area-to-hull-box-area ratio
                hull_area = Polygon(hull).area
                hull_box_area = Polygon(hull_box).area
                is_valid_hull_area = hull_area >= self.valid_area_threshold * hull_box_area

                # return score as average of inverse distance to closest match
                if is_valid_hull_area and is_valid_ratio:
                    pts[..., 0] += template_box[0]
                    pts[..., 1] += template_box[1]
                    anchor_center = np.mean(pts[:, 0, :], axis=0).tolist()
                    anchor_location = pts[:, 0, :]

                    dst[..., 0] = dst[..., 0] / scale + shift[0]
                    dst[..., 1] = dst[..., 1] / scale + shift[1]
                    found_center = np.mean(dst[:, 0, :], axis=0).tolist()
                    found_location = dst[:, 0, :]

                    return (
                        anchor_center, anchor_location,
                        found_center, found_location
                    )
        return None, None, None, None

    def _crop_roi_and_resize(self, query_image, template_image, box):
        r"""Crop ROI which possibly containing template anchor and resize it"""
        # get template anchor box coordinates relative to template image size
        x1, y1, x2, y2 = box
        x, y = x1 / template_image.shape[1], y1 / template_image.shape[0]
        w = (x2 - x1) / template_image.shape[1]
        h = (y2 - y1) / template_image.shape[0]

        # crop ROI
        pad_ratio = (self.roi_to_template_box_ratio - 1.0) / 2
        x1 = max(min(x - w * pad_ratio, 1.0), 0.0)
        y1 = max(min(y - h * pad_ratio, 1.0), 0.0)
        x2 = max(min(x + w * self.roi_to_template_box_ratio, 1.0), 0.0)
        y2 = max(min(y + h * self.roi_to_template_box_ratio, 1.0), 0.0)
        x1, y1 = int(x1 * query_image.shape[1]), int(y1 * query_image.shape[0])
        x2, y2 = int(x2 * query_image.shape[1]), int(y2 * query_image.shape[0])
        query_image = query_image[y1:y2, x1:x2]

        # resize ROI
        query_image_max_size = max(query_image.shape[:2])
        if self.image_max_size and query_image_max_size > self.image_max_size:
            ratio = self.image_max_size / query_image_max_size
            query_image = cv2.resize(query_image, (0, 0), fx=ratio, fy=ratio)
        else:
            ratio = 1.0

        return query_image, (x1, y1), ratio
