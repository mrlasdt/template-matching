
# @brief  Line parser based on template boxes, i.e. simply crop text boxes given the original image and text boxes' coordinates
class TemplateBoxParser():
    def __init__(self):
        pass

    ##
    # @brief  Run line parser
    #
    # @param images: Refer to interface
    # @param metadata: Refer to interface. Each metadata dict, i.e. correspond to an image, has format
    #   
    #   {
    #       "boxes": [
    #           (top, left, w, h),
    #           (top, left, w, h),
    #           ...
    #       ]
    #   }
    #   
    #   where coordinates are absolute coordinates
    #
    # @return cropped_images: Refer to interface
    def run(self, images, metadata):
        cropped_images = []
        for image, _metadata in zip(images, metadata):
            _cropped_images = []
            for box in _metadata["boxes"]:
                # y, x, w, h = box
                # x1, y1, x2, y2 = x, y, x + w, y + h
                x1, y1, x2, y2 = box 
                # print(bo)
                _cropped_images.append(image[y1:y2, x1:x2].copy())
            cropped_images.append(_cropped_images)
        return cropped_images


