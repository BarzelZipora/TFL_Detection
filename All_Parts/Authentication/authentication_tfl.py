try:
    import numpy as np

    from PIL import Image
except ImportError:
    print("Need to fix the installation")
    raise


class Authentication_TFL:
    def __init__(self, path, model):
        self.path = path
        self.model = model
        self.image = None


    def __cropping(self, coord):
        width, height = 2048, 1024
        left = max(0, coord[0] - 41)
        top = max(0, coord[1] - 41)
        right = min(width, coord[0] + 40)
        bottom = min(height, coord[1] + 40)
        image_from_array = Image.fromarray(self.image)
        cropped_image = image_from_array.crop((left, top, right, bottom))
        if np.array(cropped_image).shape != (81, 81, 3):
            return None
        return cropped_image

    def __is_tfl(self,image):
        crop_shape = (81, 81)
        test_image =np.array(image).reshape([-1] + list(crop_shape) + [3])
        predictions = self.model.predict(test_image)
        predicted_label = 1 if predictions > 0.5 else 0
        return predicted_label

    # np.mean(predicted_label == val['labels']

    def __get_tfls(self, candidates):
        candidates_tfl = []
        for index, pixel in enumerate(candidates):
            crop_image = self.__cropping(candidates[index])
            if crop_image and self.__is_tfl(crop_image):
                candidates_tfl.append(pixel)
        return candidates_tfl

    def run(self, red_candidates, green_candidates):
        self.image = np.array(Image.open(self.path))
        red_tfls = self.__get_tfls(red_candidates)
        green_tfls = self.__get_tfls(green_candidates)
        return red_tfls, green_tfls
