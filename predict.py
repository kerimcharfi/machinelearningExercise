import json

import cv2
import numpy as np
from dacite import from_dict
from tensorflow import keras
from detection.paletten_check.crop_slots import crop_slot_imgs
from detection.paletten_check.palette_data_classes import PaletteData
from server.DTO import IntParam, PaletteInspectionDTO, PaletteSlotDTO

with open("detection/paletten_check/slots.json") as f:
    palette_data = from_dict(PaletteData, json.loads(f.read()))

params = {
    # "pallete_reference_circles_area": 9300,
    "palette_reference_circles_area": IntParam(1950),
    "palette_reference_circles_circularity_score": IntParam(17)
}

model = keras.models.load_model('detection/experimental_approaches/keras/keras_30.h5')

IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
CLASS_NAMES = ['issue', 'ok']


def inspect_palette(img):
    slot_imgs = crop_slot_imgs(img, palette_data, params)
    inspection_result = PaletteInspectionDTO(palette_detected=False, palette_ok=False, slots=[])
    if slot_imgs is None:
        return inspection_result
    inspection_result.palette_detected = True

    predictions = model.predict(np.array(slot_imgs))
    pred_labels = np.argmax(predictions, axis=1)
    for prediction_label_index, slot in zip(pred_labels, palette_data.slots):
        if prediction_label_index == 0:
            inspection_result.palette_ok = False
        slot_result = PaletteSlotDTO(id=slot.id, x=slot.x, y=slot.y, state=CLASS_NAMES[prediction_label_index])
        inspection_result.slots.append(slot_result)
    return inspection_result


if __name__ == '__main__':


    img = cv2.imread("imgs/pallet/raw/issue@09.08.2022 17;46;17.jpg", cv2.IMREAD_COLOR)

    result = inspect_palette(img)
    print(result)
