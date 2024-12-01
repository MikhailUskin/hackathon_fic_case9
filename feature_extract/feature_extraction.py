import cv2
import requests
import numpy as np
import onnxruntime as ort
import albumentations as A
from scipy.spatial import distance
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from pathlib import Path
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from segmentation_models_pytorch import utils
import torch


class FeatureExtraction:
    def __init__(
        self,
        onnx_path="./assets/osnet_x1_0.onnx",
        device="cuda",
    ):
        self.onnx_path = onnx_path
        self.device = device
        if self.device.lower() == "cpu":
            if ort.get_device() == "CUDA":
                print("CUDA available, if you want to switch your device into CUDA")
            self.ort_session = ort.InferenceSession(
                self.onnx_path, providers=["CPUExecutionProvider"]
            )
        elif self.device.lower() == "cuda":
            self.ort_session = ort.InferenceSession(
                self.onnx_path,
                providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"],
            )
        else:
            raise ValueError("Choose between CPU or CUDA!")
        self.model_height, self.model_width = self.ort_session.get_inputs()[0].shape[
            2:4
        ]
        self.image_augmentation = A.Compose(
            [A.Resize(self.model_height, self.model_width), A.Normalize(), ToTensorV2()]
        )

    def predict_img(self, img):
        image = self._preprocessing_img(img)
        input_onnx = self.ort_session.get_inputs()[0].name
        output_onnx = self.ort_session.run(None, {input_onnx: image})
        return output_onnx

    def _preprocessing_img(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = self.image_augmentation(image=np.array(image))["image"]
        image = np.expand_dims(image, axis=0)
        return image


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = "resnet32"

    # Load pretrained model and preprocessing function
    model = smp.from_pretrained(checkpoint).eval().to(device)
    preprocessing = A.Compose.from_pretrained(checkpoint)

    feature_extraction = FeatureExtraction(r"C:\Users\ArtyGod\PycharmProjects\yolop\mcmot-baseline-main\assets\osnet_x1_0.onnx")
    datasets_path = r"C:\Users\ArtyGod\PycharmProjects\yolop\mcmot-baseline-main\assets\datasets"
    cam0 = cv2.imread(datasets_path+r"\object_0\frame_00595_object_0.jpg")

    # Preprocess image
    image = np.array(cam0)
    normalized_image = preprocessing(image=image)["image"]
    input_tensor = torch.as_tensor(normalized_image)
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
    input_tensor = input_tensor.to(device)

    # Perform inference
    with torch.no_grad():
        output_mask = model(input_tensor)

    # Postprocess mask
    mask = torch.nn.functional.interpolate(
        output_mask, size=image.shape[:2], mode="bilinear", align_corners=False
    )
    mask = mask[0].argmax(0).cpu().numpy()
    # Plot results
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.axis("off")
    plt.imshow(image)
    plt.title("Input Image")

    plt.subplot(122)
    plt.axis("off")
    plt.imshow(mask)
    plt.title("Output Mask")

    plt.show()

    cam1 = cv2.imread(datasets_path+r"\object_1\frame_01131_object_1.jpg")
    photo_list = Path(datasets_path+r"\object_1").rglob('*.jpg')
    photo_list = sorted(photo_list)
    cam0 = cv2.resize(cam0, (128, 256))
    cv2.waitKey(0)
    output0 = feature_extraction.predict_img(cam0)[0][0]
    x=[]
    y=[]
    i=0
    for photo in photo_list:
        cam1 = cv2.imread(photo)
        cam1 = cv2.resize(cam1, (128, 256))
        out = model.forward(cam1)
        cv2.imshow("out", out)
        output1 = feature_extraction.predict_img(cam1)[0][0]
        y.append(distance.cosine(output0, output1))
        x.append(i)
        i=i+1
        print(f"Done: ", i/ len(photo_list)*100, " %")
    plt.plot(x, y)
    plt.show()
