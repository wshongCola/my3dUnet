from lib.model.network import UNet3D, UNet3D_sigmoid, UNet3D_tanh

model_factory = dict()
model_factory['UNet3D'] = UNet3D
model_factory['UNet3D_sigmoid'] = UNet3D_sigmoid
model_factory['UNet3D_tanh'] = UNet3D_tanh
