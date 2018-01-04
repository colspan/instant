require './instant'
onnx_obj = ONNX.new("../data/VGG16.onnx")

batch_size = 1;
channel_num = 3;
height = 224;
width = 224;

onnx_obj.make_model(batch_size, channel_num, height, width)
