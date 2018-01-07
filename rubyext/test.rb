require 'rmagick'
include Magick

require './instant'

# load ONNX file
onnx_obj = ONNX.new("../data/VGG16.onnx")

conv1_1_in_name = "140326425860192"
fc6_out_name = "140326200777976"
softmax_out_name = "140326200803680"

# conditions for inference
condition = {
  :batch_size => 1,
  :channel_num => 3,
  :height => 224,
  :width => 224,
  :input_layer => conv1_1_in_name,
  :output_layers => [fc6_out_name, softmax_out_name]
}

# make model for inference under 'condition'
model = onnx_obj.make_model(condition)

# load dataset
image = Image.read("../data/Light_sussex_hen.jpg").first
image = image.resize_to_fill(condition[:width], condition[:height])
imageset = [image]

# execute inference
inference_result = model.inference(imageset)

# load category definition
categories = File.read('../data/synset_words.txt').split("\n")

# sort by score
sorted_result = inference_result[softmax_out_name].zip(categories).sort_by{|x| -x[0]}

# display result
sorted_result[0,5].each do |result|
  puts "#{result[1]} : #{result[0]}"
end
