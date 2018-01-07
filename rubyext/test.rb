require 'rmagick'
include Magick

require './instant'

# load ONNX file
onnx_obj = ONNX.new("../data/VGG16.onnx")

conv1_1_in_name = "140326425860192"
fc6_out_name = "140326200777976"
softmax_out_name = "140326200803680"

# load dataset
imagelist = [
  "../data/Light_sussex_hen.jpg", #"../data/Light_sussex_hen.jpg"
]

# conditions for inference
condition = {
  :batch_size => imagelist.length,
  :channel_num => 3,
  :height => 224,
  :width => 224,
  :input_layer => conv1_1_in_name,
  :output_layers => [fc6_out_name, softmax_out_name]
}

# prepare dataset
imageset = []
imagelist.each do |image_filepath|
  image = Image.read(image_filepath).first
  imageset << image.resize_to_fill(condition[:width], condition[:height])
end

# make model for inference under 'condition'
model = onnx_obj.make_model(condition)

# execute inference
inference_results = model.inference(imageset)

# load category definition
categories = File.read('../data/synset_words.txt').split("\n")

inference_results.each_with_index do |inference_result, i|
    puts "=== Result for #{imagelist[i]} ==="

    # sort by score
    sorted_result = inference_result[softmax_out_name].zip(categories).sort_by{|x| -x[0]}

    # display result
    sorted_result[0,5].each do |result|
    puts "#{result[1]} : #{result[0]}"
    end
end

