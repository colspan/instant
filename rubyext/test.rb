require 'rmagick'
include Magick

require './instant'

# ONNXファイルを読み込む
onnx_obj = ONNX.new("../data/VGG16.onnx")

conv1_1_in_name = "140326425860192"
fc6_out_name = "140326200777976"
softmax_out_name = "140326200803680"

# 推論の条件
condition = {
  :batch_size => 1,
  :channel_num => 3,
  :height => 224,
  :width => 224,
  :input_layer => conv1_1_in_name,
  :output_layers => [fc6_out_name, softmax_out_name]
}

# 推論に用いるネットワークを構築する
model = onnx_obj.make_model(condition)

# 推論対象の画像を読み込む
image = Image.read("../data/Light_sussex_hen.jpg").first
image = image.resize_to_fill(condition[:width], condition[:height])
imageset = [image]

# 推論を実行
inference_result = model.inference(imageset)

# カテゴリ定義を読み込む
categories = File.read('../data/synset_words.txt').split("\n")

# スコアでソートする
sorted_result = inference_result[softmax_out_name].zip(categories).sort_by{|x| -x[0]}
# [softmax_out_name]

# 推論結果を表示する
sorted_result[0,5].each do |result|
  puts "#{result[1]} : #{result[0]}"
end
