require 'rmagick'
include Magick

require './instant'

# ONNXファイルを読み込む
onnx_obj = ONNX.new("../data/VGG16.onnx")

# 推論の条件
batch_size = 1;
channel_num = 3;
height = 224;
width = 224;

# 推論に用いるネットワークを構築する
model = onnx_obj.make_model(batch_size, channel_num, height, width)

# 推論対象の画像を読み込む
image = Image.read("../data/Light_sussex_hen.jpg").first
image = image.resize_to_fill(width, height)

# p image.export_pixels.map { |pix| pix/257 }.each_slice(3).each_slice(image.columns).to_a
imageset = [image]
inference_result = model.inference(imageset)

# カテゴリ定義を読み込む
categories = File.read('../data/synset_words.txt').split("\n")

# スコアでソートする
sorted_result = inference_result.zip(categories).sort_by{|x| -x[0]}

# 推論結果を表示する
sorted_result[0,5].each do |result|
  puts "#{result[1]} : #{result[0]}"
end
