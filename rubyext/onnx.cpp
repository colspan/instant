#include <iostream>
#include <new>
#include <ruby.h>

#include "../instant/instant.hpp"

struct instantONNX {
    onnx::ModelProto* onnx;
};

static instantONNX* getONNX(VALUE self) {
    instantONNX* p;
    Data_Get_Struct(self, instantONNX, p);
    return p;
}

static void wrap_onnx_free(instantONNX* p) {
    std::cout << "DEBUG ONNX free" << std::endl;
    p->onnx->~ModelProto();
    ruby_xfree(p);
}

static VALUE wrap_onnx_alloc(VALUE klass) {
    void* p = ruby_xmalloc(sizeof(instantONNX));
    return Data_Wrap_Struct(klass, NULL, wrap_onnx_free, p);
}

static VALUE wrap_onnx_init(VALUE self, VALUE vfilename) {
    char* filename = StringValuePtr(vfilename);
    // Load ONNX model
    getONNX(self)->onnx = new onnx::ModelProto(instant::load_onnx(filename));

    return Qnil;
}

struct instantModel {
    instant::model* model;
    int batch_size;
    int channel_num;
    int width;
    int height;
    char* input_layer;
    std::string output_layers;
};

static instantModel* getModel(VALUE self) {
    instantModel* p;
    Data_Get_Struct(self, instantModel, p);
    return p;
}

static void wrap_model_free(instantModel* p) {
    // delete &(p->model); 不要?
    delete p->input_layer;
    std::cout << "DEBUG ONNXModel free" << std::endl;
    ruby_xfree(p);
}

static VALUE wrap_model_alloc(VALUE klass) {
    void* p = ruby_xmalloc(sizeof(instantModel));
    return Data_Wrap_Struct(klass, NULL, wrap_model_free, p);
}

static VALUE wrap_model_init(VALUE self, VALUE vonnx, VALUE condition) {

    // TODO 型チェックを行う

    int batch_size = getModel(self)->batch_size =
      NUM2INT(rb_hash_aref(condition, rb_to_symbol(rb_str_new2("batch_size"))));
    int channel_num = getModel(self)->channel_num = NUM2INT(
      rb_hash_aref(condition, rb_to_symbol(rb_str_new2("channel_num"))));
    int height = getModel(self)->height =
      NUM2INT(rb_hash_aref(condition, rb_to_symbol(rb_str_new2("height"))));
    int width = getModel(self)->width =
      NUM2INT(rb_hash_aref(condition, rb_to_symbol(rb_str_new2("width"))));
    VALUE vinput_layer =
      rb_hash_aref(condition, rb_to_symbol(rb_str_new2("input_layer")));

    getModel(self)->input_layer =
      new char[strlen(StringValuePtr(vinput_layer))];
    strncpy(getModel(self)->input_layer, StringValuePtr(vinput_layer),
            strlen(StringValuePtr(vinput_layer)));

    std::vector<int> input_dims{batch_size, channel_num, height, width};

    // TODO 外部から入力を受けるようにする
    VALUE vfc6_out_name = rb_str_new2("140326200777976");
    VALUE vsoftmax_out_name = rb_str_new2("140326200803680");

    auto model = instant::make_model(
      *(getONNX(vonnx)->onnx),
      {std::make_tuple(getModel(self)->input_layer, instant::dtype_t::float_,
                       input_dims, mkldnn::memory::format::nchw)},
      {StringValuePtr(vfc6_out_name), StringValuePtr(vsoftmax_out_name)});
    getModel(self)->model = model;

    return Qnil;
}

static VALUE wrap_onnx_makeModel(VALUE self, VALUE condition) {

    VALUE args[] = {self, condition};
    VALUE klass = rb_const_get(rb_cObject, rb_intern("ONNXModel"));
    VALUE obj = rb_class_new_instance(2, args, klass);

    return obj;
}

static VALUE wrap_model_inference(VALUE self, VALUE images) {

    int image_num = NUM2INT(rb_funcall(images, rb_intern("length"), 0, NULL));

    std::vector<float> image_data(getModel(self)->channel_num *
                                  getModel(self)->width *
                                  getModel(self)->height);
    // RMagick の形式を変換する
    for(int i; i < image_num; i++) {
        VALUE image = rb_ary_entry(images, 0);
        VALUE raw_values =
          rb_funcall(image, rb_intern("export_pixels"), 0, NULL);
        auto value_num =
          NUM2INT(rb_funcall(raw_values, rb_intern("length"), 0, NULL));
        for(int y = 0; y < getModel(self)->height; ++y) {
            for(int x = 0; x < getModel(self)->width; ++x) {
                for(int c = 0; c < getModel(self)->channel_num; c++) {
                    image_data[c * (getModel(self)->width *
                                    getModel(self)->height) +
                               y * getModel(self)->width + x] =
                      static_cast<float>(
                        NUM2INT(rb_ary_entry(
                          raw_values, getModel(self)->channel_num *
                                          (x + y * getModel(self)->height) +
                                        c)) /
                        257);
                }
            }
        }
    }
    // Copy input image data to model's input array
    auto& input_array =
      getModel(self)->model->input(getModel(self)->input_layer);

    std::copy(image_data.begin(), image_data.end(),
              instant::fbegin(input_array));

    // Run inference
    auto const& output_table = getModel(self)->model->run();

    // Get output
    auto const& fc6_out_arr =
      instant::find_value(output_table, "140326200777976");

    auto const& softmax_out_arr =
      instant::find_value(output_table, "140326200803680");
    // 配列に積んで返す
    VALUE result_array = rb_ary_new();
    for(int i = 0; i < instant::total_size(softmax_out_arr); ++i) {
        rb_ary_push(result_array, DBL2NUM(instant::fat(softmax_out_arr, i)));
    }
    return result_array;
}

/**
 * require時に呼び出し
 */
extern "C" void Init_instant() {
    VALUE onnx = rb_define_class("ONNX", rb_cObject);

    rb_define_alloc_func(onnx, wrap_onnx_alloc);
    rb_define_private_method(onnx, "initialize",
                             RUBY_METHOD_FUNC(wrap_onnx_init), 1);
    rb_define_method(onnx, "make_model", RUBY_METHOD_FUNC(wrap_onnx_makeModel),
                     1);

    VALUE model = rb_define_class("ONNXModel", rb_cObject);

    rb_define_alloc_func(model, wrap_model_alloc);
    rb_define_private_method(model, "initialize",
                             RUBY_METHOD_FUNC(wrap_model_init), 2);

    rb_define_method(model, "inference", RUBY_METHOD_FUNC(wrap_model_inference),
                     1);
}
