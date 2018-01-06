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
    std::vector<std::tuple<std::string, instant::dtype_t,
                           std::vector<int> const&, mkldnn::memory::format>>
      input_def;
    std::vector<std::string> output_def;
};

static instantModel* getModel(VALUE self) {
    instantModel* p;
    Data_Get_Struct(self, instantModel, p);
    return p;
}

static void wrap_model_free(instantModel* p) {
    // delete &(p->model); 不要?
    std::cout << "DEBUG ONNXModel free" << std::endl;
    ruby_xfree(p);
}

static VALUE wrap_model_alloc(VALUE klass) {
    void* p = ruby_xmalloc(sizeof(instantModel));
    return Data_Wrap_Struct(klass, NULL, wrap_model_free, p);
}

static VALUE wrap_model_init(VALUE self, VALUE vonnx, VALUE vbatch_size,
                             VALUE vchannel_num, VALUE vheight, VALUE vwidth) {

    int batch_size = NUM2INT(vbatch_size);
    int channel_num = NUM2INT(vchannel_num);
    int height = NUM2INT(vheight);
    int width = NUM2INT(vwidth);

    std::vector<int> input_dims{batch_size, channel_num, height, width};

    // TODO 外部から入力を受けるようにする
    VALUE vconv1_1_in_name = rb_str_new2("140326425860192");
    VALUE vfc6_out_name = rb_str_new2("140326200777976");
    VALUE vsoftmax_out_name = rb_str_new2("140326200803680");

    auto model = instant::make_model(
      *(getONNX(vonnx)->onnx),
      {std::make_tuple(StringValuePtr(vconv1_1_in_name),
                       instant::dtype_t::float_, input_dims,
                       mkldnn::memory::format::nchw)},
      {StringValuePtr(vfc6_out_name), StringValuePtr(vsoftmax_out_name)});
    getModel(self)->model = model;

    return Qnil;
}

static VALUE wrap_onnx_makeModel(VALUE self, VALUE vbatch_size,
                                 VALUE vchannel_num, VALUE vheight,
                                 VALUE vwidth) {

    VALUE args[] = {self, vbatch_size, vchannel_num, vheight, vwidth};
    VALUE klass = rb_const_get(rb_cObject, rb_intern("ONNXModel"));
    VALUE obj = rb_class_new_instance(5, args, klass);

    return obj;
}

static VALUE wrap_model_inference(VALUE self, VALUE image) {
    VALUE vconv1_1_in_name = rb_str_new2("140326425860192");

    // std::vector<float> image_data(mat.channels() * mat.rows * mat.cols);
    std::cout << "DEBUG test 1 " << std::endl;
    // Copy input image data to model's input array
    auto& input_array = getModel(self)->model->input("140326425860192");
    std::cout << "DEBUG test 2 " << std::endl;

    // std::copy(image_data.begin(), image_data.end(),
    // instant::fbegin(input_array));

    // Run inference
    auto const& output_table = getModel(self)->model->run();
    std::cout << "DEBUG test 3 " << std::endl;

    // TODO
    // Get output
    auto const& fc6_out_arr =
      instant::find_value(output_table, "140326200777976");
    std::cout << "fc6_out: ";
    for(int i = 0; i < 5; ++i) {
        std::cout << instant::fat(fc6_out_arr, i) << " ";
    }
    std::cout << "...\n";

    auto const& softmax_out_arr =
      instant::find_value(output_table, "140326200803680");

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
                     4);

    VALUE model = rb_define_class("ONNXModel", rb_cObject);

    rb_define_alloc_func(model, wrap_model_alloc);
    rb_define_private_method(model, "initialize",
                             RUBY_METHOD_FUNC(wrap_model_init), 5);

    rb_define_method(model, "inference", RUBY_METHOD_FUNC(wrap_model_inference),
                     1);
}
