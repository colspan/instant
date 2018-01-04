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
};

static instantModel* getModel(VALUE self) {
    instantModel* p;
    Data_Get_Struct(self, instantModel, p);
    return p;
}

static void wrap_model_free(instantModel* p) {
    delete p->model;
    ruby_xfree(p);
}

static VALUE wrap_model_alloc(VALUE klass) {
    void* p = ruby_xmalloc(sizeof(instantModel));
    return Data_Wrap_Struct(klass, NULL, wrap_model_free, p);
}

static VALUE wrap_model_init(VALUE self, VALUE vonnx, VALUE vbatch_size, VALUE vchannel_num,
                             VALUE vheight, VALUE vwidth) {

    constexpr auto batch_size = 1;
    constexpr auto channel_num = 3;
    constexpr auto height = 224;
    constexpr auto width = 224;
    // int batch_size = NUM2INT(vbatch_size);
    // int channel_num = NUM2INT(vchannel_num);
    // int height = NUM2INT(vheight);
    // int width = NUM2INT(vwidth);

    std::vector<int> input_dims{batch_size, channel_num, height, width};

    auto conv1_1_in_name = "140326425860192";
    auto fc6_out_name = "140326200777976";
    auto softmax_out_name = "140326200803680";

    getModel(self)->model = NULL;
    // TODO getModel(self)->modelにmake_modelの結果を代入する
    instant::make_model(
      *(getONNX(vonnx)->onnx),
      {std::make_tuple(conv1_1_in_name, instant::dtype_t::float_, input_dims,
                       mkldnn::memory::format::nchw)},
      {fc6_out_name, softmax_out_name});

    return Qnil;
}

static VALUE wrap_onnx_makeModel(VALUE self, VALUE vbatch_size,
                                 VALUE vchannel_num, VALUE vheight,
                                 VALUE vwidth) {

    VALUE klass = rb_const_get(rb_cObject, rb_intern("ONNXModel"));
    VALUE obj = rb_class_new_instance(0, NULL, klass);

    return obj;
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
                             RUBY_METHOD_FUNC(wrap_model_init), 1);
}
