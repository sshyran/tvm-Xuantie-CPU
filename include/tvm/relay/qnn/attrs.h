/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/relay/qnn/attrs.h
 * \brief Auxiliary attributes for qnn operators.
 */
#ifndef TVM_RELAY_QNN_ATTRS_H_
#define TVM_RELAY_QNN_ATTRS_H_

#include <tvm/ir/attrs.h>

#include <string>

namespace tvm {
namespace relay {
namespace qnn {

/*! \brief Attribute for requantize operator */
struct RequantizeAttrs : public tvm::AttrsNode<RequantizeAttrs> {
  int axis;
  std::string rounding;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(RequantizeAttrs, "relay.attrs.RequantizeAttrs") {
    TVM_ATTR_FIELD(axis)
        .describe(
            "The output channel axis for channel wise quantization. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
    TVM_ATTR_FIELD(rounding).set_default("UPWARD").describe(
        "Defines the rounding direction when the value is midway between"
        "two representable values. There are two supported modes - UPWARD"
        "or TONEAREST. Both modes behave exactly same except at the"
        "midpoints between the two representable values. At the midpoint,"
        "UPWARD rounds towards positive infinity (for example -1.5 will be"
        "rounded to -1). TONEAREST is the standard rounding where the"
        "value is rounded away from zero at midpoints (for example, -1.5"
        "rounds to -2). More context can be found at following gblic manual"
        "https://www.gnu.org/software/libc/manual/html_node/Rounding.html.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

/*! \brief Attribute for quantize operator */
struct QuantizeAttrs : public tvm::AttrsNode<QuantizeAttrs> {
  int32_t output_zero_point;
  double output_scale;
  DataType out_dtype;
  int axis;

  TVM_DECLARE_ATTRS(QuantizeAttrs, "relay.attrs.QuantizeAttrs") {
    TVM_ATTR_FIELD(out_dtype).describe("Output data type, can be one of [int8 or uint8].");
    TVM_ATTR_FIELD(axis)
        .describe(
            "The output channel axis for channel wise quantization. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
  }
};

/*! \brief Attribute for dequantize operator */
struct DequantizeAttrs : public tvm::AttrsNode<DequantizeAttrs> {
  int axis;

  TVM_DECLARE_ATTRS(DequantizeAttrs, "relay.attrs.DequantizeAttrs") {
    TVM_ATTR_FIELD(axis)
        .describe(
            "The channel axis for channel wise dequantization. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
  }
};

/*! \brief Attribute for nn_init operator */
struct NNInitAttrs : public tvm::AttrsNode<NNInitAttrs> {
  int32_t output_zero_point;
  double output_scale;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(NNInitAttrs, "relay.attrs.NNInitAttrs") {
    TVM_ATTR_FIELD(out_dtype).describe("Output data type, can be one of [int8 or uint8].");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero_point for the activation of this op.");
    TVM_ATTR_FIELD(output_scale).describe("The scale for the activation of this op.");
  }
};

/*! \brief Attribute for nn_deinit operator */
struct NNDeinitAttrs : public tvm::AttrsNode<NNDeinitAttrs> {
  int32_t input_zero_point;
  double input_scale;

  TVM_DECLARE_ATTRS(NNDeinitAttrs, "relay.attrs.NNDeinitAttrs") {
    TVM_ATTR_FIELD(input_zero_point).describe("The zero_point for the input tensor of this op.");
    TVM_ATTR_FIELD(input_scale).describe("The scale for the input tensor of this op.");
  }
};

/*! \brief Attributes used in QNN concatenate operator */
struct QnnConcatenateAttrs : public tvm::AttrsNode<QnnConcatenateAttrs> {
  Array<tvm::PrimExpr> input_scales;
  Array<tvm::PrimExpr> input_zero_points;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  int32_t output_zero_point;
  int axis;
  String layer_name;
  TVM_DECLARE_ATTRS(QnnConcatenateAttrs, "relay.attrs.QnnConcatenateAttrs") {
    TVM_ATTR_FIELD(input_scales).describe("The list of scales of input quantized tensors.");
    TVM_ATTR_FIELD(input_zero_points)
        .describe("The list of zero points of input quantized tensors.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero_point for the output tensor.");
    TVM_ATTR_FIELD(output_scale).describe("The scale for the output tensor.");
    TVM_ATTR_FIELD(axis)
        .describe(
            "The axis at which the input arrays are concatenated."
            "Should lie in range `[-ndim, ndim)`.")
        .set_default(0);
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};  // struct QnnConcatenateAttrs

/*! \brief Attribute for QNN Conv2d operator */
struct QnnConv2DAttrs : public tvm::AttrsNode<QnnConv2DAttrs> {
  // Traditional conv2d attributes.
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilation;
  int groups;
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  std::string data_layout;
  std::string kernel_layout;
  std::string out_layout;
  DataType out_dtype;

  // Quantization related attributes.
  int32_t input_zero_point;
  int32_t kernel_zero_point;
  // The input tensor scale and kernel tensor scales are stored
  // for easy access to this information.
  double input_scale;
  double kernel_scale;

  TVM_DECLARE_ATTRS(QnnConv2DAttrs, "relay.attrs.QnnConv2DAttrs") {
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "on both sides for padding number of points");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(groups).set_default(1).describe(
        "Controls the connections between inputs and outputs."
        "At groups=1, all inputs are convolved to all outputs."
        "At groups=2, the operation becomes equivalent to having two convolution"
        "layers side by side, each seeing half the input channels, and producing"
        "half the output channels, and both subsequently concatenated.");
    TVM_ATTR_FIELD(channels)
        .describe(
            "The number of output channels in the convolution."
            " If it is not set, inferred by shape of the weight.")
        .set_default(NullValue<IndexExpr>());
    TVM_ATTR_FIELD(kernel_size)
        .describe("Specifies the dimensions of the convolution window.")
        .set_default(NullValue<Array<IndexExpr>>());
    TVM_ATTR_FIELD(data_layout)
        .set_default("NCHW")
        .describe(
            "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Convolution is applied on the 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("OIHW")
        .describe(
            "Dimension ordering of weight. Can be 'OIHW', 'OIHW16o16i', etc."
            "'O', 'I', 'H', 'W' stands for num_filter, input_channel, height, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Default to be same as input layout.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(kernel_zero_point).describe("The zero point of the kernel tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The quantization scale for the input tensor.");
    TVM_ATTR_FIELD(kernel_scale).describe("The quantization scale for the weight tensor.");
  }
};

/*! \brief Attribute for QNN DeConv2d operator */
struct QnnConv2DTransposeAttrs : public tvm::AttrsNode<QnnConv2DTransposeAttrs> {
  // Traditional conv2d attributes.
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> output_padding;
  Array<IndexExpr> dilation;
  int groups;
  std::string data_layout;
  std::string kernel_layout;
  std::string out_layout;
  DataType out_dtype;

  // Quantization related attributes.
  int32_t input_zero_point;
  int32_t kernel_zero_point;
  // The input tensor scale and kernel tensor scales are stored
  // for easy access to this information.
  double input_scale;
  double kernel_scale;

  TVM_DECLARE_ATTRS(QnnConv2DTransposeAttrs, "relay.attrs.QnnConv2DTransposeAttrs") {
    TVM_ATTR_FIELD(channels)
        .set_default(NullValue<IndexExpr>())
        .describe(
            "The dimensionality of the output space"
            "i.e. the number of output channels in the convolution.");
    TVM_ATTR_FIELD(kernel_size)
        .describe("The dimensions of the convolution window.")
        .set_default(NullValue<Array<IndexExpr>>());
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("The strides of the convolution.");
    TVM_ATTR_FIELD(output_padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe("Zero-padding added to one side of the output.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "on both sides for padding number of points");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(groups).set_default(1).describe(
        "Controls the connections between inputs and outputs."
        "At groups=1, all inputs are convolved to all outputs."
        "At groups=2, the operation becomes equivalent to having two convolution"
        "layers side by side, each seeing half the input channels, and producing"
        "half the output channels, and both subsequently concatenated.");
    TVM_ATTR_FIELD(data_layout)
        .set_default("NCHW")
        .describe(
            "Dimension ordering of data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Convolution is applied on the 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("OIHW")
        .describe(
            "Dimension ordering of data and weight. Can be 'OIHW', 'OIHW16o16i', etc."
            "'O', 'I', 'H', 'W' stands for num_filter, input_channel, height, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Default to be same as input layout.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(kernel_zero_point).describe("The zero point of the kernel tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The quantization scale for the input tensor.");
    TVM_ATTR_FIELD(kernel_scale).describe("The quantization scale for the weight tensor.");
  }
};

/*! \brief Attribute for QNN PRelu operator */
struct QnnPReluAttrs : public tvm::AttrsNode<QnnPReluAttrs> {
  int axis;
  // Quantization related attributes.
  double input_scale;
  double alpha_scale;
  int32_t input_zero_point;
  int32_t alpha_zero_point;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnPReluAttrs, "relay.attrs.QnnPReluAttrs") {
    TVM_ATTR_FIELD(axis).describe("The axis at which the input arrays are computed.");
    TVM_ATTR_FIELD(input_scale).describe("The quantization scale for the input tensor.");
    TVM_ATTR_FIELD(alpha_scale).describe("The quantization scale for the weight tensor.");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(alpha_zero_point).describe("The zero point of the kernel tensor.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIDeConv2DAttrs : public tvm::AttrsNode<QnnCSIDeConv2DAttrs> {
  // Traditional conv2d attributes.
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> output_padding;
  Array<IndexExpr> dilation;
  int groups;
  std::string data_layout;
  std::string kernel_layout;
  std::string out_layout;
  DataType out_dtype;

  // Quantization related attributes.
  int32_t input_zero_point;
  int32_t kernel_zero_point;
  int32_t output_zero_point;
  // The input tensor scale and kernel tensor scales are stored
  // for easy access to this information.
  double input_scale;
  double kernel_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;

  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIDeConv2DAttrs, "relay.attrs.QnnCSIDeConv2DAttrs") {
    TVM_ATTR_FIELD(channels)
        .set_default(NullValue<IndexExpr>())
        .describe(
            "The dimensionality of the output space"
            "i.e. the number of output channels in the convolution.");
    TVM_ATTR_FIELD(kernel_size)
        .describe("The dimensions of the convolution window.")
        .set_default(NullValue<Array<IndexExpr>>());
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("The strides of the convolution.");
    TVM_ATTR_FIELD(output_padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe("Zero-padding added to one side of the output.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "on both sides for padding number of points");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(groups).set_default(1).describe(
        "Controls the connections between inputs and outputs."
        "At groups=1, all inputs are convolved to all outputs."
        "At groups=2, the operation becomes equivalent to having two convolution"
        "layers side by side, each seeing half the input channels, and producing"
        "half the output channels, and both subsequently concatenated.");
    TVM_ATTR_FIELD(data_layout)
        .set_default("NCHW")
        .describe(
            "Dimension ordering of data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Convolution is applied on the 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("OIHW")
        .describe(
            "Dimension ordering of data and weight. Can be 'OIHW', 'OIHW16o16i', etc."
            "'O', 'I', 'H', 'W' stands for num_filter, input_channel, height, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Default to be same as input layout.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(kernel_zero_point).describe("The zero point of the kernel tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The quantization scale for the input tensor.");
    TVM_ATTR_FIELD(kernel_scale).describe("The quantization scale for the weight tensor.");
    TVM_ATTR_FIELD(output_scale).describe("The quantization scale for the output tensor.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

/*! \brief Attribute for QNN Conv2d operator */
struct QnnCSIConv2DAttrs : public tvm::AttrsNode<QnnCSIConv2DAttrs> {
  // Traditional conv2d attributes.
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilation;
  int groups;
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  std::string data_layout;
  std::string kernel_layout;
  std::string out_layout;
  DataType out_dtype;

  // Quantization related attributes.
  int32_t input_zero_point;
  int32_t kernel_zero_point;
  int32_t output_zero_point;
  // The input tensor scale and kernel tensor scales are stored
  // for easy access to this information.
  double input_scale;
  double kernel_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIConv2DAttrs, "relay.attrs.QnnCSIConv2DAttrs") {
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "on both sides for padding number of points");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(groups).set_default(1).describe(
        "Controls the connections between inputs and outputs."
        "At groups=1, all inputs are convolved to all outputs."
        "At groups=2, the operation becomes equivalent to having two convolution"
        "layers side by side, each seeing half the input channels, and producing"
        "half the output channels, and both subsequently concatenated.");
    TVM_ATTR_FIELD(channels)
        .describe(
            "The number of output channels in the convolution."
            " If it is not set, inferred by shape of the weight.")
        .set_default(NullValue<IndexExpr>());
    TVM_ATTR_FIELD(kernel_size)
        .describe("Specifies the dimensions of the convolution window.")
        .set_default(NullValue<Array<IndexExpr>>());
    TVM_ATTR_FIELD(data_layout)
        .set_default("NCHW")
        .describe(
            "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Convolution is applied on the 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("OIHW")
        .describe(
            "Dimension ordering of weight. Can be 'OIHW', 'OIHW16o16i', etc."
            "'O', 'I', 'H', 'W' stands for num_filter, input_channel, height, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Default to be same as input layout.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_scale).describe("The quantization scale for the input tensor.");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(kernel_scale).describe("The quantization scale for the weight tensor.");
    TVM_ATTR_FIELD(kernel_zero_point).describe("The zero point of the kernel tensor.");
    TVM_ATTR_FIELD(output_scale).describe("The quantization scale for the output tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIConv2DChannelAttrs : public tvm::AttrsNode<QnnCSIConv2DChannelAttrs> {
  // Traditional conv2d attributes.
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilation;
  int groups;
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  std::string data_layout;
  std::string kernel_layout;
  std::string out_layout;
  DataType out_dtype;

  // Quantization related attributes.
  int32_t input_zero_point;
  int32_t output_zero_point;
  // The input tensor scale and kernel tensor scales are stored
  // for easy access to this information.
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIConv2DChannelAttrs, "relay.attrs.QnnCSIConv2DChannelAttrs") {
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "on both sides for padding number of points");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(groups).set_default(1).describe(
        "Controls the connections between inputs and outputs."
        "At groups=1, all inputs are convolved to all outputs."
        "At groups=2, the operation becomes equivalent to having two convolution"
        "layers side by side, each seeing half the input channels, and producing"
        "half the output channels, and both subsequently concatenated.");
    TVM_ATTR_FIELD(channels)
        .describe(
            "The number of output channels in the convolution."
            " If it is not set, inferred by shape of the weight.")
        .set_default(NullValue<IndexExpr>());
    TVM_ATTR_FIELD(kernel_size)
        .describe("Specifies the dimensions of the convolution window.")
        .set_default(NullValue<Array<IndexExpr>>());
    TVM_ATTR_FIELD(data_layout)
        .set_default("NCHW")
        .describe(
            "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Convolution is applied on the 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("OIHW")
        .describe(
            "Dimension ordering of weight. Can be 'OIHW', 'OIHW16o16i', etc."
            "'O', 'I', 'H', 'W' stands for num_filter, input_channel, height, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Default to be same as input layout.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_scale).describe("The quantization scale for the input tensor.");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_scale).describe("The quantization scale for the output tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

/*! \brief Attributes used in convolution3D operators */
struct QnnCSIConv3DAttrs : public tvm::AttrsNode<QnnCSIConv3DAttrs> {
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilation;
  int groups;
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  std::string data_layout;
  std::string kernel_layout;
  std::string out_layout;
  DataType out_dtype;

  // Quantization related attributes.
  int32_t input_zero_point;
  int32_t kernel_zero_point;
  int32_t output_zero_point;
  // The input tensor scale and kernel tensor scales are stored
  // for easy access to this information.
  double input_scale;
  double kernel_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIConv3DAttrs, "relay.attrs.QnnCSIConv3DAttrs") {
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "three int : back, bottom, right will use same padding as front, top, left"
            "six int : padding width in the order of (front, top, left, back, bottom,"
            "right)");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1, 1, 1}))
        .describe("Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(groups).set_default(1).describe(
        "Controls the connections between inputs and outputs."
        "At groups=1, all inputs are convolved to all outputs."
        "At groups=2, the operation becomes equivalent to having two convolution"
        "layers side by side, each seeing half the input channels, and producing"
        "half the output channels, and both subsequently concatenated.");
    TVM_ATTR_FIELD(channels)
        .describe(
            "The number of output channels in the convolution."
            " If it is not set, inferred by shape of the weight.")
        .set_default(NullValue<IndexExpr>());
    TVM_ATTR_FIELD(kernel_size)
        .describe("Specifies the dimensions of the convolution window.")
        .set_default(NullValue<Array<IndexExpr>>());
    TVM_ATTR_FIELD(data_layout)
        .set_default("NCDHW")
        .describe(
            "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
            "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
            "dimensions respectively. Convolution is applied on the 'D', 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("OIDHW")
        .describe(
            "Dimension ordering of weight. Can be 'OIDHW', 'OIDHW16o16i', etc."
            "'O', 'I', 'D', 'H', 'W' stands for num_filter, input_channel, depth, height,"
            "and width dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output. Can be 'NCDHW', 'NDHWC', etc."
            "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
            "dimensions respectively. Default to be same as input layout.");

    // use 0 bits to indicate none.
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_scale).describe("The quantization scale for the input tensor.");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(kernel_scale).describe("The quantization scale for the weight tensor.");
    TVM_ATTR_FIELD(kernel_zero_point).describe("The zero point of the kernel tensor.");
    TVM_ATTR_FIELD(output_scale).describe("The quantization scale for the output tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

/*! \brief Attributes used in transposed convolution operator */
struct QnnCSIDeConv3DAttrs : public tvm::AttrsNode<QnnCSIDeConv3DAttrs> {
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> output_padding;
  Array<IndexExpr> dilation;
  int groups;
  std::string data_layout;
  std::string kernel_layout;
  std::string out_layout;
  DataType out_dtype;

  // Quantization related attributes.
  int32_t input_zero_point;
  int32_t kernel_zero_point;
  int32_t output_zero_point;
  // The input tensor scale and kernel tensor scales are stored
  // for easy access to this information.
  double input_scale;
  double kernel_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIDeConv3DAttrs, "relay.attrs.QnnCSIDeConv3DAttrs") {
    TVM_ATTR_FIELD(channels)
        .set_default(NullValue<IndexExpr>())
        .describe(
            "The dimensionality of the output space"
            "i.e. the number of output channels in the convolution.");
    TVM_ATTR_FIELD(kernel_size)
        .describe("The dimensions of the convolution window.")
        .set_default(NullValue<Array<IndexExpr>>());
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1, 1}))
        .describe("The strides of the convolution.");
    TVM_ATTR_FIELD(output_padding)
        .set_default(Array<IndexExpr>({0, 0, 0}))
        .describe(
            "Zero-padding added to one side of the output."
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "three int : front, bottom, right will use same padding as back, top, left"
            "six int : padding width in the order of (front, top, left, back, bottom, right)");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "three int : front, bottom, right will use same padding as back, top, left"
            "six int : padding width in the order of (front, top, left, back, bottom, right)");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1, 1, 1}))
        .describe("Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(groups).set_default(1).describe(
        "Controls the connections between inputs and outputs."
        "At groups=1, all inputs are convolved to all outputs."
        "At groups=2, the operation becomes equivalent to having two convolution"
        "layers side by side, each seeing half the input channels, and producing"
        "half the output channels, and both subsequently concatenated.");
    TVM_ATTR_FIELD(data_layout)
        .set_default("NCDHW")
        .describe(
            "Dimension ordering of data. Can be 'NCDHW', 'NDHWC', etc."
            "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
            "dimensions respectively. Convolution is applied on the 'D', 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("OIDHW")
        .describe(
            "Dimension ordering of data and weight. Can be 'OIDHW', 'OIDHW16o16i', etc."
            "'O', 'I', 'D', 'H', 'W' stands for num_filter, input_channel, depth, height, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output. Can be 'NCDHW', 'NDHWC', etc."
            "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
            "dimensions respectively. Default to be same as input layout.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_scale).describe("The quantization scale for the input tensor.");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(kernel_scale).describe("The quantization scale for the weight tensor.");
    TVM_ATTR_FIELD(kernel_zero_point).describe("The zero point of the kernel tensor.");
    TVM_ATTR_FIELD(output_scale).describe("The quantization scale for the output tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

/*! \brief Attribute for QNN binary operator */
struct QnnBinaryOpAttrs : public tvm::AttrsNode<QnnBinaryOpAttrs> {
  int32_t lhs_zero_point;
  double lhs_scale;
  int32_t rhs_zero_point;
  double rhs_scale;
  int32_t output_zero_point;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnBinaryOpAttrs, "relay.attrs.QnnBinaryOpAttrs") {
    TVM_ATTR_FIELD(lhs_zero_point).describe("The zero_point for the lhs input tensor of this op.");
    TVM_ATTR_FIELD(lhs_scale).describe("The scale for the lhs input tensor of this op.");
    TVM_ATTR_FIELD(rhs_zero_point).describe("The zero_point for the rhs input tensor of this op.");
    TVM_ATTR_FIELD(rhs_scale).describe("The scale for the rhs input tensor of this op.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero_point for the activation of this op.");
    TVM_ATTR_FIELD(output_scale).describe("The scale for the activation of this op.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

/*! \brief Attributes for qnn dense operator */
struct QnnDenseAttrs : public tvm::AttrsNode<QnnDenseAttrs> {
  int axis;
  IndexExpr units;
  DataType out_dtype;
  // Quantization related attributes.
  int32_t input_zero_point;
  int32_t kernel_zero_point;
  double input_scale;
  double kernel_scale;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnDenseAttrs, "relay.attrs.QnnDenseAttrs") {
    TVM_ATTR_FIELD(units).describe("Number of hidden units of the dense transformation.");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(kernel_zero_point).describe("The zero point of the kernel tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(kernel_scale).describe("The kernel tensor scale.");
    TVM_ATTR_FIELD(axis)
        .describe(
            "The channel axis for channel wise dequantization. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

/*! \brief Attributes for qnn dense operator */
struct QnnCSIDenseAttrs : public tvm::AttrsNode<QnnCSIDenseAttrs> {
  IndexExpr units;
  DataType out_dtype;
  // Quantization related attributes.
  int32_t input_zero_point;
  int32_t kernel_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double kernel_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIDenseAttrs, "relay.attrs.QnnCSIDenseAttrs") {
    TVM_ATTR_FIELD(units).describe("Number of hidden units of the dense transformation.");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(kernel_zero_point).describe("The zero point of the kernel tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(kernel_scale).describe("The kernel tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
  }
};

struct QnnCSIUnaryAttrs : public tvm::AttrsNode<QnnCSIUnaryAttrs> {
  // Quantization related attributes.
  DataType out_dtype;
  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIUnaryAttrs, "relay.attrs.QnnCSIUnaryAttrs") {
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSILeakyReluAttrs : public tvm::AttrsNode<QnnCSILeakyReluAttrs> {
  double alpha;
  // Quantization related attributes.
  DataType out_dtype;
  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSILeakyReluAttrs, "relay.attrs.QnnCSILeakyReluAttrs") {
    TVM_ATTR_FIELD(alpha).describe("Multiplier, use when x in data < 0");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIPReluAttrs : public tvm::AttrsNode<QnnCSIPReluAttrs> {
  int axis;
  // Quantization related attributes.
  DataType out_dtype;
  int32_t input_zero_point;
  int32_t alpha_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double alpha_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIPReluAttrs, "relay.attrs.QnnCSIPReluAttrs") {
    TVM_ATTR_FIELD(axis).describe("The axis at which the input arrays are computed.");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(alpha_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(alpha_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIMaxPool2DAttrs : public tvm::AttrsNode<QnnCSIMaxPool2DAttrs> {
  // Quantization related attributes.
  DataType out_dtype;
  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  bool ceil_mode;
  Array<IndexExpr> pool_size;
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  String layout;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIMaxPool2DAttrs, "relay.attrs.QnnCSIMaxPool2DAttrs") {
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(pool_size)
        .set_default(Array<IndexExpr>({2, 2}))
        .describe("Specifies the filter of the pool.");
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the strides of the pool.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "on both sides for padding number of points");
    TVM_ATTR_FIELD(ceil_mode).describe("ceil mode of pooling.");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Convolution is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIMaxPool2DLocatAttrs : public tvm::AttrsNode<QnnCSIMaxPool2DLocatAttrs> {
  Array<IndexExpr> pool_size;
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  std::string layout;
  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  bool ceil_mode;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIMaxPool2DLocatAttrs, "relay.attrs.QnnCSIMaxPool2DLocatAttrs") {
    TVM_ATTR_FIELD(pool_size).describe("Size of the pooling windows.");
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "two int : bottom, right will use same padding as top, left"
            "four int : padding width in the order of (top, left, bottom, right)");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Pooling is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(ceil_mode).set_default(false).describe(
        "When true, will use ceil instead of floor to compute the output shape.");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIAvgPool2DAttrs : public tvm::AttrsNode<QnnCSIAvgPool2DAttrs> {
  // Quantization related attributes.
  DataType out_dtype;
  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  bool ceil_mode;
  bool count_include_pad;
  Array<IndexExpr> pool_size;
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  std::string layout;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIAvgPool2DAttrs, "relay.attrs.QnnCSIAvgPool2DAttrs") {
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(pool_size)
        .set_default(Array<IndexExpr>({2, 2}))
        .describe("Specifies the filter of the pool.");
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the strides of the pool.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "on both sides for padding number of points");
    TVM_ATTR_FIELD(ceil_mode).describe("ceil mode of pooling.");
    TVM_ATTR_FIELD(count_include_pad)
        .describe("When true, will include padding to compute the average.");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Convolution is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIReshapeAttrs : public tvm::AttrsNode<QnnCSIReshapeAttrs> {
  // Quantization related attributes.
  bool reverse;
  Array<Integer> newshape;
  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIReshapeAttrs, "relay.attrs.QnnCSIReshapeAttrs") {
    TVM_ATTR_FIELD(newshape).describe("The shape of output tensor.");
    TVM_ATTR_FIELD(reverse).describe("");
    TVM_ATTR_FIELD(input_scale).describe("The scale of input tensor.");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of input tensor.");
    TVM_ATTR_FIELD(output_scale).describe("The scale of output tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of output tensor.");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIBroadCastToAttrs : public tvm::AttrsNode<QnnCSIBroadCastToAttrs> {
  // Quantization attributes.
  Array<Integer> shape;
  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIBroadCastToAttrs, "relay.attrs.QnnCSIBroadCastToAttrs") {
    TVM_ATTR_FIELD(shape).describe("The shape of output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The scale of input tensor.");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of input tensor.");
    TVM_ATTR_FIELD(output_scale).describe("The scale of output tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of output tensor.");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIAxisAttrs : public tvm::AttrsNode<QnnCSIAxisAttrs> {
  // Quantization related attributes.
  int32_t axis;
  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIAxisAttrs, "relay.attrs.QnnCSIAxisAttrs") {
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(axis).set_default(-1).describe("The axis to sum over when computing softmax.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIExpandDimsAttrs : public tvm::AttrsNode<QnnCSIExpandDimsAttrs> {
  // Quantization related attributes.
  int32_t axis;
  int32_t num_newaxis;
  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIExpandDimsAttrs, "relay.attrs.QnnCSIExpandDimsAttrs") {
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(axis).set_default(-1).describe("The axis to sum over when computing softmax.");
    TVM_ATTR_FIELD(num_newaxis)
        .describe("Number of axises to be inserted. Should be >= 0.")
        .set_lower_bound(0)
        .set_default(1);
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSILRNAttrs : public tvm::AttrsNode<QnnCSILRNAttrs> {
  int size;
  int axis;
  double bias;
  double alpha;
  double beta;
  // Quantization related attributes.

  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSILRNAttrs, "relay.attrs.QnnCSILRNAttrs") {
    TVM_ATTR_FIELD(size).set_default(5).describe(
        "The size of the local region to be considered for normalization.");
    TVM_ATTR_FIELD(axis).set_default(1).describe("Axis of input data layout channel.");
    TVM_ATTR_FIELD(bias).set_default(2).describe("The offset parameter to avoid division by 0.");
    TVM_ATTR_FIELD(alpha).set_default(0.0001).describe("The scaling parameter.");
    TVM_ATTR_FIELD(beta).set_default(0.75).describe("The exponent parameter.");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
    TVM_ATTR_FIELD(out_dtype).describe("The output tensor dtpe");
  }
};

struct QnnCSIGlobalAvgPoolAttrs : public tvm::AttrsNode<QnnCSIGlobalAvgPoolAttrs> {
  std::string layout;
  // Quantization related attributes.

  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIGlobalAvgPoolAttrs, "relay.attrs.QnnCSIGlobalAvgPoolAttrs") {
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(out_dtype).describe("The output tensor dtype.");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Convolution is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIGlobalMaxPoolAttrs : public tvm::AttrsNode<QnnCSIGlobalMaxPoolAttrs> {
  std::string layout;
  // Quantization related attributes.

  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIGlobalMaxPoolAttrs, "relay.attrs.QnnCSIGlobalMaxPoolAttrs") {
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(out_dtype).describe("The output tensor dtype.");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Convolution is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSITransposeAttrs : public tvm::AttrsNode<QnnCSITransposeAttrs> {
  Array<Integer> axes;
  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSITransposeAttrs, "relay.attrs.QnnCSITransposeAttrs") {
    TVM_ATTR_FIELD(axes).describe("The channel of output tensor.");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIProposalAttrs : public tvm::AttrsNode<QnnCSIProposalAttrs> {
  Array<IndexExpr> scales;
  Array<IndexExpr> ratios;
  int feature_stride;
  double threshold;
  int rpn_pre_nms_top_n;
  int rpn_post_nms_top_n;
  int rpn_min_size;
  bool iou_loss;
  Array<tvm::PrimExpr> input_scales;
  Array<tvm::PrimExpr> input_zero_points;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  int32_t output_zero_point;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIProposalAttrs, "relay.attrs.QnnCSIProposalAttrs") {
    TVM_ATTR_FIELD(input_zero_points).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scales).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(scales)
        .set_default(Array<IndexExpr>({4.0f, 8.0f, 16.0f, 32.0f}))
        .describe("Used to generate anchor windows by enumerating scales");
    TVM_ATTR_FIELD(ratios)
        .set_default(Array<IndexExpr>({0.5f, 1.0f, 2.0f}))
        .describe("Used to generate anchor windows by enumerating ratios");
    TVM_ATTR_FIELD(feature_stride)
        .set_default(16)
        .describe(
            "The size of the receptive field each unit in the convolution layer of the rpn,"
            "for example the product of all stride's prior to this layer.");
    TVM_ATTR_FIELD(threshold).set_default(0.7).describe(
        "IoU threshold of non-maximum suppresion (suppress boxes with IoU >= this threshold)");
    TVM_ATTR_FIELD(rpn_pre_nms_top_n)
        .set_default(6000)
        .describe("Number of top scoring boxes to apply NMS. -1 to use all boxes");
    TVM_ATTR_FIELD(rpn_post_nms_top_n)
        .set_default(300)
        .describe("Number of top scoring boxes to keep after applying NMS to RPN proposals");
    TVM_ATTR_FIELD(rpn_min_size).set_default(16).describe("Minimum height or width in proposal");
    TVM_ATTR_FIELD(iou_loss).set_default(false).describe("Usage of IoU Loss");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIPSROIPoolingAttrs : public tvm::AttrsNode<QnnCSIPSROIPoolingAttrs> {
  double spatial_scale;
  int output_dim;
  int group_size;

  Array<tvm::PrimExpr> input_scales;
  Array<tvm::PrimExpr> input_zero_points;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  int32_t output_zero_point;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIPSROIPoolingAttrs, "relay.attrs.QnnCSIPSROIPoolingAttrs") {
    TVM_ATTR_FIELD(input_zero_points).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scales).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(spatial_scale).describe("Size of map_size/input_data_size.");
    TVM_ATTR_FIELD(output_dim).describe("Output size.");
    TVM_ATTR_FIELD(group_size).describe("Size of roi bin.");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIROIPoolingAttrs : public tvm::AttrsNode<QnnCSIROIPoolingAttrs> {
  Array<IndexExpr> pooled_size;
  double spatial_scale;

  Array<tvm::PrimExpr> input_scales;
  Array<tvm::PrimExpr> input_zero_points;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  double output_scale;
  int32_t output_zero_point;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIROIPoolingAttrs, "relay.attrs.QnnCSIROIPoolingAttrs") {
    TVM_ATTR_FIELD(pooled_size).describe("The output shape");
    TVM_ATTR_FIELD(input_zero_points).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scales).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(spatial_scale).describe("Size of map_size/input_data_size.");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIReduceAttrs : public tvm::AttrsNode<QnnCSIReduceAttrs> {
  Array<Integer> axis;
  bool keepdims;
  bool exclude;

  DataType out_dtype;
  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIReduceAttrs, "relay.attrs.QnnCSIReduceAttrs") {
    TVM_ATTR_FIELD(axis)
        .set_default(NullValue<Array<Integer>>())
        .describe(R"code(The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.

      If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.)code");

    TVM_ATTR_FIELD(keepdims).set_default(false).describe(
        "If this is set to `True`, the reduced axes are left "
        "in the result as dimension with size one.");
    TVM_ATTR_FIELD(exclude).set_default(false).describe(
        "Whether to perform reduction on axis that are NOT in axis instead.");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIPadAttrs : public tvm::AttrsNode<QnnCSIPadAttrs> {
  Array<Array<IndexExpr>> pad_width;
  std::string pad_mode;
  double pad_value;

  DataType out_dtype;
  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIPadAttrs, "relay.attrs.QnnCSIPadAttrs") {
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(pad_width).describe("The pad of out data.");
    TVM_ATTR_FIELD(pad_mode).describe("Pad mode.");
    TVM_ATTR_FIELD(pad_value).describe("The pad of value.");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSISqueezeAttrs : public tvm::AttrsNode<QnnCSISqueezeAttrs> {
  Array<Integer> axis;

  DataType out_dtype;
  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  String layer_name;
  TVM_DECLARE_ATTRS(QnnCSISqueezeAttrs, "relay.attrs.QnnCSISqueezeAttrs") {
    TVM_ATTR_FIELD(axis)
        .set_default(NullValue<Array<Integer>>())
        .describe(R"code(The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.)code");

    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIRequantizeAttrs : public tvm::AttrsNode<QnnCSIRequantizeAttrs> {
  int axis;
  std::string rounding;
  DataType out_dtype;
  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  String layer_name;

  TVM_DECLARE_ATTRS(RequantizeAttrs, "relay.attrs.QnnCSIRequantizeAttrs") {
    TVM_ATTR_FIELD(axis)
        .describe(
            "The output channel axis for channel wise quantization. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
    TVM_ATTR_FIELD(rounding).set_default("UPWARD").describe(
        "Defines the rounding direction when the value is midway between"
        "two representable values. There are two supported modes - UPWARD"
        "or TONEAREST. Both modes behave exactly same except at the"
        "midpoints between the two representable values. At the midpoint,"
        "UPWARD rounds towards positive infinity (for example -1.5 will be"
        "rounded to -1). TONEAREST is the standard rounding where the"
        "value is rounded away from zero at midpoints (for example, -1.5"
        "rounds to -2). More context can be found at following gblic manual"
        "https://www.gnu.org/software/libc/manual/html_node/Rounding.html.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

/*! \brief Attributes used in Unpooling operators */
struct QnnCSIUnPoolingAttrs : public tvm::AttrsNode<QnnCSIUnPoolingAttrs> {
  Array<IndexExpr> scales;
  Array<IndexExpr> out_padding;

  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  std::string layout;
  DataType out_dtype;
  String layer_name;
  TVM_DECLARE_ATTRS(QnnCSIUnPoolingAttrs, "relay.attrs.QnnCSIUnPoolingAttrs") {
    TVM_ATTR_FIELD(scales).describe("Output channel size of psroipooling.");
    TVM_ATTR_FIELD(out_padding).describe("The pad of out data");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of data and weight. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Convolution is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIUpSamplingAttrs : public tvm::AttrsNode<QnnCSIUpSamplingAttrs> {
  double scale_h;
  double scale_w;
  int32_t align_corners;
  std::string method;

  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  std::string layout;
  String layer_name;
  TVM_DECLARE_ATTRS(QnnCSIUpSamplingAttrs, "relay.attrs.QnnCSIUpSamplingAttrs") {
    TVM_ATTR_FIELD(scale_h).describe("Output scale of upsampling.");
    TVM_ATTR_FIELD(scale_w).describe("Output scale of upsampling.");
    TVM_ATTR_FIELD(method)
        .set_default("nearest_neighbor")
        .describe(
            "Specify the mode to use for scaling."
            "nearest_neighbor -  Nearest Neighbor"
            "bilinear - Bilinear Interpolation"
            "bicubic - Bicubic Interpolation");
    TVM_ATTR_FIELD(align_corners)
        .set_default(false)
        .describe("Should be true to preserve the values at the corner pixels");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of data and weight. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Convolution is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIBatchNormAttrs : public tvm::AttrsNode<QnnCSIBatchNormAttrs> {
  int axis;
  double epsilon;
  bool center;
  bool scale;

  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  String layer_name;
  TVM_DECLARE_ATTRS(QnnCSIBatchNormAttrs, "relay.attrs.QnnCSIBatchNormAttrs") {
    TVM_ATTR_FIELD(axis).describe("Specify which shape axis denotes the channel.").set_default(1);
    TVM_ATTR_FIELD(epsilon)
        .describe("Small float added to variance to avoid dividing by zero")
        .set_default(1e-5);
    TVM_ATTR_FIELD(center)
        .describe("If True, add offset of beta to normalized tensor. If False, beta is ignored")
        .set_default(true);
    TVM_ATTR_FIELD(scale)
        .describe(
            "If True, multiply by gamma. If False, gamma is not used. "
            "When the next layer is piecewise linear (also, e.g., nn.relu), "
            "this can be disabled since the scaling will be done by the next layer.")
        .set_default(true);
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIStridedSliceAttrs : public tvm::AttrsNode<QnnCSIStridedSliceAttrs> {
  Array<Integer> begin;
  Array<Integer> end;
  Array<Integer> strides;

  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIStridedSliceAttrs, "relay.attrs.QnnCSIStridedSliceAttrs") {
    TVM_ATTR_FIELD(begin).describe("Indices for begin of slice, begin index is also inclusive");
    TVM_ATTR_FIELD(end).describe("Indices for end of slice, end index is exclusive");
    TVM_ATTR_FIELD(strides).set_default(Array<Integer>({})).describe("Stride values of the slice");

    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
  }
};

struct QnnCSISplitAttrs : public tvm::AttrsNode<QnnCSISplitAttrs> {
  ObjectRef indices_or_sections;
  int axis;

  double input_scale;
  int32_t input_zero_point;
  Array<tvm::PrimExpr> output_scales;
  Array<Integer> output_zero_points;
  DataType out_dtype;
  String layer_name;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;

  TVM_DECLARE_ATTRS(QnnCSISplitAttrs, "relay.attrs.QnnCSISplitAttrs") {
    TVM_ATTR_FIELD(indices_or_sections)
        .describe(
            "Indices or sections to split into. Accepts an int or a tuple"
            "If indices_or_sections is an integer, the input will be divided equally"
            "along given axis. If such a split is not possible, an error is raised."
            "If indices_or_sections is a tuple of sorted integers,"
            "the entries indicate where along axis the array is split.");
    TVM_ATTR_FIELD(axis).set_default(0).describe("the axis to be splitted.");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_points).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scales).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting.");
  }
};

struct QnnCSISegmentAttrs : public tvm::AttrsNode<QnnCSISegmentAttrs> {
  int32_t length;
  // Quantization related attributes.
  DataType out_dtype;
  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSISegmentAttrs, "relay.attrs.QnnCSISegmentAttrs") {
    TVM_ATTR_FIELD(length).describe("The length of output.");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting.");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

/*! \brief Attributes for 3D avg pool operator */
struct QnnCSIAvgPool3DAttrs : public tvm::AttrsNode<QnnCSIAvgPool3DAttrs> {
  Array<IndexExpr> pool_size;
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  std::string layout;
  bool ceil_mode;
  bool count_include_pad;

  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;
  TVM_DECLARE_ATTRS(QnnCSIAvgPool3DAttrs, "relay.attrs.QnnCSIAvgPool3DAttrs") {
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(pool_size).describe("Size of the pooling windows.");
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "three int : back, bottom, right will use same padding as front, top, left"
            "six int : padding width in the order of (front, top, left, back, bottom, right)");
    TVM_ATTR_FIELD(layout).set_default("NCDHW").describe(
        "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
        "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
        "dimensions respectively. Pooling is applied on the 'D', 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(ceil_mode).set_default(false).describe(
        "When true, will use ceil instead of floor to compute the output shape.");
    TVM_ATTR_FIELD(count_include_pad)
        .set_default(false)
        .describe("When true, will include padding to compute the average");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

/*! \brief Attributes for 3D avg pool operator */
struct QnnCSIMaxPool3DAttrs : public tvm::AttrsNode<QnnCSIMaxPool3DAttrs> {
  Array<IndexExpr> pool_size;
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  std::string layout;
  bool ceil_mode;
  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIMaxPool3DAttrs, "relay.attrs.QnnCSIMaxPool3DAttrs") {
    TVM_ATTR_FIELD(pool_size).describe("Size of the pooling windows.");
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "three int : back, bottom, right will use same padding as front, top, left"
            "six int : padding width in the order of (front, top, left, back, bottom, right)");
    TVM_ATTR_FIELD(layout).set_default("NCDHW").describe(
        "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
        "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
        "dimensions respectively. Pooling is applied on the 'D', 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(ceil_mode).set_default(false).describe(
        "When true, will use ceil instead of floor to compute the output shape.");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

/*! \brief Attributes used in image crop_and_resize operator */
struct QnnCSICropResizeAttrs : public tvm::AttrsNode<QnnCSICropResizeAttrs> {
  Array<IndexExpr> crop_size;
  std::string layout;
  std::string method;
  double extrapolation_value;
  DataType out_dtype;

  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  String layer_name;
  TVM_DECLARE_ATTRS(QnnCSICropResizeAttrs, "relay.attrs.QnnCSICropResizeAttrs") {
    TVM_ATTR_FIELD(crop_size).set_default(NullValue<Array<IndexExpr>>()).describe("Target Size.");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Resize is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(method)
        .set_default("bilinear")
        .describe(
            "Specify the mode to use for scaling."
            "nearest_neighbor -  Nearest Neighbor"
            "bilinear - Bilinear Interpolation");
    TVM_ATTR_FIELD(extrapolation_value)
        .set_default(0.0)
        .describe("Specify value for extrapolation.");
    TVM_ATTR_FIELD(out_dtype).set_default(NullValue<DataType>()).describe("Output data type.");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

/*! \brief Attributes used in subpixel operators */
struct QnnCSISubPixelAttrs : public tvm::AttrsNode<QnnCSISubPixelAttrs> {
  int block_size;
  std::string layout;
  std::string mode;

  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSISubPixelAttrs, "relay.attrs.QnnCSISubPixelAttrs") {
    TVM_ATTR_FIELD(block_size)
        .describe("The size of subpixel blocks to compose or decompose.")
        .set_default(1);
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively.");
    TVM_ATTR_FIELD(mode).set_default("DCR").describe(
        "Indicates order in which channels are accessed. Must be one of"
        "DCR or CDR.");
    TVM_ATTR_FIELD(out_dtype).set_default(NullValue<DataType>()).describe("Output data type.");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};  // struct SubPixelAttrs

/*! \brief Attributes for Clip operator */
struct QnnCSIClipAttrs : public tvm::AttrsNode<QnnCSIClipAttrs> {
  double a_min;
  double a_max;

  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIClipAttrs, "relay.attrs.QnnCSIClipAttrs") {
    TVM_ATTR_FIELD(a_min).describe("The minimum clip value.");
    TVM_ATTR_FIELD(a_max).describe("The maximum clip value.");
    TVM_ATTR_FIELD(out_dtype).set_default(NullValue<DataType>()).describe("Output data type.");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

/*! \brief Attributes for dilation2d operator */
struct QnnCSIDilation2DAttrs : public tvm::AttrsNode<QnnCSIDilation2DAttrs> {
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilations;
  std::string data_layout;
  std::string kernel_layout;
  DataType out_dtype;

  int32_t input_zero_point;
  int32_t kernel_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double kernel_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  String layer_name;
  TVM_DECLARE_ATTRS(QnnCSIDilation2DAttrs, "relay.attrs.QnnCSIDilation2DAttrs") {
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the strides of the sliding window. [stride_height, stride_width].");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "two int : bottom, right will use same padding as top, left"
            "four int : padding width in the order of (top, left, bottom, right)");
    TVM_ATTR_FIELD(dilations)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the dilation rate to use. [dilation_height, dilation_width]");
    TVM_ATTR_FIELD(data_layout)
        .set_default("NCHW")
        .describe(
            "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Convolution is applied on the 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("IHW")
        .describe(
            "Dimension ordering of weight. Can be 'IHW', 'HWI', etc."
            "'I', 'H', 'W' stands for input_channel, height, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(kernel_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(kernel_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

/*! \brief Attributes used in extract_image_patches operators */
struct QnnCSIExtractImagePatchesAttrs : public tvm::AttrsNode<QnnCSIExtractImagePatchesAttrs> {
  Array<IndexExpr> ksizes;
  Array<IndexExpr> strides;
  Array<IndexExpr> rates;
  Array<IndexExpr> padding;
  std::string layout;

  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;
  TVM_DECLARE_ATTRS(QnnCSIExtractImagePatchesAttrs, "relay.attrs.QnnCSIExtractImagePatchesAttrs") {
    TVM_ATTR_FIELD(ksizes).describe("kernel size.");
    TVM_ATTR_FIELD(strides).describe("stride size.");
    TVM_ATTR_FIELD(rates).describe("The dilated size");
    TVM_ATTR_FIELD(padding).describe("The padding size");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of data and weight. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Convolution is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

/*! \brief Attributes used in fill operators */
struct QnnCSIFullAttrs : public tvm::AttrsNode<QnnCSIFullAttrs> {
  Array<Integer> shape;

  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSIFullAttrs, "relay.attrs.QnnCSIFullAttrs") {
    TVM_ATTR_FIELD(shape).describe("fill shape.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

/*! \brief Attributes used in fill operators */
struct QnnCSITakeAttrs : public tvm::AttrsNode<QnnCSITakeAttrs> {
  Integer axis;
  String mode;

  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSITakeAttrs, "relay.attrs.QnnCSITakeAttrs") {
    TVM_ATTR_FIELD(axis)
        .set_default(NullValue<Integer>())
        .describe("The axis over which to select values.");
    TVM_ATTR_FIELD(mode).set_default("clip").describe(
        "Specify how out-of-bound indices will behave."
        "clip - clip to the range (default)"
        "wrap - wrap around the indices"
        "fast - no clip or wrap around (user must make sure indices are in-bound)");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

/*! \brief Attributes used in fill operators */
struct QnnCSINonMaximumSuppressionAttrs : public tvm::AttrsNode<QnnCSINonMaximumSuppressionAttrs> {
  Optional<Integer> max_output_size;
  double iou_threshold;
  bool force_suppress;
  int top_k;
  int coord_start;
  int score_index;
  int id_index;
  bool return_indices;
  bool invalid_to_bottom;

  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;

  TVM_DECLARE_ATTRS(QnnCSINonMaximumSuppressionAttrs,
                    "relay.attrs.QnnCSINonMaximumSuppressionAttrs") {
    TVM_ATTR_FIELD(max_output_size).describe("Max number of output valid boxes for each instance.");
    TVM_ATTR_FIELD(iou_threshold)
        .set_default(0.5)
        .describe("Non-maximum suppression iou threshold.");
    TVM_ATTR_FIELD(force_suppress)
        .set_default(false)
        .describe("Suppress all detections regardless of class_id.");
    TVM_ATTR_FIELD(top_k).set_default(-1).describe(
        "Keep maximum top k detections before nms, -1 for no limit.");
    TVM_ATTR_FIELD(coord_start)
        .set_default(2)
        .describe("Start index of the consecutive 4 coordinates.");
    TVM_ATTR_FIELD(score_index).set_default(1).describe("Index of the scores/confidence of boxes.");
    TVM_ATTR_FIELD(id_index).set_default(0).describe("Axis index of id.");
    TVM_ATTR_FIELD(return_indices)
        .set_default(true)
        .describe("Whether to return box indices in input data.");
    TVM_ATTR_FIELD(invalid_to_bottom)
        .set_default(false)
        .describe("Whether to move all invalid bounding boxes to the bottom.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

/*! \brief Attributes used in topk operators */
struct QnnCSITopKAttrs : public tvm::AttrsNode<QnnCSITopKAttrs> {
  int32_t k;
  int32_t axis;
  bool is_ascend;
  std::string ret_type;
  DataType dtype;

  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;
  TVM_DECLARE_ATTRS(QnnCSITopKAttrs, "relay.attrs.QnnCSITopKAttrs") {
    TVM_ATTR_FIELD(k).describe("Number of top elements to select");
    TVM_ATTR_FIELD(axis).set_default(-1).describe("Axis along which to sort the input tensor.");
    TVM_ATTR_FIELD(ret_type).set_default("both").describe(
        "The return type [both, values, indices]."
        "both - return both top k data and indices."
        "values - return top k data only."
        "indices - return top k indices only.");
    TVM_ATTR_FIELD(is_ascend).set_default(false).describe(
        "Whether to sort in ascending or descending order."
        "By default, sort in descending order");
    TVM_ATTR_FIELD(dtype)
        .set_default(NullValue<DataType>())
        .describe("Data type of the output indices.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};

struct QnnCSIOneHotAttrs : public tvm::AttrsNode<QnnCSIOneHotAttrs> {
  int depth;
  int axis;
  DataType dtype;

  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;
  TVM_DECLARE_ATTRS(QnnCSIOneHotAttrs, "relay.attrs.QnnCSIOneHotAttrs") {
    TVM_ATTR_FIELD(depth).set_default(1).describe("Depth of the one hot dimension.");
    TVM_ATTR_FIELD(axis).set_default(-1).describe("Axis to fill.");
    TVM_ATTR_FIELD(dtype).set_default(NullValue<DataType>()).describe("Output data type.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};  // struct OneHotAttrs

struct QnnCSITileAttrs : public tvm::AttrsNode<QnnCSITileAttrs> {
  Array<Integer> reps;

  int32_t input_zero_point;
  int32_t output_zero_point;
  double input_scale;
  double output_scale;
  Array<IndexExpr> max_values;
  Array<IndexExpr> min_values;
  DataType out_dtype;
  String layer_name;
  TVM_DECLARE_ATTRS(QnnCSITileAttrs, "relay.attrs.QnnCSITileAttrs") {
    TVM_ATTR_FIELD(reps).describe(
        "The number of times for repeating the tensor a."
        "Each dim sizeof reps must be a positive integer.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point).describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point).describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale).describe("The input tensor scale.");
    TVM_ATTR_FIELD(output_scale).describe("The output tensor scale.");
    TVM_ATTR_FIELD(max_values).describe("max value of inputs and output.");
    TVM_ATTR_FIELD(min_values).describe("min value of inputs and output.");
    TVM_ATTR_FIELD(layer_name).describe("The name of this layer");
  }
};  // struct TileAttrs

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_QNN_ATTRS_H_
