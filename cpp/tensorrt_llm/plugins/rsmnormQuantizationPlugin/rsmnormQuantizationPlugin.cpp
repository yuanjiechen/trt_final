/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tensorrt_llm/plugins/rsmnormQuantizationPlugin/rsmnormQuantizationPlugin.h"
#include "tensorrt_llm/kernels/reorder_rsmlayernorm.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using nvinfer1::plugin::RsmnormQuantizationPluginCreator;
using nvinfer1::plugin::RsmnormQuantizationPlugin;

static const char* RSMNORM_QUANTIZATION_PLUGIN_VERSION{"1"};
static const char* RSMNORM_QUANTIZATION_PLUGIN_NAME{"RmsnormQuantization"};
PluginFieldCollection RsmnormQuantizationPluginCreator::mFC{};
std::vector<PluginField> RsmnormQuantizationPluginCreator::mPluginAttributes;

RsmnormQuantizationPlugin::RsmnormQuantizationPlugin(
    float eps, bool dynamicActivationScaling, bool useDiffOfSquares, nvinfer1::DataType type)
    : mEps(eps)
    , mDynActScaling(dynamicActivationScaling)
    , mUseDiffOfSquares(useDiffOfSquares)
    , mType(type)
{
}

// Parameterized constructor
RsmnormQuantizationPlugin::RsmnormQuantizationPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mEps);
    read(d, mUseDiffOfSquares);
    read(d, mDynActScaling);
    read(d, mType);
    PLUGIN_ASSERT(d == a + length);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* RsmnormQuantizationPlugin::clone() const noexcept
{
    auto* plugin = new RsmnormQuantizationPlugin(mEps, mUseDiffOfSquares, mDynActScaling, mType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs RsmnormQuantizationPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    if (outputIndex == 0)
    {
        // Quantized output
        return inputs[outputIndex];
    }

    // Dynamic scaling output if enabled
    try
    {
        PLUGIN_ASSERT(outputIndex == 1);
        DimsExprs ret;
        ret.nbDims = inputs[0].nbDims;
        for (int di = 0; di < ret.nbDims - 1; ++di)
        {
            ret.d[di] = inputs[0].d[di];
        }
        ret.d[ret.nbDims - 1] = exprBuilder.constant(1);
        return ret;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool RsmnormQuantizationPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    const int totalPoses = 6 + static_cast<int>(mDynActScaling);
    PLUGIN_ASSERT(0 <= pos && pos < totalPoses);
    PLUGIN_ASSERT(nbInputs == 4);
    // std::cout << pos << "  " << int32_t(inOut[pos].type )<< "  " << int32_t(inOut[pos].format) << std::endl;
    if (pos < nbInputs)
    {   
        
        switch (pos)
        {
        case 0: 
            // std::cout << "case0  " << int32_t((inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR)) <<std::endl;
            return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
        case 1: 
            // std::cout << "case1  " << int32_t((inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR)) <<std::endl;
            return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
        case 2: 
            // std::cout << "case2  " << int32_t((inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR)) <<std::endl;
            return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
        case 3: 
            // std::cout << "case3  " << int32_t((inOut[pos].type == nvinfer1::DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR)) <<std::endl;
            return (inOut[pos].type == nvinfer1::DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR);
        }
    }
    if (pos == 4)
    {
        // Quantized output
        // std::cout << "case4  " << int32_t((inOut[pos].type == nvinfer1::DataType::kINT8)) <<std::endl;
        return (inOut[pos].type == nvinfer1::DataType::kHALF) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    // Dynamic scaling if enabled
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void RsmnormQuantizationPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t RsmnormQuantizationPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int RsmnormQuantizationPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // inputs
    //     input [M(*), N]
    //     weight [N, ]
    //     bias [N, ]
    //     scale_to_int [1]
    // outputs
    //     output [M(*), N]
    //     dynamic_scaling [M(*), 1] (optional output)

    int m = 1;
    int b,c;
    b = inputDesc[0].dims.d[0];
    c = inputDesc[0].dims.d[1];
    for (int i = 0; i < inputDesc[0].dims.nbDims - 1; ++i)
    {
        m *= inputDesc[0].dims.d[i];
    }
    const int n = inputDesc[1].dims.d[0];


    half* output = reinterpret_cast<half*>(outputs[0]);
    float* dynamic_scale = mDynActScaling ? reinterpret_cast<float*>(outputs[1]) : nullptr;

    if (mType == DataType::kHALF)
    {
        const half* input = reinterpret_cast<const half*>(inputs[0]);
        const half* weight = reinterpret_cast<const half*>(inputs[1]);
        // const half* zero_point = reinterpret_cast<const half*>(inputs[2]);
        const half* scale = reinterpret_cast<const half*>(inputs[2]);
        const long* dst_index = reinterpret_cast<const long*>(inputs[3]);
        reorder_rsm_norm_fp16<half>((const half*)input,(half*)nullptr, (const half*)weight,(const half*)scale,(const half*)nullptr, dst_index, output,mEps, b, c);
    }
    else if (mType == DataType::kFLOAT)
    {
        perror("Not support float\n");
        return 0;
        // const float* input = reinterpret_cast<const float*>(inputs[0]);
        // const float* weight = reinterpret_cast<const float*>(inputs[1]);
        // const float* bias = reinterpret_cast<const float*>(inputs[2]);
        // invokeGeneralLayerNorm(
        //     (float*) nullptr, input, weight, bias, mEps, m, n, stream, mUseDiffOfSquares, scale, dynamic_scale, output);
        // reorder_rsm_norm_fp16(
        //     (float*) nullptr, input, weight, bias, mEps, m, n, stream, mUseDiffOfSquares, scale, dynamic_scale, output);
    }

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType RsmnormQuantizationPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert((mDynActScaling && index < 2) || (~mDynActScaling && index == 0));
    if (index == 0)
    {
        // Output 0 quantized output of layer norm
        return nvinfer1::DataType::kINT8;
    }
    // Output 1 dynamic act scaling
    return nvinfer1::DataType::kFLOAT;
}

// IPluginV2 Methods

const char* RsmnormQuantizationPlugin::getPluginType() const noexcept
{
    return RSMNORM_QUANTIZATION_PLUGIN_NAME;
}

const char* RsmnormQuantizationPlugin::getPluginVersion() const noexcept
{
    return RSMNORM_QUANTIZATION_PLUGIN_VERSION;
}

int RsmnormQuantizationPlugin::getNbOutputs() const noexcept
{
    return 1 + static_cast<int>(mDynActScaling);
}

int RsmnormQuantizationPlugin::initialize() noexcept
{
    return 0;
}

void RsmnormQuantizationPlugin::terminate() noexcept {}

size_t RsmnormQuantizationPlugin::getSerializationSize() const noexcept
{
    return sizeof(mEps) + sizeof(mUseDiffOfSquares) + sizeof(mDynActScaling) + sizeof(mType);
}

void RsmnormQuantizationPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mEps);
    write(d, mUseDiffOfSquares);
    write(d, mDynActScaling);
    write(d, mType);
    assert(d == a + getSerializationSize());
}

void RsmnormQuantizationPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void RsmnormQuantizationPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* RsmnormQuantizationPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////

RsmnormQuantizationPluginCreator::RsmnormQuantizationPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1e-5f));
    mPluginAttributes.emplace_back(PluginField("use_diff_of_squares", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("dyn_act_scaling", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* RsmnormQuantizationPluginCreator::getPluginName() const noexcept
{
    return RSMNORM_QUANTIZATION_PLUGIN_NAME;
}

const char* RsmnormQuantizationPluginCreator::getPluginVersion() const noexcept
{
    return RSMNORM_QUANTIZATION_PLUGIN_VERSION;
}

const PluginFieldCollection* RsmnormQuantizationPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* RsmnormQuantizationPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    float eps;
    nvinfer1::DataType type;
    bool useDiffOfSquares;
    bool dynamicActivationScaling;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "eps"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            eps = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "dyn_act_scaling"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            dynamicActivationScaling = static_cast<bool>(*(static_cast<const bool*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "use_diff_of_squares"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            useDiffOfSquares = static_cast<bool>(*(static_cast<const bool*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new RsmnormQuantizationPlugin(eps, useDiffOfSquares, dynamicActivationScaling, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* RsmnormQuantizationPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call RsmnormQuantizationPlugin::destroy()
    try
    {
        auto* obj = new RsmnormQuantizationPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void RsmnormQuantizationPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* RsmnormQuantizationPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
