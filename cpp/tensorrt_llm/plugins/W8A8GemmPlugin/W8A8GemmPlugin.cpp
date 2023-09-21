/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#include "tensorrt_llm/plugins/W8A8GemmPlugin/W8A8GemmPlugin.h"
#include <numeric>

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels::cutlass_kernels;
using nvinfer1::plugin::W8A8GemmPluginCreator;
using nvinfer1::plugin::W8A8GemmPlugin;

static const char* SQ_GEMM_PLUGIN_VERSION{"1"};
static const char* SQ_GEMM_PLUGIN_NAME{"W8A8Gemm"};
PluginFieldCollection W8A8GemmPluginCreator::mFC{};
std::vector<PluginField> W8A8GemmPluginCreator::mPluginAttributes;

W8A8GemmPlugin::W8A8GemmPlugin(bool perChannelScaling, bool perTokenScaling, nvinfer1::DataType type)
{
    init(perChannelScaling, perTokenScaling, type);
}

// Parameterized constructor
W8A8GemmPlugin::W8A8GemmPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    bool perChannelScaling = false, perTokenScaling = false;
    nvinfer1::DataType type;
    read(d, perChannelScaling);
    read(d, perTokenScaling);
    read(d, type);
    read(d, mMinM);
    read(d, mMaxM);
    read(d, mN);
    read(d, mK);
    int selectedMapSize;
    read(d, selectedMapSize);
    perfMapType selectedTacticsMap;
    for (int ii = 0; ii < selectedMapSize; ++ii)
    {
        std::pair<int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig> config;
        read(d, config);
        selectedTacticsMap.insert(config);
    }
    init(perChannelScaling, perTokenScaling, type);
    m_sqGemmRunner->setSelectedTactics(selectedTacticsMap);
    m_sqGemmRunner->setMaxM(mMaxM);
    PLUGIN_ASSERT(d == a + length);
}

void W8A8GemmPlugin::setSelectedTactics(const perfMapType& selectedTacticsMap)
{
    m_sqGemmRunner->setSelectedTactics(selectedTacticsMap);
}

void W8A8GemmPlugin::init(bool perChannelScaling, bool perTokenScaling, nvinfer1::DataType type)
{
    mType = type;
    if (mType == nvinfer1::DataType::kHALF)
    {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<half>>();
    }
    else if (mType == nvinfer1::DataType::kFLOAT)
    {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<float>>();
    }
    else if (mType == nvinfer1::DataType::kINT32)
    {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<int32_t>>();
    }
    else
    {
        // TODO (nkorobov): add bf16 support
        PLUGIN_ASSERT(false);
    }
    m_quantOption = QuantOption::make(perChannelScaling, perTokenScaling);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* W8A8GemmPlugin::clone() const noexcept
{
    auto* plugin
        = new W8A8GemmPlugin(m_quantOption.hasPerChannelScaling(), m_quantOption.hasPerTokenScaling(), mType);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->setProblemSize(mMinM, mMaxM, mN, mK);
    plugin->setSelectedTactics(m_sqGemmRunner->getSelectedTactics());
    plugin->setMaxM(m_sqGemmRunner->getMaxM());
    return plugin;
}

void W8A8GemmPlugin::setMaxM(int maxM)
{
    mMaxM = maxM;
    m_sqGemmRunner->setMaxM(maxM);
}

nvinfer1::DimsExprs W8A8GemmPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_ASSERT(nbInputs == 4);
        PLUGIN_ASSERT(outputIndex == 0);
        const int nbDimsA = inputs[0].nbDims;
        PLUGIN_ASSERT(nbDimsA >= 2);
        DimsExprs ret;
        ret.nbDims = nbDimsA;
        for (int ii = 0; ii < nbDimsA - 1; ++ii)
        {
            ret.d[ii] = inputs[0].d[ii];
        }
        ret.d[nbDimsA - 1] = inputs[1].d[0];
        return ret;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool W8A8GemmPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    switch (pos)
    {
    case 0:
        // activation
        return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
    case 1:
        // weights
        // FIXME
        // Dirty hack to overcome TRT int8 limitatition with plugins
        // Weights are required to be fp32, but will be reinterpreted as int8 in enqueue
        // Weights stored in checkpoint should have int8 type
        // Because of the reinterpretation, input weights have shape 4 times smaller than required
        // in_channels has to be divisible by 4
        return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
    case 2:
        // scales channels
    case 3:
        // scales tokens
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    case 4:
        // out
        return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
    default:
        // Never should be here
        assert(false);
        return false;
    }
}

void W8A8GemmPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    mMinM = std::accumulate(in[0].min.d, in[0].min.d + in[0].min.nbDims - 1, 1, std::multiplies<int>());
    mMaxM = std::accumulate(in[0].max.d, in[0].max.d + in[0].max.nbDims - 1, 1, std::multiplies<int>());

    const int maxK = in[0].max.d[in[0].max.nbDims - 1];
    const int maxN = in[1].max.d[0];
    const int minK = in[0].min.d[in[0].min.nbDims - 1];
    const int minN = in[1].min.d[0];

    TLLM_CHECK_WITH_INFO(minN == maxN, "Variable out channels is not allowed");
    TLLM_CHECK_WITH_INFO(minK == maxK, "Variable in channels is not allowed");

    mK = maxK;
    mN = maxN;

    m_workspaceMaxSize = m_sqGemmRunner->getWorkspaceSize(mMaxM, maxN, maxK);
}

size_t W8A8GemmPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return m_workspaceMaxSize;
}

int W8A8GemmPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // inputs
    //     mat1           [M(*), K]
    //     mat2           [N, K]
    //     scale_tokens   [M, 1] if has_per_token_scaling else [1, 1]
    //     scale_channels [1, N] if has_per_channel_scaling else [1, 1]
    // outputs
    //     mat [M(*), N]
    int m = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m *= inputDesc[0].dims.d[ii];
    }
    const int n = inputDesc[1].dims.d[0];
    // std::cout << n << std::endl;
    const int k = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
    const int wsSize = m_sqGemmRunner->getWorkspaceSize(m, n, k);

    m_sqGemmRunner->gemm(reinterpret_cast<const int8_t*>(inputs[0]), reinterpret_cast<const int8_t*>(inputs[1]),
        m_quantOption, reinterpret_cast<const float*>(inputs[3]), reinterpret_cast<const float*>(inputs[2]),
        reinterpret_cast<void*>(outputs[0]), m, n, k, reinterpret_cast<char*>(workspace), wsSize, stream);

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType W8A8GemmPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    return mType;
}

// IPluginV2 Methods

const char* W8A8GemmPlugin::getPluginType() const noexcept
{
    return SQ_GEMM_PLUGIN_NAME;
}

const char* W8A8GemmPlugin::getPluginVersion() const noexcept
{
    return SQ_GEMM_PLUGIN_VERSION;
}

int W8A8GemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int W8A8GemmPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void W8A8GemmPlugin::terminate() noexcept {}

size_t W8A8GemmPlugin::getSerializationSize() const noexcept
{
    const auto& selectedTactics = m_sqGemmRunner->getSelectedTactics();
    return 2 * sizeof(bool) +        // Per token + per channel flags
        sizeof(nvinfer1::DataType) + // dtype
        4 * sizeof(int) +            // Problem sizes (minM, maxM, N, K)
        sizeof(int) +                // selected tactics constainer num of elems
        selectedTactics.size()
        * sizeof(
            std::pair<int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig>); // selected tactics container size
}

void W8A8GemmPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, m_quantOption.hasPerChannelScaling());
    write(d, m_quantOption.hasPerTokenScaling());
    write(d, mType);
    write(d, mMinM);
    write(d, m_sqGemmRunner->getMaxM());
    write(d, mN);
    write(d, mK);
    const auto& selectedTacticsMap = m_sqGemmRunner->getSelectedTactics();
    write(d, static_cast<int>(selectedTacticsMap.size()));
    for (const auto& pair : selectedTacticsMap)
    {
        write(d, pair);
    }
    assert(d == a + getSerializationSize());
}

void W8A8GemmPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void W8A8GemmPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* W8A8GemmPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void W8A8GemmPlugin::setProblemSize(int minM, int maxM, int n, int k)
{
    mMinM = minM;
    mMaxM = maxM;
    mN = n;
    mK = k;
}

void W8A8GemmPlugin::allocateTmpData()
{
    cudaMalloc(&mATmp, mMaxM * mK * sizeof(int8_t));
    cudaMalloc(&mBTmp, mN * mK * sizeof(int8_t));
    cudaMalloc(&mCTmp, mMaxM * mN * (mType == nvinfer1::DataType::kFLOAT ? 4 : 2));
    cudaMalloc(&mAlphaRowTmp, mMaxM * sizeof(float));
    cudaMalloc(&mAlphaColTmp, mN * sizeof(float));
    cudaMalloc(&mWorkspaceTmp, m_sqGemmRunner->getWorkspaceSize(mMaxM, mN, mK));
}

void W8A8GemmPlugin::freeTmpData()
{
    cudaFree(mATmp);
    cudaFree(mBTmp);
    cudaFree(mCTmp);
    cudaFree(mAlphaRowTmp);
    cudaFree(mAlphaColTmp);
    cudaFree(mWorkspaceTmp);
}

void W8A8GemmPlugin::configGemm()
{
    if (mMaxM == -1 || mMinM == -1 || mN == -1 || mK == -1)
    {
        return;
    }
    if (!m_sqGemmRunner->hasSelectedTactics())
    {
        allocateTmpData();
        m_sqGemmRunner->profileGemms(
            m_quantOption, mMinM, mMaxM, mN, mK, mATmp, mBTmp, mCTmp, mAlphaColTmp, mAlphaRowTmp, mWorkspaceTmp);
        m_sqGemmRunner->setMaxM(mMaxM);
        freeTmpData();
    }
}

///////////////

W8A8GemmPluginCreator::W8A8GemmPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("has_per_channel_scaling", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("has_per_token_scaling", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* W8A8GemmPluginCreator::getPluginName() const noexcept
{
    return SQ_GEMM_PLUGIN_NAME;
}

const char* W8A8GemmPluginCreator::getPluginVersion() const noexcept
{
    return SQ_GEMM_PLUGIN_VERSION;
}

const PluginFieldCollection* W8A8GemmPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* W8A8GemmPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    bool perTokenScaling, perChannelScaling;
    nvinfer1::DataType type;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "has_per_channel_scaling"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            perChannelScaling = static_cast<bool>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "has_per_token_scaling"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            perTokenScaling = static_cast<bool>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new W8A8GemmPlugin(perChannelScaling, perTokenScaling, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* W8A8GemmPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call W8A8GemmPlugin::destroy()
    try
    {
        auto* obj = new W8A8GemmPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void W8A8GemmPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* W8A8GemmPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
