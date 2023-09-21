# trt2023

## 总述

介绍本工作是 [NVIDIA TensorRT Hackathon 2023](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/Hackathon2023/HackathonGuide.md) 的参赛题目，具体选题是2+4，我们想使用tensorrt-llm实现rptq量化，并使他运行在w4a4精度下
原模型地址：[https://github.com/hahnyuan/RPTQ4LLM](https://github.com/hahnyuan/RPTQ4LLM)
这是一个图文对话模型，输入图片和问题，即可和模型进行对话，交流关于图片的内容，模型会记住问答的上下文并理解图片的内容，以下为demo场景图片
优化效果：40个测试数据，torch 145 秒，trt 115 秒

1. 完成度：使用trtllm实现新模型，使trtllm支持图片特征输入，在minigpt4上实现rptq(目前进度fake量化可行)，实现rmsnorm plugin，改进smoothquant plugin中存在的bug，找到trtllm的bug
2. 总结：minigpt4 rptq int8部分工作我们大约进行了半个月的尝试，先后尝试cutlass，cublas两种矩阵计算库，多种参数类型组合，不同plugin参数配置，但是仍然无法得到正确的输出，并不确定是否是trt或者量化过程出现的问题。如果还有时间，我们会替换原本rptq的量化方法ema-minmax到基本的minmax再做尝试

### 准备工作

1. 容器环境：cuda 12.1，tensorrt9.0.0，主文件夹：./examples/MiniGPT4
2. clone github代码[https://github.com/yuanjiechen/trt_final](https://github.com/yuanjiechen/trt_final)
3. 根据[https://github.com/Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)的指令，下载llama-7b和delta模型，合并成vicuna-7b模型，放置于./examples/MiniGPT-4/weight目录下；从相同仓库下载minigpt4 checkpoint，放置于./examples/MiniGPT-4/parameter目录下，文件名默认为prerained_minigpt4_7b.pth
4. 从[https://drive.google.com/file/d/1DhPCZcchixTrFV4N69WFK23pTcdk0B45/view?usp=drive_link](https://drive.google.com/file/d/1DhPCZcchixTrFV4N69WFK23pTcdk0B45/view?usp=drive_link)下载vocab.npy放置于./examples/MiniGPT-4/vocab.npy，这是字典文件
5. 从[https://drive.google.com/file/d/1MJsgNSUDkRGhmssNnv3foR3K4gtU_BPz/view?usp=drive_link](https://drive.google.com/file/d/1MJsgNSUDkRGhmssNnv3foR3K4gtU_BPz/view?usp=drive_link)下载qformer.onnx放置于./examples/MiniGPT-4/qformer.onnx，这是qformer onnx文件
6. （仓库下没有cpp/build，仓库下没有上述weight，parameter空文件夹）编译tensorrt-llm，使用./scripts/build_wheel.py，编译tensorrt-llm，把./tensorrt_llm（非编译输出）替换系统安装环境的tensorrt_llm，例如，容器安装环境为/usr/local/libs/python3.8/dist-packages/tensorrt_llm。
注意：编译输出的文件只需替换 (libnvinfer_plugin_tensorrt_llm.so) 到/usr/local/libs/python3.8/dist-packages/tensorrt_llm/libs文件夹，不要用编译输出的py文件去替换系统安装环境的tensorrt_llm
7. 执行./examples/MiniGPT4/preprocess.sh，（由于容器中的tensorrt9.0.0存在bug，有很大机率vit engine生成的是错误的，我们在测试机中没有启用编译vit engine的代码，只编译vicuna-7b，使用之前编译好的vit engine），编译qformer，vicuna-7b （若测试环境为tensorrt9.0.0，不保证不会出现bug，尽量使用已经编译的engine测试，结果是正确的；为了减少出现bug的几率，我们把vit和qformer放在两个engine，实际上是可以放同一个的）。
执行preprocess.sh会同时输出torch和tensorrt结果并比较。

## 主要开发工作

### 开发工作的难点

1. 不熟悉任何llm模型，对于其中用到的一些技术需要时间了解，例如rope，kv-cache，smooth quant等等
2. trt-llm专为llm设计，输入为[batch，seqlen]的嵌入词表int32类型，我们选题的minigpt4合并了图像特征输入，需要vicuna接受三维float16 [batch, seqlen, feature]作为输入，即需要修改处理输入的generation.py，难度较大
3. 实现blip2(vit+q-former)过程中存在不确定性，同时要考虑支持trt-llm对于大模型的一些加速feature，需要实现过程中同时支持这些feature
4. 尝试实现rptq, rptq包含gptq和reindex等统计模块，官方代码仅支持opt模型，需要读懂完整代码并添加llama的实现。简单来说，rptq统计layernorm每个输出通道的最大最小值，进行分组重排后量化，weight部分使用gptq量化，预计实现w4a4的量化精度（较多bug）
5. 整合整个系统，寻找测试数据集对比优化结果
6. 尝试实现对单张图片的多轮对话

### 开发与优化流程

1. 熟悉minigpt4模型和trt-llm
    1. 下载llama 7b模型权重，pretrained minigpt4模型权重，eva_vit模型权重，vicuna-7b-delta模型权重，blip2模型权重，合并llama7b与vicuna-7b-delta权重来得到vicuna模型权重，把vicuna，blip2，minigpt4，eva_vit四个模型权重放到home下某个目录，并修改./minigpt4/configs/models/minigpt4.yaml内的模型路径。同时修改./eval_configs/minigpt4_eval.yaml中的模型路径，设定low_resource flag为false，即可执行torch版的minigpt4模型
    遇到的bug：报错无法初始化tokenizer，解决方法，找不到pretrained minigpt4/模型路径有误，下载模型/把模型搬到另一个文件夹之后修改./eval_configs/minigpt4_eval.yaml中的路径，重新执行demo.py

        ![Untitled](https://github.com/yuanjiechen/trt_final/assets/43607076/502fe2c8-4028-478d-876a-a063d6d1fe8d)
        
        执行minigpt4 torch demo 模型
       
        ![Untitled 1](https://github.com/yuanjiechen/trt_final/assets/43607076/0c760ccf-bb0f-4a8d-b051-69d3404f7df0)
        
        查看输出发现，有一部分参数未被加载，再看代码，这部分参数在初始化模型时会被计算，用于位置编码，加载与否不影响结果。注意输入blip2的提示词，这部分提示词在每次执行时是随机取的，需要固定一个以获得固定的模型输出。启动后上传图片即可对话。
        
    3. 分析minigpt4模型
    观察minigpt4由三块组成，vit，q-former，vicuna，其中vicuna部分和trt-llm已经实现的llama部分可能可以共用，尝试使用example/llama/build.py加载vicuna参数并执行，得到正常的输出结果。vit和q-former用于处理图片输入，在seqlen维度上和提问的文本特征进行拼接后输入vicuna模型
    
    4. 熟悉trt-llm
    trt-llm包装了trt原有的底层api，采用类似torch的图结构，实际上仍然为静态图，模型定义与torch方法类似，模型forward实际为network定义过程，输入只需知道tensor形状，每过一个Module即把这个Module加入trt.INetworkDefinition中，并且可以在return之前把最后计算结果mark_output作为模型输出，模型定义完的weight需要手动导入。build与推理过程与trt一致，额外需要分配大量缓存空间优化llm。在engine执行阶段，仍然需要手动指定每一个输入的形状和buffer，在minigpt4包含的llama-7b模型中，tensorrt-llm支持了kv-cache，gpt-attention等优化手段。
2. 修改generation.py，llama-model.py，使输入支持图像特征，尽可能保持原有函数不变
思路：要使输入支持[batch，seqlen，feature]，首先要修改engine输入定义部分，dtype改成gp16，形状修改为3维。同时，输入已经不需要再查词表获取特征，所以注释掉vocab_embedding步骤。
观察generation，decode过程由多个step组成，每个step的输出[batch, max_len, feature]会经过faster-transformer.DynamicDecodeOp解码，得到[batch, max_len]的int32 tensor，由于engine已经改成支持三维输入，为了避免修改复杂的faster-transofrmer代码，提取vocab_embedding权重，对输出执行一次F.embedding操作得到[batch, max_len, feature]输入，同时为了保持显存地址不变，赋值输入到同一个tensor。
注意：调换了下一个step的输入准备工作和DynamicDecodeOp的解码过程，使输入准备之前可以得到转成三维的正确输入。
为了准备输入形状相关buffer显存分配，仍然需给一个dummy input_ids二维输入，注意，其中的id不能为stop id和pad id，需要限制randint范围，否则会出错。
由于训练vicuna时使用的stop id 为###，需要把生成的stop id改为###对应的835，但是并不work，检查中间输出的output_ids发现###被识别为2277号token，修改stop id为2277问题解决
以上，达成修改trt-llm推理部分代码以支持图片-文本混合特征输入，同时保留原本接口可用的目标。
尝试过的失败方法：把三维输入重新压缩成二维，取argmax，不成立，丢失太多信息
3. 实现vit，目标：使用trt-llm的接口实现vit
    1. 在model.py实现模型结构定义，block，attention，……等结构。dropout相关结构无需实现，在推理过程中drop_rate=0,。某些接口实现不同于torch，例如无法对tensor使用index slice，只能使用slice，view等接口访问对应位置；parameter类型不同于tensor，tensor在实现架构过程中不会保存值；无法对parameter进行concat，需要先提取其value。在学习以上层接口使用方法之后，按照vit结构实现整个模型
    2. 在weight.py导入模型weight，模仿已有的例子导入每个weight的numpy格式数据
    3. 在build.py实现参数导入和build过程
    4. 测试实现完的vit：build engine之后接入minigpt4替代pytorch模型，发现输出文本与使用torch vit不一致，但是文本仍然与图片相关且通顺。对比torch与trt模型输出，norm值差距接近600(后来知道正常engine norm值差距在50-90左右)。在排除模型结构，权重加载等模块出错之后，重新build几次模型，发现有大概20%几率能生成norm值较小且文本输出与torch一致的模型(过程中不修改代码，只重复进行build)。如果选择fp32 engine生成，结果正常，选择fp16 engine生成，大概率会生成错误的engine，已提交该bug（该bug已在未来发布的tensorrt9.0.1中修复，但是仍然可以作为bug上报；本地不打算更新tensorrt版本，预防出现更多bug，暂时使用可以执行的vit engine），先使用正常输出的engine继续优化。
4. 没有时间实现q-former了，使用onnx-trtexec这个流程跑通，同时实现一份固定形状的engine推理文件

5. 实现rptq
在官方的rptq实现中，并没有包含llama的实现，于是我们需要完整读懂代码并实现一份llama-rptq；原理：通过对input重排序实现分组量化，根据input重排序index重排weight，再对weight使用gptq进行量化实现低精度计算
整体思路：先在rptq实现llama获取量化参数，加载到trt-llm的kernel进行实际量化

    ![cover](https://github.com/yuanjiechen/trt_final/assets/43607076/ce5ea5fb-112b-4831-bce9-74d0e421fe02)
    
    上图为rptq作者提供的流程图，例子为opt模型，为了保留gptattention plugin的完整，我们只打算实现R1, R4, R5；其中R1，R4对输出进行重排序，为了保证后续输出不变需要对受到影响的weight进行行重排序，R5对输入进行重排序，需要对受到影响的weight行或列进行重排序。
    
    1. 需要依次实现如下文件 (位于./examples/MiniGPT-4/rptq-master文件夹下)
    llama → 主要模型wrap文件，定义基本模型参数，停止符号，词长度等等
    int_llama_layer → 替换原本decoder layer的模块为量化模块，例如QuantLinear，ReorderRMSNorm等
    llama_reorder_quantize → 核心rptq量化文件，不同的模型几乎无法通用，需要根据llama decoder的结构及rptq的方法重新思考及实现
    main → 接口调度文件，控制量化参数
    quant_transformer_layer → 控制每层量化具体执行过程，不同于opt需要重新实现
    trt-llm weight → 手动导入量化weight，zero point，scale到trt plugin 
    2. 量化数据使用minigpt4第二阶段训练数据，通过vit以及qformer产生编码了图片特征及文本特征的三维[batch，seqlen，feature]向量，打包成npy再加载
    3. 量化完保存的weight是fake量化的结果，可以使用保存下来的scale和zero point回到int8/int4
    4. 使用rptq的gptq量化的过程中，发现量化前后self_attn.proj层(mlp的最后一层)误差非常大，经排查，发现作者默认的数据量只有128，且seq_len过大，使得大量pad 0，调整这两个参数后gptq量化误差恢复正常
    5. 在trt-llm部分，需要实现量化参数的导入及plugin编写两步骤，前者导入每一层的scale，zeropoint等值到对应的模块，由于gptq保存的结果是fake量化的结果，需要在导入时使用scale，zeropoint转为int8之后导入trt plugin；后者需要编写两个plugin，w8a8矩阵乘法plugin以及reindex RMSnorm plugin
    6. 核心难点：非对称量化w8a8（cutlass没有包装好的w4a4给trt-llm使用，查看了int8_gemm_template.h，实现过程非常复杂，且cutlass文档几乎没有……）
    现有的kernel：参考smoothquantplugin，现有gemm int8只支持四个输入，input，weight，scale input，scale weight，并且weight需要fp32类型而数据为int8进行传入，非对称量化需要传入zero point，在算完int8 gemm之后进行零点误差补偿，再反量化为fp16，现有kernel接口无法传入zero point，不满足要求
    非对称量化所需kernel：根据下图所示情况，由于两个zero point计算过于复杂，于是我们只考虑对input进行非对称量化，也就是z2=0，DQ部分的零点误差补偿所需传入的结果是一个长度为N且值固定(与input无关)的int32_t向量，刚好可以作为bias传入

        ![屏幕截图 2023-09-18 201441](https://github.com/yuanjiechen/trt_final/assets/43607076/3f977cdb-787c-4a39-b041-f20245581cff)
       
        ![屏幕截图 2023-09-18 201845](https://github.com/yuanjiechen/trt_final/assets/43607076/89203b48-de92-4786-9022-9299c13743ed)
        
    8. 改造思路：如上图，预先确认cutlass接口支持传入bias，但是trt-llm实现时这部分参数给了null。首先逐层改造int8_gemm_template.h下的接口，传入一个int32_t类型的指针作为bias；但是，由于接口限制，bias需要与output dtype相同，bias又无法传入fp16的值，于是放弃尝试，仍然执行w8a8对称量化
    改造结果：失败，存在cuda runtime error （后来知道这个报错与增加bias无关，后面会解释；但是当时以为两者存在关联且不清楚内部计算顺序：完成矩阵乘法后先减去bias或者先做dq，如果按照bias与output均需要fp16来看应该是先做dq，这样的话即使传入bias结果还是错的，我们需要先加bias再dq，遂放弃）。虽然改造失败，但是思路应该是对的，可以对input进行非对称量化，考虑到input数值范围差距较大，可以提升量化性能；经过测试，即使是对称量化，性能也不会下降太多，在本次比赛将采用对称量化。
    input非对称量化原理图如下，DQ部分还有一步乘上input和weight scale的步骤未写

        ![IMG_0005](https://github.com/yuanjiechen/trt_final/assets/43607076/3081addc-3015-42e2-9906-ed6169c9aabe)
        
    10. 实现w8a8对称量化plugin
        1. 观察现有plugin：现有smoothquantgemm plugin，支持int8 gemm，接口输入为int8 activition，fp32/fp16 weight，float scale for act.，float scale for weight，粗看没啥问题，完美兼容，只需解决把int8 weight压缩到fp16 dtype下即可
        2. 为了保证输入的weight是fp16，使用一行代码对int8 weight进行压缩：
        int8_arr = np.ascontiguousarray(fp16_in.astype(np.int8).view(’<f2’))
        3. 第一个bug：问题并没有这么简单，Failed to initialize cutlass int8 gemm. Error: Internal error出现，本以为是给的参数不对，反复检查后发现参数无误，并且在plugin enqueue函数下无法cout任何输出，这点比较奇怪；继续排查cutlass文件，发现在gemm_universal_base.h下的initialize报错，分配共享显存的时候出错，cudaInvilidValue，怀疑是分配的数量过大导致报错，但是当时也没有更好的方法，于是上报给导师的同时和其他组(TIntIn)同学进行交流，他们也以为是自己数据传入问题，一交流才发现是bug。在后续其他组同学给出的提示下，发现确实由于显存分配过多导致报错，且这部分发生在选择gemm策略阶段，只要能选择出一个策略即可执行，这也解决了我之前的疑惑，enqueue无法输出任何信息，还没有执行到那个阶段，于是对int8_gemm_template.h进行修改，在选择策略的函数下添加try catch，防止选择到不合适策略时直接报错，问题解决。
           
            ![Screenshot_from_2023-09-19_09-37-07](https://github.com/yuanjiechen/trt_final/assets/43607076/bdf0b74a-0193-430b-80e7-8c4350678617)
            
        5. 第二个bug：plugin可以正常编译且执行了，但是输出的文本不对，于是检查输入和python代码，发现rptq这个方法实际上无法使用int8 gemm进行加速。
        观察下图，左边为input和对应scale_a，右边为weight和对应scale_b，由于rptq需要进行列交换（参考红色箭头），再对input列进行分组量化，scale_a为列方向，然而，进行完int8矩阵乘法之后，输出矩阵大小为Seqlen*Out，进行反量化时，Out方向的scale_b可以进行反量化，但是Feature维度已经在矩阵乘法时消去，scale_a为Feature方向长度，无法进行反量化，在作者原代码实现时使用fake量化，没有这个问题，目前看来该问题无法解决，只能使用fake量化模拟rptq的结果。
            ![bug2](https://github.com/yuanjiechen/trt_final/assets/43607076/5b2c0428-55f8-422c-abc4-bb011284b006)

            
        6. 第三个bug：由于feature方向量化无法进行输出反量化操作，后续我们都替换为seqlen方向量化进行优化。
        int8量化下输出结果错误，我们尝试使用smoothquant plugin进行int8 gemm。在采用seqlen维度量化规避d.中的问题之后，输出结果错误。首先排查输入数据，input使用cast转到int8，weight通过numpy.view把两个int8数据压缩进一个fp16可表达的16bit空间，同时shape减半，在打印测试验证压缩过程正确之后，进行fake测试结果错误。再进行fake量化排查值的范围，发现在linear中对input的scale乘100之后再进行fake量化时输出正常，于是验证linear中是否出现数值溢出，并对input做int8的fake量化，紧接着对fake量化输出进行反量化，输出转为fp32格式表达（如下图），经过测试输出正确，从而确定了linear中出现了数值溢出问题。进一步，我们使用smoothquant ****plugin替换matmul，使用int8精度的输入和int8精度的weight，两者的float scale输入到smoothquant plugin，结果输出错误，我们排查并尝试了多种input与weight的dtype，并尝试在plugin输入时scale全给1，在plugin输出后乘上两个方向的scale（保证scale的乘法工作在fp32下），还是不work，确认是smoothquant plugin输出问题，选择尝试其他plugin。
        下一步我们修改gemmplugin的cublas，发现他使用的是cublaslt的gemm接口，根据文档要求把int8所需的dtype，compute type设定完成后，此时查文档得知输出为int32，然后在python处使用cast把输出转为float32，结果和smoothquant plugin一致，暂时排除plugin引起问题的情况。
        后续我们也尝试了把weight/scale之后做clip，限制到fp16可表示的范围内，但是还是没用，输出乱码，至此，各种dtype，压缩int8值到fp16/fp32，两种gemm plugin，cast，round等等方法的排列组合均尝试过，输出还是乱码，不排除是tensorrt的bug，但是时间来不及了，只能跑一个可以输出的版本提交。

            ![Screenshot_from_2023-09-21_18-50-01](https://github.com/yuanjiechen/trt_final/assets/43607076/079c843f-9516-4ed3-bcb0-6271af7cda17)

            ![Screenshot_from_2023-09-21_18-51-55](https://github.com/yuanjiechen/trt_final/assets/43607076/eaf579df-785a-42eb-b79b-06ec2911cc64)

            
        8. 实现rptq中reorder rmsnorm quant模块，并实现对应的cuda算子。该算子主要实现rmsnorm基于rptq中权重分块的重排序过程，并与量化参数scale相结合，使得输出为量化后的int8数值。最终后续线性运算模块将rmsnorm的量化参数scale融合到weights中。
    

### 优化效果

由于fake量化输出时间会变长，相比torch更长，没有比较该结果。

启用了gemm plugin，gpt attention plugin，完成vit，q-former，vicuna-7b模型的转化，40个测试数据，torch 145 秒，trt 115 秒

### 送分题答案

1. 从huggingface下载gpt2-median模型，使用hf_gpt_convert.py提取权重参数，再使用build.py生成engine，最后使用run.py —max_otuput_len=8得到输出如下：
    
    ###以下为输出###
    Input: Born in north-east France, Soyer trained as a
    Output:  chef before moving to London in the early
    ###以上为输出###
    
2. 使用上一题build的engine执行summarize.py，无法自动下载，手动下载数据集放到.cache/huggingface文件夹之后再执行，得到以下输出
###以下为输出###
    
    [09/12/2023-12:27:39] [TRT-LLM] [I]  Input : ['(CNN)James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV\'s "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he\'d been a busy actor for decades in theater and in Hollywood, Best didn\'t become famous until 1979, when "The Dukes of Hazzard\'s" cornpone charms began beaming into millions of American homes almost every Friday night. For seven seasons, Best\'s Rosco P. Coltrane chased the moonshine-running Duke boys back and forth across the back roads of fictitious Hazzard County, Georgia, although his "hot pursuit" usually ended with him crashing his patrol car. Although Rosco was slow-witted and corrupt, Best gave him a childlike enthusiasm that got laughs and made him endearing. His character became known for his distinctive "kew-kew-kew" chuckle and for goofy catchphrases such as "cuff \'em and stuff \'em!" upon making an arrest. Among the most popular shows on TV in the early \'80s, "The Dukes of Hazzard" ran until 1985 and spawned TV movies, an animated series and video games. Several of Best\'s "Hazzard" co-stars paid tribute to the late actor on social media. "I laughed and learned more from Jimmie in one hour than from anyone else in a whole year," co-star John Schneider, who played Bo Duke, said on Twitter. "Give Uncle Jesse my love when you see him dear friend." "Jimmy Best was the most constantly creative person I have ever known," said Ben Jones, who played mechanic Cooter on the show, in a Facebook post. "Every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life\'s many passions." Born Jewel Guy on July 26, 1926, in Powderly, Kentucky, Best was orphaned at 3 and adopted by Armen and Essa Best, who renamed him James and raised him in rural Indiana. Best served in the Army during World War II before launching his acting career. In the 1950s and 1960s, he accumulated scores of credits, playing a range of colorful supporting characters in such TV shows as "The Twilight Zone," "Bonanza," "The Andy Griffith Show" and "Gunsmoke." He later appeared in a handful of Burt Reynolds\' movies, including "Hooper" and "The End." But Best will always be best known for his "Hazzard" role, which lives on in reruns. "Jimmie was my teacher, mentor, close friend and collaborator for 26 years," Latshaw said. "I directed two of his feature films, including the recent \'Return of the Killer Shrews,\' a sequel he co-wrote and was quite proud of as he had made the first one more than 50 years earlier." People we\'ve lost in 2015 . CNN\'s Stella Chan contributed to this story.']
    [09/12/2023-12:27:39] [TRT-LLM] [I]
    Reference : ['James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88 .\n"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV .']
    [09/12/2023-12:27:39] [TRT-LLM] [I]
    Output : [[' James Best died of pneumonia.']]
    [09/12/2023-12:27:39] [TRT-LLM] [I] ---------------------------------------------------------
    Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
    [09/12/2023-12:27:40] [TRT-LLM] [I] ---------------------------------------------------------
    [09/12/2023-12:27:40] [TRT-LLM] [I] HF Generated :
    [09/12/2023-12:27:40] [TRT-LLM] [I]  Input : ['(CNN)James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV\'s "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he\'d been a busy actor for decades in theater and in Hollywood, Best didn\'t become famous until 1979, when "The Dukes of Hazzard\'s" cornpone charms began beaming into millions of American homes almost every Friday night. For seven seasons, Best\'s Rosco P. Coltrane chased the moonshine-running Duke boys back and forth across the back roads of fictitious Hazzard County, Georgia, although his "hot pursuit" usually ended with him crashing his patrol car. Although Rosco was slow-witted and corrupt, Best gave him a childlike enthusiasm that got laughs and made him endearing. His character became known for his distinctive "kew-kew-kew" chuckle and for goofy catchphrases such as "cuff \'em and stuff \'em!" upon making an arrest. Among the most popular shows on TV in the early \'80s, "The Dukes of Hazzard" ran until 1985 and spawned TV movies, an animated series and video games. Several of Best\'s "Hazzard" co-stars paid tribute to the late actor on social media. "I laughed and learned more from Jimmie in one hour than from anyone else in a whole year," co-star John Schneider, who played Bo Duke, said on Twitter. "Give Uncle Jesse my love when you see him dear friend." "Jimmy Best was the most constantly creative person I have ever known," said Ben Jones, who played mechanic Cooter on the show, in a Facebook post. "Every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life\'s many passions." Born Jewel Guy on July 26, 1926, in Powderly, Kentucky, Best was orphaned at 3 and adopted by Armen and Essa Best, who renamed him James and raised him in rural Indiana. Best served in the Army during World War II before launching his acting career. In the 1950s and 1960s, he accumulated scores of credits, playing a range of colorful supporting characters in such TV shows as "The Twilight Zone," "Bonanza," "The Andy Griffith Show" and "Gunsmoke." He later appeared in a handful of Burt Reynolds\' movies, including "Hooper" and "The End." But Best will always be best known for his "Hazzard" role, which lives on in reruns. "Jimmie was my teacher, mentor, close friend and collaborator for 26 years," Latshaw said. "I directed two of his feature films, including the recent \'Return of the Killer Shrews,\' a sequel he co-wrote and was quite proud of as he had made the first one more than 50 years earlier." People we\'ve lost in 2015 . CNN\'s Stella Chan contributed to this story.']
    [09/12/2023-12:27:40] [TRT-LLM] [I]
    Reference : ['James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88 .\n"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV .']
    [09/12/2023-12:27:40] [TRT-LLM] [I]
    Output : [[' James Best died of pneumonia.']]
    [09/12/2023-12:27:40] [TRT-LLM] [I] ---------------------------------------------------------
    [09/12/2023-12:44:44] [TRT-LLM] [I] TensorRT-LLM (total latency: 3.583282709121704 sec)
    [09/12/2023-12:44:44] [TRT-LLM] [I] TensorRT-LLM beam 0 result
    [09/12/2023-12:44:44] [TRT-LLM] [I]   rouge1 : 0.21458993951991212
    [09/12/2023-12:44:44] [TRT-LLM] [I]   rouge2 : 0.061022639415477836
    [09/12/2023-12:44:44] [TRT-LLM] [I]   rougeL : 0.1688631667234835
    [09/12/2023-12:44:44] [TRT-LLM] [I]   rougeLsum : 0.18803534299200658
    [09/12/2023-12:44:44] [TRT-LLM] [I] Hugging Face (total latency: 9.523822784423828 sec)
    [09/12/2023-12:44:44] [TRT-LLM] [I] HF beam 0 result
    [09/12/2023-12:44:44] [TRT-LLM] [I]   rouge1 : 0.22089149352609289
    [09/12/2023-12:44:44] [TRT-LLM] [I]   rouge2 : 0.06127009262128831
    [09/12/2023-12:44:44] [TRT-LLM] [I]   rougeL : 0.16982143879321
    [09/12/2023-12:44:44] [TRT-LLM] [I]   rougeLsum : 0.19046700771609248
    ###以上为输出###
    
    看起来trt输出与torch一致，没有区别，执行一次需要半小时，不知道为什么，并且由于存在bug，修改了几行代码内容以执行，以下是bug截图
   
    ![Untitled 2](https://github.com/yuanjiechen/trt_final/assets/43607076/29666b1d-db98-4da3-9c67-a3e9fbf00941)
   
    ![Untitled](https://github.com/yuanjiechen/trt_final/assets/43607076/39af88f5-d5bd-4da6-83e4-849ee2ccb361)
    

### Bugs

[https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues/91](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues/91) 在8.c修好了

[https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues/87](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues/87)

### 想说的话
bug实在是太多了，部分warning，error安排不合理，比如unused weight会报错failed to set weight xxx，其实这个weight 已经set，只不过没有在forward使用；如果support format不支持，是否可以指出是哪一个参数不支持，省去打印每一个参数dtype的时间
转trt模型20%成功率的bug实在是磨人！！！
cutlass可不可以加个类似cublas，cudnn一样好读的文档！！！
