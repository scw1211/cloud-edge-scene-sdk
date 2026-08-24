# 最终提交包就绪检查

本仓库按自包含提交口径整理。最终评分、报告、演示和验证器只读取仓库内的正式文件，不依赖工作站外部目录。

## 一键验证

从仓库根目录执行：

```bash
python3 results/industrial_global_optimization_delivery_v1/verify_delivery.py
python3 results/final_submission_v1/verify_final_submission.py
python3 scenes/freeway_traffic/video_demo/app.py --check
sha256sum -c model_bundle/final/qwen3_5_0p8b_q4/SHA256SUMS
```

全部命令必须退出0。验证器只读，不启动模型、不改阈值、不连接生产设备。

## 最终唯一口径

- 最终评分矩阵共15项：`PASS 15 / PARTIAL 0 / FAIL 0 / N/A 0`。
- 创新性统一为三项框架主张。第一项为模型绑定的紧凑语义编码与单token推理：raw16将平均输入167.296→16 tokens、服务端模型TTFT 359.997→35.644 ms，降低90.099%，240/240更快；树阈值IR跨异构实例payload降低51.715%且预测和路径严格一致。第二项为面向关联任务冲突感知的摘要先行渐进证据调度：400次首次summary、380次因高风险、不确定性或模型/规则分歧等闭集原因按需feature、20次summary足够、普通事件主动raw为0，应用JSON降低21.464565%，五类结果400/400等价；通信量以完整raw为基线，业务语义以冻结正式路径A（11,113,184 B）为参考；受控冲突只回拉exact 2份raw。第三项为安全门控双时标响应与可修订终态：安全响应相对阻塞权威终态均值561.247→159.703 ms，12个需云确认动作零提前授权；25/25截止后迟到组升级终态，错误final为0。
- 创新性边界固定：raw16是表示与训练合同匹配的联合效果，动作一致率32/240，不声称业务等价或纯token因果；树IR排除4,254,279 B共享工件，主机Python开销不代表Jetson端到端时延。summary-first按闭集条件升级，正式100窗自然道路集冲突为0；工业仅验证状态驱动热图上传，当前无云端反向pull，不称与交通同闭环。端点实验是同一次运行的反事实配对，185/400局部决策后来修订；三臂实验使用受控到达条件和受控时钟，不表示现实故障率或计算性能。
- 工业阈值优化、9B教师链、确定性调度、Outbox、鉴权、缓存和发布门继续留在业务效果、稳定性或工程边界中，不作为独立创新主张。
- 边缘模型：Qwen3.5-0.8B Q4_K_M，GGUF SHA为`828f839873c7505005101544d8febb7408bc0443c153c4ba97ec4b3603445526`。
- 最终配对基线：Qwen3.5-9B Q4_K_M，GGUF SHA为`dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c`。TTFT降低86.767%；基础毫秒级业务语义门中0.8B/9B的Math、Code、NLR保持率为96.55%、100%、90%，两个硬门均PASS。能力运行请求最大GPU offload，精确tensor placement未归档；本门不证明广义通用泛化。
- d4f模型只作为困难任务按需第二通用候选；不参与当前矩阵，既有实验未通过，当前未取得发布资格、正式部署或Orin双模型内存结论。
- 工业视觉层：十产品Orin实测的RGB/红外Image AUROC为90.27%/93.03%，只说明双模态异常分数的输入质量，不是阈值或云端策略的优化前后成绩。
- 工业一致性层：模型1.0.0 canonical汇总为103/443至少一路review、18/443 normal↔anomaly硬冲突、337/443三态完全一致，硬冲突率4.063205%通过5%门。该证据只保留汇总值，没有443对row ID；自然18组也没有逐件回放，不能写成18/18。
- 工业多目标阈值开发实验：同一443条全量标签、40维阈值、65,167个坐标候选和冻结resolver，优化期间不重训resolver。硬冲突73/443→18/443，Accuracy 91.20%→94.36%，Macro-F1 86.88%→91.46%，异常Recall 95.92%→98.54%，误放行14→5、误隔离25→20、review 248→148，综合代价下降45.3685%。状态为`DEVELOPMENT_DEMO_ONLY`，候选只进入隔离开发注册表，生产未改。
- 工业30/30与交通48/48受控挑战共78/78解决，因此按赛题口径一致性为`PASS`；该受控结果不能替代自然18组逐件回放。
- 双场景900条含感知业务E2E加权均值101.497 ms，831/900（92.3333%）不超过0.2 s，按赛题平均E2E门裁决为`PASS`。交通全Qwen primary的400个单节点事件请求均值/P95为159.703/232.502 ms，331/400不超过0.2 s；400/400调用Q4，242条建议采纳、158条由规则硬授权回退。四节点共享单slot的max窗口均值227.446 ms仅作并发诊断，不作为赛事门。选择性Q4的81.257/151.885 ms（事件/窗口均值）只作低时延消融。工业500条从prepared NCHW开始，均值/P95为54.933/62.046 ms；两场景不报总体P95。
- 正式专用云边能力仅评价交通：同一1600条normal与weak-network事件上，边0.8B Q4/云端交通ExtraTrees的Accuracy/Macro-F1/Weighted-F1保持率91.5194%/95.9627%/95.2893%，三门均PASS。ExtraTrees读取226维特征，Q4读取部署原生raw16；离线800条排除，9B只负责异步复核和教师标签。
- 正式交通策略为“全事件Qwen建议＋规则硬授权＋Student回退”。官方事件口径卡裁决`PASS_ALL_QWEN_PRIMARY_PER_NODE_E2E`；仓库默认复现配置使用primary，现网未现场切换。旧源包的window门裁决仅是被事件单位口径取代的历史诊断。
- 最终自动化回归：根框架与工业套件376/376，交通场景128/128，演示检查9/9，合计513/513通过、0跳过、0失败、0错误，traffic smoke通过；道路集、选择性补证据与summary-first专项83/83为定向重跑，已包含在完整套件中，不重复计数。完整回归使用traffic Python环境和scikit-learn 1.7.2，收据为`results/final_submission_v1/evidence/system/final_test_execution_summary.json`。
- 交通节点规模适应性：冻结100窗口的4→3节点对照两臂均100%完成，存活事件风险/动作/路由100%一致，窗口mean 40.784→20.000 ms、吞吐+52.945%；传感器覆盖保留75%，证据见`results/final_traffic_node_scale_ablation_v1/`。
- 赛题逐项写作入口：`results/final_submission_v1/contest_requirement_test_report.md`及对应JSON。

## 模型与工业资产

- `model_bundle/final/qwen3_5_0p8b_q4`：最终边缘GGUF、身份、SHA和适配器发布包。
- `model_bundle/final/qwen3_5_9b`：9B Q4最终身份、配置、tensor index、chat template、运行时与TTFT正式门结果；不包含大权重。
- `scenes/industrial_anomaly/perception/assets`：共享ONNX、Orin TensorRT engine，以及10产品×2模态PatchCore bank。
- 工业公开主结果资产：真实Orin十产品RGB/红外感知证据，以及多目标阈值开发实验的复算摘要和验证器。

## 演示检查

```bash
python3 scenes/freeway_traffic/video_demo/app.py --host 127.0.0.1 --port 8088
```

打开`http://127.0.0.1:8088`，依次检查：

1. 交通与工业两种演示都能切换四阶段；
2. 工业RGB和红外图片都能显示；
3. 最终评分矩阵显示15项与15/0/0/0状态；
4. 技术详情明确9B为低频异步辅助审计；
5. 未配置真实设备档案时，一键部署明确保持只读。

## 打包前人工检查

- 仓库中没有密钥、令牌、远端缓存、训练中间件或可重建benchmark目录；
- 最终报告、评分矩阵、演示和README的工业冲突计数一致；
- 保留legacy 18件ID清单未保存、无法逐件列名的限制；
- TTFT和基础毫秒级业务语义门明确PASS；能力运行只声明请求最大GPU offload且精确tensor placement未归档；
- 工业阈值开发实验明确标记`DEVELOPMENT_DEMO_ONLY`，没有写成生产收益；
- 创新性材料统一为三项框架创新，没有把工业阈值、教师链、调度或可靠性组件单列成创新；
- raw16、tree-IR、summary-first、端点配对和三臂终态实验的边界与主报告5.1一致；
- 没有把自然冲突解决率写成已验证；
- 压缩包名按“单位-姓名-作品名称-联系电话”填写。
