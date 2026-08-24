# 最终提交演示

默认启动时，这个界面只读取仓库内的冻结证据、模型身份和演示图片，不会把不同实验拼成新指标。只有显式加载服务端授权的设备profile，且用户在一键部署页二次确认后，控制器才会执行真实远程安装。

## 启动

```bash
python3 scenes/freeway_traffic/video_demo/app.py --check
python3 scenes/freeway_traffic/video_demo/app.py --host 127.0.0.1 --port 8088
```

浏览器打开 `http://127.0.0.1:8088`。默认服务地址只用于只读健康检查，可以在页面里临时修改。

## 页面

- 系统演示：交通四区域与工业双模态的四阶段业务流程。
- 一键部署：服务端配置真实设备档案后才可执行；未配置时保持只读。
- 运行监控：读取云边服务健康、Outbox、最终业务结果和冲突计数。
- 四类评分页：展示当前冻结结果及其适用边界。

## 一键部署拓扑编排

部署页支持两类节点，二者的权限边界不同：

- 服务端授权节点来自所选 `profile.nodes`。页面只保存 `node_id`，显示角色、服务地址和 `credential-ready` 状态，不接收或展示SSH密钥。
- 手动节点由用户填写HTTP服务根地址。它可以参加只读连通探测、绑定关系预览和拓扑JSON导出，不能进入SSH安装或真实部署。

拓扑至少包含1个cloud和1个edge。用户可以指定主cloud，并为每个edge单独选择上游cloud。草案保存在浏览器 `localStorage`，刷新页面后可以继续编辑。切换服务端档案时，授权成员按该档案的 `default_topology` 重建，手动节点继续保留。

“检测全部节点”调用多节点只读接口：

```json
{
  "nodes": [
    {
      "id": "manual_edge_1",
      "role": "edge",
      "label": "手动边缘 1",
      "endpoint": "http://127.0.0.1:18101"
    }
  ],
  "timeout_seconds": 3
}
```

请求地址为 `POST /api/connections/test-topology`。catalog会向页面公开授权节点的健康检查根地址，不公开SSH用户、私钥路径或运行目录；真实部署时仍只由服务端根据 `node_id` 解析管理凭据。

只有全部成员都来自同一个服务端授权档案、全部凭据就绪、场景已开放且拓扑完整时，“一键部署并验证”才启用。创建会话的关键请求体为：

```json
{
  "scenario": "traffic",
  "template": "recommended",
  "profile_id": "lab",
  "topology": {
    "primary_cloud_id": "cloud_a",
    "cloud_node_ids": ["cloud_a", "cloud_b"],
    "edge_bindings": [
      {"edge_id": "edge_1", "cloud_id": "cloud_a"},
      {"edge_id": "edge_2", "cloud_id": "cloud_b"}
    ]
  }
}
```

部署DAG按“冻结拓扑、逐节点预检、cloud启动、按绑定依次启动edge、全拓扑验证”执行。预检只检查系统与运行资产，失败时没有远端变更。远端安装或验证失败后，系统停止后续调度，只停止本session成功安装并仍活动的节点，并按edge到cloud的逆序执行补偿。页面能够读取后端返回的逐节点stage状态；部分节点成功不会显示为部署完成。

### 配置授权节点池

手动endpoint不能在浏览器中升级为授权节点。管理员需要在服务端创建v2 profile，并用 `--profiles` 启动演示服务。下面是最小的一云一边示例：

```json
{
  "schema_version": "guided-deployment-profiles/v2",
  "profiles": [
    {
      "id": "lab",
      "label": "实验室节点池",
      "enabled": true,
      "allow_deploy": true,
      "nodes": [
        {
          "id": "cloud_a",
          "role": "cloud",
          "label": "云端A",
          "service_url": "http://192.168.1.10:18100",
          "ssh": {
            "host": "192.168.1.10",
            "user": "deployer",
            "ssh_port": 22,
            "identity_file": "/etc/cloud-edge/keys/id_ed25519"
          },
          "sdk_root": "/opt/cloud-edge-scene-sdk"
        },
        {
          "id": "edge_1",
          "role": "edge",
          "label": "边缘1",
          "service_url": "http://192.168.1.11:18101",
          "ssh": {
            "host": "192.168.1.11",
            "user": "deployer",
            "ssh_port": 22,
            "identity_file": "/etc/cloud-edge/keys/id_ed25519"
          },
          "sdk_root": "/opt/cloud-edge-scene-sdk",
          "llama_binary": "/opt/cloud-edge-scene-sdk/runtime/bin/llama-server"
        }
      ],
      "default_topology": {
        "primary_cloud_id": "cloud_a",
        "cloud_node_ids": ["cloud_a"],
        "edge_bindings": [
          {"edge_id": "edge_1", "cloud_id": "cloud_a"}
        ]
      }
    }
  ]
}
```

```bash
python3 scenes/freeway_traffic/video_demo/app.py \
  --profiles /absolute/path/deployment-profiles.json \
  --host 127.0.0.1 --port 8088
```

仓库还提供了2云、3边的禁用态模板 `deployment_profiles.example.json`。先复制到仓库外，替换保留测试地址、路径和用户，在所有节点手工预检合格后再将 `allow_deploy` 改为 `true`。

`identity_file`、`sdk_root`和边缘节点的`llama_binary`必须是服务端可用的绝对路径。页面目录接口只返回节点ID、角色、标签、服务地址和凭据是否就绪，不返回SSH用户名、密钥路径或运行目录。

安全运行边界：部署控制台当前没有独立的账号系统、mTLS节点身份或跨进程部署锁，应保持绑定 `127.0.0.1`，远程管理通过SSH隧道进入，不应直接暴露到公网。

## 三项创新

### 1. 模型绑定的紧凑语义编码与单token推理

- 同一0.8B Q4模型、同240个交通事件与目标上，非原生长语义输入均值167.296 token，raw16任务ABI固定为16 token；两臂都只生成1 token。
- 服务端模型TTFT从359.997 ms降至35.644 ms，降低90.099%，240/240配对事件均更快。
- 这是模型、输入ABI与训练分布匹配的联合效果，不解释为纯token数量因果。两臂动作一致32/240，不能写成业务等价实验。
- 同卡次级证据：树模型绑定阈值区间IR相对121个有效特征的active-float32，将逐事件载荷从484 B降至233.6975 B；预测1600/1600一致，50棵树路径80000/80000一致。

### 2. 面向关联任务冲突感知的摘要先行渐进证据调度方法

- 100窗、400事件首次全部发送summary；380个事件因风险、不确定性或分歧按需回拉feature，20个事件仅靠summary即可完成，普通事件主动raw为0。
- 相对四节点完整主分区raw预送，application JSON从11,328,303 B降至8,896,732 B，降低21.464565%；完整raw只作通信主基线。
- 最终业务结果、risk level、route decision、cloud final result、冲突处理结果五类视图相对冻结正式路径（11,113,184 B）都达到400/400一致。
- 三类受控冲突只回拉目标道路集两个owner的exact 2份raw并重算；TTL过期、令牌篡改、单侧失败均不授予全局确认，补证失败动作授权为0。

### 3. 安全门控的双时标响应与可修订终态框架

- 同一冻结100窗、400事件的配对端点消融中，阻塞到权威终态的均值561.247 ms，安全门控业务响应均值159.703 ms；需云确认动作提前授权为0。
- 该数字来自同一次正式运行的反事实端点配对，并非第二套独立live A/B。
- 100组×4成员的受控时钟三臂实验中，C臂首次结果100/100、可完成组权威final恢复75/75、截止后迟到恢复25/25、错误final为0、不完整证据危险授权为0。

## 工业公开效果

- RGB Image / Pixel AUROC：90.27% / 98.08%。
- 红外 Image / Pixel AUROC：93.03% / 95.83%。
- 受控多目标阈值演示把硬动作冲突从73/443降至18/443，状态固定为 `DEVELOPMENT_DEMO_ONLY`。同一443条全量标签用于选参与评估，候选只安装到隔离开发注册表，生产端点与正式阈值未修改。
- 演示工件没有冻结的逐样本置信度证据，`cloud_final.confidence`保持为`null`。

## 含感知E2E与专用能力

- 双场景900条含感知E2E加权均值101.497 ms，831/900（92.3333%）不超过0.2 s，平均门裁决为 `PASS`。
- 交通提交计分路径采用全Qwen primary：400个单节点事件请求均值/P95为159.703/232.502 ms，331/400不超过0.2 s；400/400调用Q4，242条建议采纳，158条由规则硬授权回退。四节点共享单slot的max窗口均值227.446 ms只作并发诊断，`production_changed=false`。
- 工业500条组件组合含感知E2E均值/P95为54.933/62.046 ms；双模态并行工件均值/P95为56.394/64.539 ms。
- 正式交通专用能力使用同1600条normal/weak事件比较0.8B Q4与traffic ExtraTrees。ExtraTrees的Accuracy、Macro-F1、Weighted-F1为70.75%、56.1555%、69.6609%，Q4为64.75%、53.8884%、66.3794%，对应保持率91.5194%、95.9627%、95.2893%，三门均PASS。
- 两臂固定同event_id/target；ExtraTrees读取226维float32特征，Q4读取部署原生raw16，输入字节不同。9B仅用于教师标签和低频异步审计机制，不参与该能力分母；本项`production_changed=false`。

## 自包含证据

应用读取这些仓内目录：

- `results/final_submission_v1`
- `results/final_raw16_codec_latency_ablation_v1`
- `results/final_tree_threshold_semantic_ir_ablation_v1`
- `results/final_traffic_summary_first_on_demand_v1`
- `results/final_traffic_overlap_roadset_selective_pull_v2`
- `results/final_framework_provisional_endpoint_pairing_v1`
- `results/final_framework_revisioned_finality_three_arm_v1`
- `results/industrial_controlled_threshold_optimization_demo_v1`
- `model_bundle/final`
- `scenes/freeway_traffic/video_demo/assets`

运行 `app.py --check` 会验证15项矩阵、全部矩阵证据路径、三项创新的关键数值、本地交通主链快照与最终边缘GGUF SHA。
