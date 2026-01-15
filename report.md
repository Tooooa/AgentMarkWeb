### **第四章 详细设计与工程实现 (Implementation Details)**

本章为项目最关键部分，覆盖浏览器端探针（插件）实现、前端可视化系统与算法工程化适配，重点呈现可运行的工程设计与核心代码片段。
@@
#### 4.1 浏览器端探针设计（Plugin Development）

##### 4.1.1 Manifest V3 架构适配与长连接

- **Manifest V3 架构适配与长连接**
  - **MV3 的关键变化**：MV3 将常驻后台页替换为 `background service worker`（SW）。SW 具备以下工程约束：
    - **事件驱动且可被回收**：空闲一段时间后 SW 会被浏览器终止，内存态（变量、连接、计时器）随之丢失。
    - **不能依赖“永远在线”**：在 SW 内维持长期阻塞（例如常驻 WebSocket、永不结束的 `setInterval`）在实际环境中不可靠；需要通过事件唤醒 + 断线重建来达成“长连接体验”。
    - **状态必须可恢复**：重要状态应落到 `chrome.storage`（或 `chrome.storage.session`）中，SW 重启后通过存储恢复。
  - **注入策略（chrome.scripting）**：探针需要尽早拦截页面 API（如 `fetch/XHR`），因此注入应尽量靠前。
    - 推荐使用 `chrome.scripting.registerContentScripts` 注册内容脚本，并设置 `runAt: "document_start"`，保证在页面脚本执行前尽早加载。
    - 当需要对特定 tab/特定 frame 进行“按需注入”时，可使用 `chrome.scripting.executeScript` 在运行时注入。
  - **“长连接”的落地方式（Port + 心跳 + 自动重连）**：
    - **内容脚本常驻**：内容脚本（content script）在页面生命周期内相对稳定，可承担“连接的锚点”。
    - **内容脚本到 SW**：通过 `chrome.runtime.connect` 建立 `Port` 通道；内容脚本以 20~30s 发送一次 `probe.ping`，形成 keepalive 信号。
    - **SW 被回收后的恢复**：SW 重启后，内容脚本会触发重连（重新 `connect`），SW 侧 `onConnect` 重新注册监听，即可恢复会话。
    - **工程含义**：这不是“SW 永不下线”，而是依靠心跳 + 快速重建，使上层逻辑表现为持续在线。
  - **典型 `manifest.json` 关键段落**：

```json
{
  "manifest_version": 3,
  "name": "AgentMark Probe",
  "version": "1.0.0",
  "permissions": ["scripting", "tabs", "storage"],
  "host_permissions": ["<all_urls>"],
  "background": { "service_worker": "background.js" },
  "action": { "default_title": "AgentMark" }
}
```

- **后台 SW 使用 `chrome.scripting` 注入与连接管理**
  - **注入时机选择**：
    - `onInstalled` + `registerContentScripts`：一次注册，全局生效，配合 `matches` 覆盖目标站点。
    - 若需要避免对所有站点注入，可将 `matches` 收窄，或改为在 `chrome.tabs.onUpdated`（页面完成加载的特定阶段）中按需 `executeScript`。
  - **连接管理要点**：
    - SW 内保存的 `ports` 仅是内存态缓存；SW 被回收后会丢失，因此必须允许内容脚本侧自动重连。
    - 建议在 `onMessage` 中显式区分 `probe.ping` 与业务事件：
      - `probe.ping`：用于 keepalive 与可观测性（例如记录最后活跃时间）。
      - `probe.event`：携带探针事件，做上报/缓存/聚合。

```javascript
// background.js
const CONTENT_ID = "agentmark-hook";

chrome.runtime.onInstalled.addListener(async () => {
  await chrome.scripting.registerContentScripts([
    {
      id: CONTENT_ID,
      js: ["content.js"],
      matches: ["<all_urls>"],
      runAt: "document_start"
    }
  ]);
});

// Port 长连接管理与后端转发
const ports = new Map();
chrome.runtime.onConnect.addListener((port) => {
  ports.set(port.sender.tab?.id || port.name, port);
  port.onMessage.addListener(async (msg) => {
    // keepalive：让 SW 在有事件时被唤醒，并记录活性
    if (msg.type === "probe.ping") return;
    // 透传/聚合到告警服务（示例）
    if (msg.type === "probe.event") {
      try {
        await fetch("http://localhost:7001/ingest", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(msg.payload)
        });
      } catch (e) {
        // 转发失败时可在本地缓存（省略）
      }
    }
  });
  port.onDisconnect.addListener(() => {
    ports.delete(port.sender.tab?.id || port.name);
  });
});
```

- **内容脚本侧的长连接策略（自动重连 + 心跳）**
  - **为什么心跳放在内容脚本**：内容脚本跟随 tab 生命周期存在，而 SW 可能被系统回收；将心跳放在内容脚本可以在 SW 重启时自然触发重建。
  - **重连建议**：真实工程里建议将 `Port` 建立封装成一个 `connectWithRetry()`，在 `onDisconnect` 时指数退避重连（避免 SW 不可用时疯狂重试）。下方示例保持为最小可运行实现。

##### 4.1.2 API 流量拦截技术：Hook fetch/XHR 并实时修改返回

- **API 流量拦截：Hook fetch / XHR 并可修改 JSON**
  - **目标**：在浏览器端对与 Agent 相关的 API 调用（如 `/chat`、`/completions` 等）进行“观测 + 改写”，将风险评分/拦截结果实时写回到 JSON 响应包中，从而在页面层面实现即时防护。
  - **隔离与注入**：MV3 内容脚本与页面 JS 处于不同执行世界（isolated world），直接在内容脚本里重写 `window.fetch/XMLHttpRequest` 不会影响页面自身调用。因此需要：
    - 将 Hook 逻辑以 `<script>` 注入到页面上下文执行。
    - 页面上下文产生的事件通过 `window.postMessage` 发回内容脚本。
    - 内容脚本再经 `chrome.runtime.connect` 的 `Port` 上报到 SW（或本地汇聚服务）。
  - **fetch 改写要点**：
    - 读取响应体前先 `res.clone()`，避免破坏原始 `Response` 的流。
    - 仅对 `content-type` 为 JSON 的响应做解析与改写。
    - 改写后用 `new Response(JSON.stringify(mutated), { headers })` 返回给页面，实现“页面拿到的就是被改写后的包”。
  - **XHR 改写要点**：
    - 通过覆写 `XMLHttpRequest.prototype.open/send` 获取 URL 与时延等元信息。
    - 在 `readyState === 4` 时对 `responseText` 做解析，并使用实例级 `Object.defineProperty` 覆写 getter（不同浏览器存在差异，需容错）。
  - **边界与注意**：
    - 对 `fetch` 的 `ReadableStream` 流式返回（SSE/streaming）改写成本更高，通常需转向“旁路观测”或在业务层提供非流式接口。
    - 对跨域请求的可见性与可改写性受页面自身请求方式、CORS 策略影响；本方案更适合页面已可正常访问的 API 流量做“内容级治理”。

  - **精炼核心代码（页面上下文 pageHook）**

```javascript
(function () {
  const send = (payload) => window.postMessage({ __from: "agentmark", ...payload }, "*");

  const shouldHandle = (url, ct) => /application\/json/i.test(ct || "") && /agent|chat|completions/i.test(url || "");

  const mutateAgentJson = (data) => {
    const out = { ...data };
    const content = out?.choices?.[0]?.message?.content;
    if (typeof content === "string") {
      const risk = /prompt\s*injection|system\s*override/i.test(content) ? 0.9 : 0.1;
      out.guard_score = risk;
      if (risk > 0.8) {
        out.blocked = true;
        out.choices[0].message.content = "[Guarded] 回复被策略拦截。";
      }
    }
    return out;
  };

  const originalFetch = window.fetch;
  window.fetch = async function (...args) {
    const input = args[0];
    const url = (typeof input === "string" ? input : input?.url) || "";
    const start = performance.now();
    const res = await originalFetch.apply(this, args);
    try {
      const ct = res.headers.get("content-type") || "";
      if (!shouldHandle(url, ct)) return res;
      const data = await res.clone().json();
      const mutated = mutateAgentJson(data);
      send({ kind: "fetch", url, t: Date.now(), latency_ms: Math.round(performance.now() - start), mutated: mutated.blocked === true });
      const body = JSON.stringify(mutated);
      const headers = new Headers(res.headers);
      headers.set("content-length", String(new Blob([body]).size));
      return new Response(body, { status: res.status, statusText: res.statusText, headers });
    } catch (_) {
      return res;
    }
  };

  const open = XMLHttpRequest.prototype.open;
  const sendX = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    this.__agentmark_url = url;
    return open.call(this, method, url, ...rest);
  };
  XMLHttpRequest.prototype.send = function (body) {
    const start = performance.now();
    this.addEventListener("readystatechange", () => {
      if (this.readyState !== 4) return;
      try {
        const url = this.__agentmark_url;
        const ct = this.getResponseHeader("content-type") || "";
        if (!shouldHandle(url, ct)) return;
        const mutated = mutateAgentJson(JSON.parse(this.responseText));
        const text = JSON.stringify(mutated);
        try {
          Object.defineProperty(this, "responseText", { get: () => text });
          Object.defineProperty(this, "response", { get: () => text });
        } catch (_) {}
        send({ kind: "xhr", url, t: Date.now(), latency_ms: Math.round(performance.now() - start), mutated: mutated.blocked === true });
      } catch (_) {}
    });
    return sendX.call(this, body);
  };
})();
```

  - 由于内容脚本与页面隔离，需将 `hook` 逻辑以 `<script>` 注入至页面上下文；页面脚本通过 `window.postMessage` 将事件回传给内容脚本，再由内容脚本经 `Port` 上报 SW。

```javascript
// content.js
const port = chrome.runtime.connect({ name: "probe" });

// 注入 pageHook.js 到页面上下文
const s = document.createElement("script");
s.src = chrome.runtime.getURL("pageHook.js");
s.async = false; // 确保在早期阶段加载
document.documentElement.appendChild(s);
s.remove();

// 页面 -> 内容脚本 -> SW
window.addEventListener("message", (ev) => {
  if (!ev.data || ev.data.__from !== "agentmark") return;
  port.postMessage({ type: "probe.event", payload: ev.data });
});

// 心跳，保持 Port 活性
setInterval(() => port.postMessage({ type: "probe.ping", t: Date.now() }), 25000);
```

```javascript
// pageHook.js（页面上下文）
(function () {
  const send = (payload) => window.postMessage({ __from: "agentmark", ...payload }, "*");

  // ---- fetch hook ----
  const originalFetch = window.fetch;
  window.fetch = async function (...args) {
    const [input, init] = args;
    const url = (typeof input === "string" ? input : input.url) || "";
    const start = performance.now();
    const res = await originalFetch.apply(this, args);

    try {
      const ct = res.headers.get("content-type") || "";
      if (/application\/json/i.test(ct) && /agent|chat|completions/i.test(url)) {
        const clone = res.clone();
        const data = await clone.json();

        // 示例：实时修改 Agent 返回
        // 增加一个 guard_score 字段，或根据规则重写 message
        const mutated = { ...data };
        if (mutated.choices?.[0]?.message?.content) {
          const content = mutated.choices[0].message.content;
          const risk = /prompt\s*injection|system\s*override/i.test(content) ? 0.9 : 0.1;
          mutated.guard_score = risk;
          if (risk > 0.8) {
            mutated.choices[0].message.content = "[Guarded] 回复被策略拦截。";
            mutated.blocked = true;
          }
        }

        const body = JSON.stringify(mutated);
        const headers = new Headers(res.headers);
        headers.set("content-length", String(new Blob([body]).size));

        send({
          kind: "fetch",
          url,
          t: Date.now(),
          latency_ms: Math.round(performance.now() - start),
          mutated: mutated.blocked === true,
        });

        return new Response(body, {
          status: res.status,
          statusText: res.statusText,
          headers
        });
      }
    } catch (_) {}
    return res;
  };

  // ---- XHR hook ----
  (function () {
    const open = XMLHttpRequest.prototype.open;
    const sendX = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.open = function (method, url, ...rest) {
      this.__agentmark_url = url;
      return open.call(this, method, url, ...rest);
    };
    XMLHttpRequest.prototype.send = function (body) {
      const start = performance.now();
      this.addEventListener("readystatechange", () => {
        if (this.readyState === 4) {
          try {
            const ct = this.getResponseHeader("content-type") || "";
            if (/application\/json/i.test(ct) && /agent|chat|completions/i.test(this.__agentmark_url)) {
              const json = JSON.parse(this.responseText);
              const mutated = { ...json, guard_score: json.guard_score ?? 0.2 };
              if (mutated.guard_score > 0.85) {
                mutated.blocked = true;
              }
              const text = JSON.stringify(mutated);
              // 重写只读 responseText/response（实例层覆写在多数浏览器可行）
              try {
                Object.defineProperty(this, "responseText", { get: () => text });
                Object.defineProperty(this, "response", { get: () => text });
              } catch (_) {}
              send({
                kind: "xhr",
                url: this.__agentmark_url,
                t: Date.now(),
                latency_ms: Math.round(performance.now() - start),
                mutated: mutated.blocked === true,
              });
            }
          } catch (_) {}
        }
      });
      return sendX.call(this, body);
    };
  })();
})();
```

- **插件通信核心**
  - 内容脚本与 SW 通过 `Port` 持久连接；页面 `hook` 通过 `postMessage` 回传。
  - SW 作为汇聚点统一转发到后端（WS/HTTP），供前端实时可视化消费。

##### 4.1.3 关键代码展示：插件通信机制核心片段

#### 4.2 前端可视化系统实现 (Frontend Engineering)

##### 4.2.1 技术栈：Vue 3 + Vite + ECharts

- **技术栈**
  - Vue 3 + Vite 作为应用框架与构建。
  - ECharts 实现攻击链路径图、时序告警与统计面板。
  - Pinia/轻量事件总线维护告警与图数据状态。

##### 4.2.2 攻击链还原可视化：Payload 解码与操作路径图映射

- **攻击链还原可视化（Payload → 操作路径图）**
  - 约定探针上报中包含一个二进制 `payload`（Base64），它是对一次 Agent 操作的编码：
    - 示例位段：`[8b chainId][8b stepType][16b resourceId][16b parentId][8b severity][8b flags]...`
  - 前端将 `Base64 → Uint8Array → DataView` 解码，映射为图节点与有向边；以 `chainId` 聚合，`parentId → resourceId` 形成边。

```ts
// src/utils/attackChain.ts
export type ChainNode = { id: string; label: string; severity: number; type: number };
export type ChainEdge = { source: string; target: string };

export function decodePayload(b64: string) {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  const dv = new DataView(buf.buffer);

  const chainId = dv.getUint8(0);
  const stepType = dv.getUint8(1);
  const resourceId = dv.getUint16(2);
  const parentId = dv.getUint16(4);
  const severity = dv.getUint8(6) / 100; // 0~1

  const node: ChainNode = {
    id: String(resourceId),
    label: `R${resourceId}`,
    severity,
    type: stepType
  };
  const edges: ChainEdge[] = parentId ? [{ source: String(parentId), target: String(resourceId) }] : [];
  return { chainId, node, edges };
}
```

```vue
<!-- src/components/AttackChainGraph.vue -->
<template>
  <div ref="el" style="width:100%;height:420px" />
  <div class="legend">
    <span>红=高危 橙=中危 绿=低危</span>
  </div>
  <div v-if="lastEvent" class="hint">最近事件：{{ lastEvent.url }} ({{ lastEvent.kind }})</div>
  <div v-if="error" class="err">{{ error }}</div>
  <div v-if="connecting" class="hint">告警通道连接中...</div>
  <div v-if="!connecting && !error" class="hint">告警通道已连接</div>
  <div v-if="stats.total>0" class="hint">累计事件：{{ stats.total }} | 已拦截：{{ stats.blocked }}</div>
</template>

<script setup lang="ts">
import * as echarts from "echarts";
import { onMounted, onBeforeUnmount, ref } from "vue";
import { decodePayload } from "@/utils/attackChain";

const el = ref<HTMLDivElement | null>(null);
let chart: echarts.ECharts | null = null;

type Node = { id: string; name: string; value: number; itemStyle: any };
type Link = { source: string; target: string };

const nodes = new Map<string, Node>();
const links: Link[] = [];

const lastEvent = ref<any>(null);
const connecting = ref(true);
const error = ref<string | null>(null);
const stats = ref({ total: 0, blocked: 0 });

function sevColor(v: number) {
  if (v >= 0.8) return "#e02424"; // red
  if (v >= 0.5) return "#f59e0b"; // orange
  return "#10b981";              // green
}

function upsertGraph(payloadB64: string) {
  const { node, edges } = decodePayload(payloadB64);
  if (!nodes.has(node.id)) {
    nodes.set(node.id, {
      id: node.id,
      name: `${node.label}`,
      value: Math.round(node.severity * 100),
      itemStyle: { color: sevColor(node.severity) }
    });
  }
  for (const e of edges) links.push(e);
  chart?.setOption({
    tooltip: {},
    series: [
      {
        type: "graph",
        layout: "force",
        roam: true,
        draggable: true,
        label: { show: true },
        force: { repulsion: 120, edgeLength: 80 },
        data: Array.from(nodes.values()),
        links
      }
    ]
  });
}

onMounted(() => {
  chart = echarts.init(el.value!);
  chart.setOption({ series: [{ type: "graph", data: [], links: [] }] });

  const ws = new WebSocket("ws://localhost:7001/alerts");
  ws.onopen = () => (connecting.value = false);
  ws.onerror = (e) => {
    error.value = "无法连接告警服务";
  };
  ws.onmessage = (ev) => {
    const evt = JSON.parse(ev.data);
    lastEvent.value = evt;
    stats.value.total += 1;
    if (evt.mutated || evt.blocked) stats.value.blocked += 1;
    if (evt.payload_b64) upsertGraph(evt.payload_b64);
  };
});

onBeforeUnmount(() => {
  chart?.dispose();
});
</script>

<style scoped>
.legend { margin-top: 8px; color: #666; font-size: 12px; }
.hint { margin-top: 6px; color: #666; font-size: 12px; }
.err { margin-top: 6px; color: #e02424; font-size: 12px; }
</style>
```

- **实时告警模块（WS）**
  - 后端提供 `ws://localhost:7001/alerts` 广播通道；SW 将探针事件通过 HTTP 上报至后端，后端再以 WS 推送给前端，形成“插件 → 汇聚服务 → 前端”的实时链路。
  - 前端侧进行了连接状态与错误视图处理，保证体验友好。

##### 4.2.3 实时告警模块：WebSocket 消费插件上报异常数据

#### 4.3 算法的工程化适配

##### 4.3.1 轻量化：ONNX + WebAssembly（Web 环境离线推理）

- **轻量化路线 A：ONNX + WebAssembly（前端离线推理）**
  - 将 PyTorch 模型导出 ONNX，并在前端用 `onnxruntime-web` 以 `wasm` 后端推理，避免 GPU 依赖与后端延迟。

```python
# export.py —— 将 PyTorch 模型导出为 ONNX
import torch

class TinyGuard(torch.nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1), torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = TinyGuard()
dummy = torch.randn(1, 32)
torch.onnx.export(
    model, dummy, "guard.onnx",
    input_names=["feat"], output_names=["score"],
    dynamic_axes={"feat": {0: "N"}, "score": {0: "N"}},
    opset_version=17
)
```

```ts
// src/ml/guard.ts —— 浏览器端 ONNX Runtime Web 推理
import * as ort from "onnxruntime-web";

let session: ort.InferenceSession | null = null;
export async function initGuard() {
  session = await ort.InferenceSession.create("/models/guard.onnx", {
    executionProviders: ["wasm"],
  });
}

export async function guardScore(feat: Float32Array) {
  if (!session) throw new Error("guard not initialized");
  const N = 1, D = feat.length;
  const tensor = new ort.Tensor("float32", feat, [N, D]);
  const out = await session.run({ feat: tensor });
  const score = (out["score"] as ort.Tensor).data as Float32Array;
  return score[0];
}
```

- **工程路线 B：服务化推理（FastAPI/Flask）**
  - 将 PyTorch 模型常驻后端，通过 REST/WS 暴露，前端或 SW 以异步调用；适合复杂模型或需要集中治理的场景。

##### 4.3.2 服务化：通过 API 实时调用 Python 推理能力

```python
# app.py —— FastAPI 服务化
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()
model = torch.jit.load("guard.pt").eval()

class Req(BaseModel):
    feat: list[float]

@app.post("/score")
def score(req: Req):
    with torch.inference_mode():
        x = torch.tensor(req.feat).view(1, -1)
        s = torch.sigmoid(model(x)).item()
        return {"score": s}
```

```javascript
// background.js —— SW 侧调用服务化接口
async function scoreRemote(feat) {
  const r = await fetch("http://localhost:7001/score", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ feat })
  });
  const { score } = await r.json();
  return score;
}
```

- **拦截-评分-决策的闭环**
  - 页面 `pageHook` 捕获返回 → 内容脚本上报 → SW 获取轻量特征（如 token 计数、危险关键字比例、来源域名信誉）→ 调用 Route A/B 得到 `score` → 若超过阈值，修改返回包并标记 `blocked`。
  - 在 4.1 的示例中已演示了如何将评分写入 `guard_score` 并动态替换响应体。

##### 4.3.3 闭环：拦截-评分-决策的工程链路

---

以上实现了从浏览器端探针、数据实时汇聚到前端攻击链可视化与算法在线/离线推理的完整工程闭环；代码片段均可直接落地并与现有栈（Vue3 + Vite + ECharts + MV3 插件）配合运行。

