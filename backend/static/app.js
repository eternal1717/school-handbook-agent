/* =========================================================================
 * 知津 · 校园规章智能问答 —— 前端逻辑
 * Vue 3 + Element Plus（本地 vendor 文件加载，免 npm 构建即可运行）
 *
 * 相比改造前，这一版新增：
 *   1. 思维链展示：把 reasoning 事件流式渲染进可折叠面板；
 *   2. 阶段流水：实时显示一次问答跑了哪些环节、各花多久；
 *   3. 来源溯源：来源卡片标出「向量 / 关键词 / 双路命中」，并可定位到原文块；
 *   4. 反馈与追踪：点赞点踩 + 链路详情页；
 *   5. 规则诊断：学生描述自己的情况 → 匹配规章条文 + 数值阈值比对；
 *   6. 办事流程：从原文抽出的步骤时间线，可当待办清单勾选；
 *   7. 主题切换：亮 / 暗双主题，跟随系统偏好并记住选择。
 *
 * 有一条贯穿全文件的取舍：**能用原生 DOM 就别用组件库**。
 * Element Plus 只在表格、抽屉、下拉、提示这类「有复杂度」的地方用；
 * 按钮、分段控件、输入框、标签全部手写。
 * 原因是默认主题的视觉权重和这套设计系统冲突，覆写它的成本比自写还高。
 * ========================================================================= */
const { createApp, ref, reactive, computed, watch, nextTick, onMounted } = Vue;
const { ElMessage, ElMessageBox } = ElementPlus;

// 防 XSS：模型输出里的 HTML 先转义，再交给 marked 渲染 Markdown
function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

marked.setOptions({ breaks: true, gfm: true });

createApp({
  setup() {
    // ---------- 状态 ----------
    const page = ref('chat');
    const userId = ref('default_user');
    const conversations = ref([]);
    const conversationId = ref(null);
    const messages = ref([]);
    const draft = ref('');
    const isStreaming = ref(false);
    const documents = ref([]);
    const memories = ref([]);
    const memoryUsers = ref([]);
    const uploading = ref(false);
    const scrollRef = ref(null);

    const chunkDrawer = reactive({ open: false, loading: false, title: '', chunks: [], highlight: null });
    const traceDrawer = reactive({ open: false, loading: false, data: null });
    const traceOverview = reactive({ total: 0, avg_ms: 0, p50_ms: 0, p95_ms: 0, refusal_rate: 0, avg_hops: 0, stage_avg_ms: {} });
    const traces = ref([]);
    const traceOnlyRefused = ref(false);
    const badcases = ref([]);

    const stats = reactive({
      documents: 0, vectors: 0, conversations: 0, memories: 0, bm25_terms: 0,
      llm_mode: 'mock', llm_model: '',
      pipeline: { hybrid: false, rerank: false, agent: false, max_hops: 0, guard: false },
      feedback: { up: 0, down: 0, total: 0, satisfaction: null },
    });

    let controller = null;

    const suggestions = [
      { value: '学生请假需要办理什么手续？', tag: '规章' },
      { value: '第七十七条是怎么规定的？', tag: '精确条号' },
      { value: '考试违纪会受到什么处分？', tag: '处分' },
      { value: '如果一个学期挂了好几门课，又缺课太多，会同时触发哪些处理？', tag: '多跳' },
      { value: '奖学金是怎么评定的？', tag: '流程' },
      { value: '保研需要什么条件？', tag: '该拒答' },
    ];

    // ---------- 主题 ----------
    // 首屏主题已由 index.html 里的内联脚本定好（避免闪一帧亮色），
    // 这里只需读回来，之后靠 watch 同步到 <html>。
    const theme = ref(document.documentElement.getAttribute('data-theme') || 'light');

    function toggleTheme() {
      theme.value = theme.value === 'dark' ? 'light' : 'dark';
    }

    watch(theme, (value) => {
      document.documentElement.setAttribute('data-theme', value);
      try {
        localStorage.setItem('zhijin-theme', value);
      } catch (error) {
        /* 隐私模式写不了，忽略即可，本次会话内主题仍然生效 */
      }
    });

    // ---------- 规则诊断 ----------
    const ruleTab = ref('diagnose');
    const ruleStats = ref({ rule_total: 0, procedure_total: 0, with_threshold: 0, chunk_total: 0, by_kind: {}, built_at: '' });
    const diagInput = ref('');
    const diagLoading = ref(false);
    const diagResult = ref(null);
    const diagExamples = ['我这学期挂了 3 门课，还缺课 20 学时', '我已经休学两次了，还能再休吗'];

    const ruleKinds = ref({});
    const ruleKind = ref('');
    const ruleQuery = ref('');
    const ruleList = ref([]);
    const ruleListTotal = ref(0);

    // ---------- 办事流程 ----------
    const procedures = ref([]);
    const procTotal = ref(0);
    const procQuery = ref('');
    const procLoading = ref(false);
    const currentProc = ref(null);
    // 勾选状态按「流程 id → 步骤下标集合」存，切换流程时各自保留。
    // 不放进 currentProc 里，是因为 currentProc 会被重新赋值为新对象。
    const stepChecks = ref({});

    // ---------- 通用工具 ----------
    function formatTime(value) {
      if (!value) return '';
      return String(value).replace('T', ' ').slice(0, 16);
    }

    function renderMarkdown(text) {
      if (!text) return '';
      try {
        return marked.parse(escapeHtml(text));
      } catch (error) {
        return escapeHtml(text);
      }
    }

    // 0.4286 → 42.9；null / undefined → '—'
    function pct(value) {
      if (value === null || value === undefined) return '—';
      return (Number(value) * 100).toFixed(1);
    }

    function isNumber(value) {
      return typeof value === 'number' && !Number.isNaN(value);
    }

    function pretty(value) {
      if (value === null || value === undefined) return '—';
      try {
        return JSON.stringify(value, null, 2);
      } catch (error) {
        return String(value);
      }
    }

    // 来源命中的通道：同时被两路召回的最可信
    function channelOf(source) {
      const byVector = isNumber(source.vector_rank);
      const byBm25 = isNumber(source.bm25_rank);
      if (byVector && byBm25) return 'both';
      if (byVector) return 'vector';
      if (byBm25) return 'bm25';
      return '';
    }

    function channelLabel(source) {
      const map = { both: '双路命中', vector: '向量召回', bm25: '关键词命中' };
      return map[channelOf(source)] || '来源';
    }

    // 阶段名 → 给人看的说法。
    // 事件里（stage 事件）本来就带 label，但历史会话是从 trace 里回填的，
    // 那里只有 stage 的机器名（understand / reflect_1 / retry_2 …），得在这里翻译一遍。
    const STAGE_LABELS = {
      understand: '理解问题',
      plan: '理解问题',
      agent: '多轮检索',
      agent_loop: '多轮检索',
      agent_retrieval: '多轮检索',
      retrieval: '检索知识库',
      skip_retrieval: '无需检索',
      rerank: '重排条款',
      reflect: '自省核查',
      retry: '换个问法重查',
      generate: '生成回答',
      memory: '写入记忆',
    };

    function stageLabel(name) {
      // reflect_1 / retry_2 这类带序号的取前缀，否则会把每个跳数都变成新阶段
      const base = String(name || '').replace(/_\d+$/, '');
      return STAGE_LABELS[base] || String(name || '');
    }

    function riskTypes(risks) {
      const names = [];
      for (const item of risks || []) {
        for (const category of item.categories || []) {
          if (!names.includes(category)) names.push(category);
        }
      }
      return names.join('、');
    }

    // 意图取值来自规划层的固定枚举，这里只是给人看的说法
    function planIntentLabel(intent) {
      const map = {
        regulation: '查规章',
        procedure: '查流程',
        chitchat: '闲聊',
        out_of_scope: '越界提问',
      };
      return map[intent] || intent || '未判定';
    }

    // 只有真的改写了才显示——否则「改写后」和原问题一模一样，纯属干扰
    function planWasRewritten(plan, original) {
      if (!plan || !plan.rewritten) return false;
      return normalizeForCompare(plan.rewritten) !== normalizeForCompare(original);
    }

    function normalizeForCompare(text) {
      return String(text || '').replace(/\s+/g, '').replace(/[？?。.！!，,]/g, '');
    }

    // ---------- 顶栏文案 ----------
    const PAGE_META = {
      chat: ['问答', '引用来源可逐条核对，手册里没有的会明确拒答'],
      rules: ['规则诊断', '把条文抽成「条件 → 后果」，再和你的具体数值比对'],
      proc: ['办事流程', '从原文抽出的办理步骤，可当待办清单勾选'],
      kb: ['知识库', '已入库文档与分块，改这里会同步重建检索索引'],
      memory: ['长期记忆', '跨会话记住的内容，回答时会注入给模型'],
      trace: ['链路追踪', '每次问答的全过程留痕，用来定位是哪一环出了问题'],
    };

    const pageTitle = computed(() => (PAGE_META[page.value] || ['', ''])[0]);
    const pageDesc = computed(() => (PAGE_META[page.value] || ['', ''])[1]);

    const userInitial = computed(() => String(userId.value || '?').slice(0, 1));

    // 切页时顺手把该页要用的数据拉上。
    // 单独抽成函数是因为侧栏导航和「新建会话」都要用同一套逻辑。
    function goPage(target) {
      page.value = target;
    }

    function scrollToBottom() {
      nextTick(() => {
        const el = scrollRef.value;
        if (el) el.scrollTop = el.scrollHeight;
      });
    }

    async function api(url, options = {}) {
      const response = await fetch(url, options);
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`HTTP ${response.status} ${text.slice(0, 200)}`);
      }
      return response.json();
    }

    // ---------- 阶段流水 ----------
    // 后端只推「进入某阶段」事件，耗时先按相邻事件的时间差估算——
    // 这样不用为了计时把每个阶段都 await 一遍，进度也能实时显示。
    // 注意这个估算值是不准的（SSE 事件会成批到达），所以 done 事件里
    // 后端会带回真实耗时，到时候整体覆盖一次。见 done 分支。
    function pushStage(assistant, key, label) {
      const now = performance.now();
      for (let i = assistant.pipeline.length - 1; i >= 0; i -= 1) {
        const stage = assistant.pipeline[i];
        if (stage.state === 'active') {
          stage.state = 'done';
          stage.ms = Math.round(now - stage.startedAt);
          break;
        }
      }
      // chip 用短名（「理解问题」），status 行用完整文案（「正在理解问题…」）。
      // 单独存一个字段而不是共用，是因为 chip 要短才好扫读，状态行要完整才像人话。
      assistant.pipeline.push({
        key, label, chip: stageLabel(key), state: 'active', ms: null, startedAt: now,
      });
      assistant.status = label;
    }

    function finishStages(assistant) {
      const now = performance.now();
      for (const stage of assistant.pipeline) {
        if (stage.state === 'active') {
          stage.state = 'done';
          stage.ms = Math.round(now - stage.startedAt);
        }
      }
      assistant.status = '';
    }

    // ---------- 数据加载 ----------
    async function refreshStats() {
      try {
        Object.assign(stats, await api('/api/stats'));
      } catch (error) {
        console.error('加载统计失败', error);
      }
    }

    async function loadConversations() {
      try {
        conversations.value = await api(`/api/conversations?user_id=${encodeURIComponent(userId.value)}`);
      } catch (error) {
        console.error('加载会话失败', error);
      }
    }

    async function loadDocuments() {
      try {
        documents.value = await api('/api/knowledge/list');
      } catch (error) {
        ElMessage.error('加载知识库失败：' + error.message);
      }
    }

    async function loadMemories() {
      try {
        memories.value = await api(`/api/memory?user_id=${encodeURIComponent(userId.value)}`);
      } catch (error) {
        ElMessage.error('加载长期记忆失败：' + error.message);
      }
    }

    async function loadMemoryUsers() {
      try {
        memoryUsers.value = await api('/api/memory/users');
      } catch (error) {
        console.error('加载用户列表失败', error);
      }
    }

    async function loadTraces() {
      try {
        const query = `?limit=30${traceOnlyRefused.value ? '&only_refused=true' : ''}`;
        traces.value = await api('/api/trace' + query);
        Object.assign(traceOverview, await api('/api/trace/overview'));
      } catch (error) {
        ElMessage.error('加载链路失败：' + error.message);
      }
    }

    async function loadBadcases() {
      try {
        const result = await api('/api/feedback/badcases?limit=50');
        badcases.value = result.items || [];
        if (!badcases.value.length) ElMessage.info('暂时还没有点踩记录');
      } catch (error) {
        ElMessage.error('加载失败：' + error.message);
      }
    }

    async function exportBadcases() {
      try {
        const result = await api('/api/feedback/export?limit=50');
        if (!result.count) {
          ElMessage.info('还没有点踩记录，先点几个「没帮助」再来导出');
          return;
        }
        // 直接展示成可复制的 JSON，用户粘进 tests/eval_dataset.json 即可
        const blob = new Blob([JSON.stringify(result.cases, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'badcases_for_eval.json';
        link.click();
        URL.revokeObjectURL(url);
        ElMessage.success(`已导出 ${result.count} 条，补上 expected_keywords 后即可并入评测集`);
      } catch (error) {
        ElMessage.error('导出失败：' + error.message);
      }
    }

    // ---------- 规则诊断 ----------
    async function loadRuleStats() {
      try {
        const result = await api('/api/rulebook/stats');
        if (result.enabled === false) {
          ElMessage.warning('规则库已在配置中关闭（RULEBOOK_ENABLED=false）');
          return;
        }
        ruleStats.value = result;
      } catch (error) {
        console.error('加载规则库统计失败', error);
      }
    }

    async function runDiagnose() {
      const situation = diagInput.value.trim();
      if (!situation || diagLoading.value) return;
      diagLoading.value = true;
      diagResult.value = null;
      try {
        diagResult.value = await api('/api/rulebook/diagnose', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ situation }),
        });
      } catch (error) {
        ElMessage.error('诊断失败：' + error.message);
      } finally {
        diagLoading.value = false;
      }
    }

    async function loadRules() {
      try {
        const params = new URLSearchParams({ limit: '30' });
        if (ruleQuery.value.trim()) params.set('q', ruleQuery.value.trim());
        if (ruleKind.value) params.set('kind', ruleKind.value);
        const result = await api('/api/rulebook/rules?' + params.toString());
        ruleList.value = result.items || [];
        ruleListTotal.value = result.total || 0;
        if (!Object.keys(ruleKinds.value).length) ruleKinds.value = result.kinds || {};
      } catch (error) {
        ElMessage.error('加载规则失败：' + error.message);
      }
    }

    async function switchBrowseTab() {
      ruleTab.value = 'browse';
      // 只在首次进入时加载，避免每次切回来都重打一次接口
      if (!ruleList.value.length) await loadRules();
    }

    async function filterRules(kind) {
      ruleKind.value = kind;
      // 换种类时清掉关键词——两者是「与」的关系，不清会搜出空结果让人困惑
      ruleQuery.value = '';
      await loadRules();
    }

    // ---------- 办事流程 ----------
    async function loadProcedures() {
      if (procLoading.value) return;
      procLoading.value = true;
      try {
        const params = new URLSearchParams({ limit: '50' });
        if (procQuery.value.trim()) params.set('q', procQuery.value.trim());
        const result = await api('/api/rulebook/procedures?' + params.toString());
        procedures.value = result.items || [];
        procTotal.value = result.total || 0;
        // 当前选中的流程如果被新结果挤出去了，自动落到第一条，避免右侧停在旧内容上
        if (!procedures.value.some((p) => currentProc.value && p.id === currentProc.value.id)) {
          currentProc.value = procedures.value[0] || null;
        }
      } catch (error) {
        ElMessage.error('加载流程失败：' + error.message);
      } finally {
        procLoading.value = false;
      }
    }

    function openProcedure(item) {
      currentProc.value = item;
    }

    function isStepChecked(procId, index) {
      const set = stepChecks.value[procId];
      return Boolean(set && set.includes(index));
    }

    function toggleStep(procId, index) {
      const set = stepChecks.value[procId] || [];
      const next = set.includes(index) ? set.filter((i) => i !== index) : set.concat(index);
      // 整体替换而不是原地改数组，否则 Vue 追踪不到这次变更
      stepChecks.value = { ...stepChecks.value, [procId]: next };
    }

    function checkedCount(item) {
      const set = stepChecks.value[item.id];
      return set ? set.length : 0;
    }

    // ---------- 会话操作 ----------
    function newConversation() {
      if (isStreaming.value) {
        ElMessage.warning('正在生成回答，请先停止或等待完成');
        return;
      }
      conversationId.value = null;
      messages.value = [];
      page.value = 'chat';
      draft.value = '';
    }

    async function openConversation(id) {
      if (isStreaming.value) {
        ElMessage.warning('正在生成回答，请先停止或等待完成');
        return;
      }
      page.value = 'chat';
      conversationId.value = id;
      try {
        const history = await api(
          `/api/conversations/${id}/messages?user_id=${encodeURIComponent(userId.value)}`
        );
        messages.value = history.map((item) => ({
          role: item.role,
          content: item.content,
          reasoning: item.reasoning || '',
          reasonOpen: item.role === 'assistant' && Boolean(item.reasoning),
          sources: (item.sources || []).map((s) => ({ ...s, open: false })),
          pipeline: [],
          plan: null,
          risks: [],
          memoryItems: [],
          status: '',
          pending: false,
          messageId: item.id,
          traceId: item.trace_id || null,
          feedback: item.feedback || '',
          refused: item.role === 'assistant' && item.content.indexOf('手册中未找到相关内容') === 0,
        }));

        // 把「上一条用户提问」挂到助手消息上。有了它，理解卡片才能判断
        // 「问题到底有没有被改写」，否则拿不出原始问题做对比。
        messages.value.forEach((m, i) => {
          if (m.role === 'assistant' && i > 0 && messages.value[i - 1].role === 'user') {
            m.questionText = messages.value[i - 1].content;
          }
        });

        // 回填链路信息。不做这一步，历史会话里「问题理解」和「阶段耗时」都是空的——
        // 而这些恰恰是最该被看到的部分：用户点开旧对话，也应该能看见
        // 当时问题被改写成了什么、每一步花了多久，而不是一条光秃秃的回答。
        const pending = messages.value.filter((m) => m.traceId).slice(-8);
        await Promise.all(pending.map(async (m) => {
          try {
            const trace = await api(`/api/trace/${m.traceId}`);
            if (trace.plan) {
              m.plan = {
                intent: trace.plan.intent,
                complexity: trace.plan.complexity,
                rewritten: trace.rewritten || '',
                subQueries: trace.plan.sub_queries || [],
                needsRetrieval: trace.plan.needs_retrieval,
                reason: trace.plan.reason || '',
              };
            }
            m.pipeline = (trace.stages || [])
              .filter((s) => s.ms > 0)
              .map((s) => ({ key: s.stage, label: stageLabel(s.stage), state: 'done', ms: s.ms }));
          } catch (error) {
            // 链路可能已被清理，或这条消息本来就没记链路——静默跳过，
            // 不能因为回填失败就让整个会话打不开
          }
        }));

        scrollToBottom();
      } catch (error) {
        ElMessage.error('加载历史消息失败：' + error.message);
      }
    }

    async function removeConversation(id) {
      try {
        await ElMessageBox.confirm('确定删除这个会话及其全部消息吗？', '删除会话', { type: 'warning' });
      } catch {
        return;
      }
      try {
        await api(`/api/conversations/${id}`, { method: 'DELETE' });
        if (conversationId.value === id) newConversation();
        await loadConversations();
        await refreshStats();
        ElMessage.success('已删除');
      } catch (error) {
        ElMessage.error('删除失败：' + error.message);
      }
    }

    // ---------- 事件分流 ----------
    function handleEvent(event, assistant) {
      switch (event.type) {
        case 'meta':
          if (event.conversation_id) conversationId.value = event.conversation_id;
          break;

        case 'stage':
          pushStage(assistant, event.stage, event.label || event.stage);
          break;

        case 'plan':
          // 意图和复杂度不铺开显示，但把改写结果留作「可解释性」的证据
          assistant.plan = {
            intent: event.intent,
            complexity: event.complexity,
            needsRetrieval: event.needs_retrieval,
            rewritten: event.rewritten,
            subQueries: event.sub_queries || [],
            reason: event.reason || '',
          };
          break;

        case 'retrieval':
          assistant.retrieval = event.stats || {};
          break;

        case 'rerank':
          assistant.rerank = event;
          break;

        case 'reflection':
          // 自省每跳记一条，用于展示「资料不足 → 换问法重查」的过程
          if (!assistant.reflections) assistant.reflections = [];
          assistant.reflections.push({
            hop: event.hop,
            sufficient: event.sufficient,
            missing: event.missing || '',
          });
          break;

        case 'guard':
          // 提问侧和检索侧都会各发一次 guard 事件，要累加而不是覆盖，
          // 否则前面那次（用户提问里的可疑内容）会被后面这次冲掉。
          assistant.risks = [...assistant.risks, ...(event.risks || [])];
          break;

        case 'sources':
          assistant.sources = (event.sources || []).map((s) => ({ ...s, open: false }));
          break;

        case 'reasoning':
          // 思维链单独累加，默认展开：让用户看见推理过程，而不是黑盒
          assistant.reasoning += event.content || '';
          assistant.reasonOpen = true;
          scrollToBottom();
          break;

        case 'token':
          assistant.content += event.content || '';
          // 正文一开始输出就自动收起思维链，避免刷屏；用户想看可以再展开
          if (assistant.reasoning && assistant.reasonAutoOpen !== false) {
            assistant.reasonOpen = false;
            assistant.reasonAutoOpen = false;
          }
          scrollToBottom();
          break;

        case 'status':
          assistant.status = event.message;
          break;

        case 'memory':
          assistant.memoryItems = event.items || [];
          ElMessage.success('已更新长期记忆');
          break;

        case 'refusal':
          assistant.refused = true;
          break;

        case 'done':
          if (event.message_id) assistant.messageId = event.message_id;
          if (event.trace_id) assistant.traceId = event.trace_id;
          if (isNumber(event.max_similarity)) assistant.maxSimilarity = event.max_similarity;
          // 用后端的真实阶段耗时覆盖前端的估算值。
          // 上面的 pushStage 是按事件到达时间差估的，SSE 成批到达时会算出
          // 「理解问题 0ms、筛选条款 9 秒」这种明显不对的数字。
          // 后端 recorder 里本来就是真实计时，用它重排一遍最省事也最准。
          if (Array.isArray(event.stages) && event.stages.length) {
            assistant.pipeline = event.stages
              .filter((s) => s.ms > 0)
              .map((s) => ({
                key: s.stage,
                label: stageLabel(s.stage),
                chip: stageLabel(s.stage),
                state: 'done',
                ms: s.ms,
              }));
          }
          break;

        case 'error':
          assistant.content += `\n\n[服务端错误] ${event.message}`;
          ElMessage.error(event.message);
          break;

        default:
          break;
      }
    }

    // ---------- 提问 ----------
    async function sendQuestion() {
      const question = draft.value.trim();
      if (!question || isStreaming.value) return;

      isStreaming.value = true;
      const assistant = reactive({
        role: 'assistant',
        content: '',
        reasoning: '',
        reasonOpen: true,
        reasonAutoOpen: true,
        sources: [],
        pipeline: [],
        risks: [],
        reflections: [],
        plan: null,
        // 留一份原始问题：理解卡片要拿它和「改写后」比对，
        // 只有真的改写了才显示改写结果，否则会平白多出一行噪音。
        questionText: question,
        retrieval: null,
        rerank: null,
        memoryItems: [],
        status: '正在理解问题…',
        pending: true,
        refused: false,
        messageId: null,
        traceId: null,
        feedback: '',
        maxSimilarity: null,
      });

      messages.value.push({
        role: 'user', content: question, sources: [], pipeline: [],
        memoryItems: [], status: '', pending: false,
      });
      messages.value.push(assistant);
      draft.value = '';
      scrollToBottom();

      controller = new AbortController();
      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question,
            user_id: userId.value,
            conversation_id: conversationId.value,
          }),
          signal: controller.signal,
        });

        if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        // 手动解析 SSE：按空行切分事件块，避免半截包导致 JSON.parse 报错
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          let boundary = buffer.indexOf('\n\n');
          while (boundary >= 0) {
            const block = buffer.slice(0, boundary);
            buffer = buffer.slice(boundary + 2);
            const dataLine = block.split('\n').find((line) => line.startsWith('data:'));
            if (dataLine) {
              try {
                handleEvent(JSON.parse(dataLine.slice(5).trim()), assistant);
              } catch (error) {
                console.warn('无法解析事件', dataLine, error);
              }
            }
            boundary = buffer.indexOf('\n\n');
          }
        }
      } catch (error) {
        if (error.name === 'AbortError') {
          assistant.content += '\n\n（已停止生成）';
        } else {
          assistant.content += `\n\n[请求失败] ${error.message}`;
          ElMessage.error('请求失败：' + error.message);
        }
      } finally {
        assistant.pending = false;
        finishStages(assistant);
        isStreaming.value = false;
        controller = null;
        scrollToBottom();
        loadConversations();
        refreshStats();
      }
    }

    function stopStreaming() {
      if (controller) controller.abort();
    }

    function onKeydown(event) {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendQuestion();
      }
    }

    // ---------- 回答操作 ----------
    async function copyAnswer(message) {
      try {
        await navigator.clipboard.writeText(message.content || '');
        ElMessage.success('已复制回答');
      } catch (error) {
        ElMessage.error('复制失败，请手动选择文本');
      }
    }

    async function rate(message, rating) {
      if (!message.messageId) {
        ElMessage.warning('这条回答还没落库，稍等一下再评价');
        return;
      }
      // 再点一次同一个按钮 = 取消评价
      if (message.feedback === rating) {
        try {
          await api(`/api/feedback/${message.messageId}?user_id=${encodeURIComponent(userId.value)}`,
            { method: 'DELETE' });
          message.feedback = '';
          await refreshStats();
        } catch (error) {
          ElMessage.error('操作失败：' + error.message);
        }
        return;
      }

      let comment = '';
      if (rating === 'down') {
        // 点踩时问一句原因——这句话之后会直接成为评测集里的备注
        try {
          const result = await ElMessageBox.prompt(
            '哪里不对？简单说一句就好（可留空），它会成为后续优化的 badcase 记录。',
            '反馈问题', { inputPlaceholder: '例如：答非所问 / 引用了不相关的条款 / 内容不完整', confirmButtonText: '提交', cancelButtonText: '跳过' }
          );
          comment = result.value || '';
        } catch {
          comment = '';
        }
      }

      try {
        await api('/api/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message_id: message.messageId,
            rating,
            user_id: userId.value,
            conversation_id: conversationId.value,
            trace_id: message.traceId,
            question: lastUserQuestion(),
            comment,
          }),
        });
        message.feedback = rating;
        ElMessage.success(rating === 'up' ? '感谢反馈' : '已记录，会用于优化检索');
        await refreshStats();
      } catch (error) {
        ElMessage.error('提交失败：' + error.message);
      }
    }

    function lastUserQuestion() {
      for (let i = messages.value.length - 1; i >= 0; i -= 1) {
        if (messages.value[i].role === 'user') return messages.value[i].content || '';
      }
      return '';
    }

    // ---------- 知识库操作 ----------
    async function handleUpload(options) {
      const file = options.file;
      uploading.value = true;
      try {
        const form = new FormData();
        form.append('file', file);
        const result = await api('/api/knowledge/upload', { method: 'POST', body: form });
        if (result.duplicated) {
          ElMessage({ type: 'warning', duration: 6000, message: result.message });
        } else if (result.skipped_chunk_count) {
          ElMessage({
            type: 'success', duration: 6000,
            message: `《${result.filename}》入库完成：新增 ${result.chunk_count} 块，`
              + `跳过重复 ${result.skipped_chunk_count} 块，向量库共 ${result.vector_total} 条`,
          });
        } else {
          ElMessage.success(
            `入库成功：${result.filename}，新增 ${result.chunk_count} 块，向量库共 ${result.vector_total} 条`
          );
        }
        await loadDocuments();
        await refreshStats();
      } catch (error) {
        ElMessage.error('入库失败：' + error.message);
      } finally {
        uploading.value = false;
      }
    }

    async function removeDocument(row) {
      try {
        await ElMessageBox.confirm(`确定删除《${row.filename}》及其全部向量吗？`, '删除文档', { type: 'warning' });
      } catch {
        return;
      }
      try {
        await api(`/api/knowledge/${row.id}`, { method: 'DELETE' });
        await loadDocuments();
        await refreshStats();
        ElMessage.success('已删除');
      } catch (error) {
        ElMessage.error('删除失败：' + error.message);
      }
    }

    // ---------- 知识块查看 / 原文定位 ----------
    async function openChunks(row, highlightIndex = null) {
      chunkDrawer.open = true;
      chunkDrawer.loading = true;
      chunkDrawer.title = row.filename;
      chunkDrawer.chunks = [];
      chunkDrawer.highlight = highlightIndex;
      try {
        const result = await api(`/api/knowledge/${row.id}/chunks`);
        chunkDrawer.chunks = result.chunks;
        chunkDrawer.title = `${row.filename}（${result.count} 个知识块）`;
        if (highlightIndex !== null) {
          nextTick(() => {
            const items = document.querySelectorAll('.chunk-panel .chunk-item');
            if (items[highlightIndex]) items[highlightIndex].scrollIntoView({ block: 'center' });
          });
        }
      } catch (error) {
        ElMessage.error('加载知识块失败：' + error.message);
      } finally {
        chunkDrawer.loading = false;
      }
    }

    // 从来源卡片跳回知识库原文：这是「引用可验证」的关键一步，
    // 用户能亲眼看到答案是从哪一段原文来的。
    async function locateChunk(source) {
      const docId = source.doc_id || String(source.id || '').split('_')[0];
      const index = isNumber(source.chunk_index) ? source.chunk_index : null;
      if (!docId) {
        ElMessage.warning('这条来源缺少文档信息');
        return;
      }
      let target = documents.value.find((doc) => doc.id === docId);
      if (!target) {
        await loadDocuments();
        target = documents.value.find((doc) => doc.id === docId);
      }
      if (!target) {
        ElMessage.warning('原文档可能已被删除');
        return;
      }
      await openChunks(target, index);
    }

    // ---------- 记忆操作 ----------
    async function removeMemory(row) {
      try {
        await ElMessageBox.confirm('确定删除这条长期记忆吗？', '删除记忆', { type: 'warning' });
      } catch {
        return;
      }
      try {
        await api(`/api/memory/${row.id}`, { method: 'DELETE' });
        await loadMemories();
        await loadMemoryUsers();
        await refreshStats();
        ElMessage.success('已删除');
      } catch (error) {
        ElMessage.error('删除失败：' + error.message);
      }
    }

    // ---------- 链路详情 ----------
    async function openTrace(traceIdOrRow) {
      const traceId = typeof traceIdOrRow === 'string' ? traceIdOrRow : traceIdOrRow.id;
      if (!traceId) return;
      traceDrawer.open = true;
      traceDrawer.loading = true;
      traceDrawer.data = null;
      try {
        traceDrawer.data = await api(`/api/trace/${traceId}`);
      } catch (error) {
        ElMessage.error('加载链路详情失败：' + error.message);
      } finally {
        traceDrawer.loading = false;
      }
    }

    // ---------- 图表数据 ----------
    const satisfactionText = computed(() => {
      const value = stats.feedback && stats.feedback.satisfaction;
      if (value === null || value === undefined) return '—';
      return `${(Number(value) * 100).toFixed(0)}%`;
    });

    const stageBars = computed(() => {
      const source = traceOverview.stage_avg_ms || {};
      const entries = Object.entries(source).filter(([, ms]) => ms > 0);
      if (!entries.length) return [];
      const max = Math.max(...entries.map(([, ms]) => ms));
      return entries
        .sort((a, b) => b[1] - a[1])
        .map(([name, ms]) => ({ name, ms, width: Math.max(3, Math.round((ms / max) * 100)) }));
    });

    const traceStages = computed(() => {
      const stages = (traceDrawer.data && traceDrawer.data.stages) || [];
      const max = Math.max(1, ...stages.map((s) => s.ms || 0));
      return stages.map((s) => ({
        stage: s.stage,
        ms: s.ms || 0,
        width: Math.max(3, Math.round(((s.ms || 0) / max) * 100)),
      }));
    });

    // 浏览器前进 / 后退、或者有人手改地址栏 hash 时，页面要跟着走。
    // 不监听的话，<a href="#rules"> 这类链接点了没反应——因为只改 hash
    // 不会触发页面重载，Vue 这边就完全不知道地址变了。
    window.addEventListener('hashchange', () => {
      const target = (location.hash || '').replace('#', '');
      if (PAGE_META[target] && target !== page.value) page.value = target;
    });

    // ---------- 页面切换时按需加载 ----------
    // 顺便把页面写进 URL hash：刷新后还停在原页，也能把某一页直接发给别人
    watch(page, (value) => {
      if (location.hash !== '#' + value) location.hash = value;
      if (value === 'kb') loadDocuments();
      if (value === 'memory') {
        loadMemories();
        loadMemoryUsers();
      }
      if (value === 'chat') loadConversations();
      if (value === 'trace') loadTraces();
      if (value === 'rules') loadRuleStats();
      // 流程列表一次就够，之后靠搜索框刷新，不必每次切页都重拉
      if (value === 'proc' && !procedures.value.length) loadProcedures();
    });

    watch(userId, async (value, oldValue) => {
      if (value === oldValue) return;
      conversationId.value = null;
      messages.value = [];
      await Promise.all([loadConversations(), loadMemories()]);
    });

    onMounted(async () => {
      // 带 hash 进来就直接落到那一页（分享链接、刷新保持页面都靠它）。
      // 赋值 page 会触发上面的 watch，该页的数据随之加载，不用在这里重复拉。
      const initial = (location.hash || '').replace('#', '');
      if (PAGE_META[initial]) page.value = initial;

      // 规则库统计也一起拉：首屏空状态要显示「抽了多少条规则」，
      // 这是第一眼就能传达「这项目不止是问答」的地方。
      await Promise.all([refreshStats(), loadConversations(), loadRuleStats()]);
    });

    return {
      page, userId, conversations, conversationId, messages, draft, isStreaming,
      documents, memories, memoryUsers, uploading, scrollRef,
      chunkDrawer, traceDrawer, traceOverview, traces, traceOnlyRefused, badcases,
      stats, suggestions,
      theme, toggleTheme, pageTitle, pageDesc, userInitial, goPage,
      ruleTab, ruleStats, diagInput, diagLoading, diagResult, diagExamples,
      ruleKinds, ruleKind, ruleQuery, ruleList, ruleListTotal,
      procedures, procTotal, procQuery, procLoading, currentProc,
      formatTime, renderMarkdown, pct, isNumber, pretty, channelOf, channelLabel, riskTypes,
      planIntentLabel, planWasRewritten,
      satisfactionText, stageBars, traceStages,
      newConversation, openConversation, removeConversation,
      sendQuestion, stopStreaming, onKeydown,
      copyAnswer, rate,
      handleUpload, removeDocument, openChunks, locateChunk, removeMemory,
      openTrace, loadTraces, loadBadcases, exportBadcases,
      loadRuleStats, runDiagnose, loadRules, filterRules, switchBrowseTab,
      loadProcedures, openProcedure, toggleStep, isStepChecked, checkedCount,
    };
  },
}).use(ElementPlus, { locale: ElementPlusLocaleZhCn }).mount('#app');
