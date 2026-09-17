/* =========================================================================
 * 知津 · 校园规章智能问答 —— 前端逻辑
 * Vue 3 + Element Plus（本地 vendor 文件加载，免 npm 构建即可运行）
 *
 * 相比改造前，这一版新增四件事：
 *   1. 思维链展示：把 reasoning 事件流式渲染进可折叠面板；
 *   2. 阶段流水：实时显示一次问答跑了哪些环节、各花多久；
 *   3. 来源溯源：来源卡片标出「向量 / 关键词 / 双路命中」，并可定位到原文块；
 *   4. 反馈与追踪：点赞点踩 + 链路详情页。
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
      '学生请假需要办理什么手续？',
      '第七十七条是怎么规定的？',
      '考试违纪会受到什么处分？',
      '奖学金是怎么评定的？',
    ];

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
    // 后端只推「进入某阶段」事件，耗时由前端按相邻事件的时间差算。
    // 这样后端不用为了计时把每个阶段都 await 一遍，前端也能实时显示进度。
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
      assistant.pipeline.push({ key, label, state: 'active', ms: null, startedAt: now });
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
          risks: [],
          memoryItems: [],
          status: '',
          pending: false,
          messageId: item.id,
          traceId: item.trace_id || null,
          feedback: item.feedback || '',
          refused: item.role === 'assistant' && item.content.indexOf('手册中未找到相关内容') === 0,
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

    // ---------- 页面切换时按需加载 ----------
    watch(page, (value) => {
      if (value === 'kb') loadDocuments();
      if (value === 'memory') {
        loadMemories();
        loadMemoryUsers();
      }
      if (value === 'chat') loadConversations();
      if (value === 'trace') loadTraces();
    });

    watch(userId, async (value, oldValue) => {
      if (value === oldValue) return;
      conversationId.value = null;
      messages.value = [];
      await Promise.all([loadConversations(), loadMemories()]);
    });

    onMounted(async () => {
      await Promise.all([refreshStats(), loadConversations()]);
    });

    return {
      page, userId, conversations, conversationId, messages, draft, isStreaming,
      documents, memories, memoryUsers, uploading, scrollRef,
      chunkDrawer, traceDrawer, traceOverview, traces, traceOnlyRefused, badcases,
      stats, suggestions,
      formatTime, renderMarkdown, pct, isNumber, pretty, channelOf, channelLabel, riskTypes,
      planIntentLabel, planWasRewritten,
      satisfactionText, stageBars, traceStages,
      newConversation, openConversation, removeConversation,
      sendQuestion, stopStreaming, onKeydown,
      copyAnswer, rate,
      handleUpload, removeDocument, openChunks, locateChunk, removeMemory,
      openTrace, loadTraces, loadBadcases, exportBadcases,
    };
  },
}).use(ElementPlus, { locale: ElementPlusLocaleZhCn }).mount('#app');
