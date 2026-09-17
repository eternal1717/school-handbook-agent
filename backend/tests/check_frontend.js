/* 前端静态检查（不需要浏览器、不需要 npm 安装）。
 *
 * 用法：node tests/check_frontend.js
 *
 * 做三件事：
 *   1. 标签配对检查 —— 抓没闭合 / 闭合错位的标签；
 *   2. 模板里 @click 调用的方法是否都在 app.js 暴露了 —— 抓「点了没反应」；
 *   3. 模板引用的数据变量是否都暴露了 —— 抓「显示空白」。
 *
 * 说明：这里没有用 Vue 自带的编译器，因为浏览器版编译模板要用 document 解析 HTML，
 * Node 里跑不起来（要装 jsdom）。上面三项已经能覆盖绝大多数改前端时踩的坑。
 */
const fs = require('fs');
const path = require('path');

const STATIC = path.join(__dirname, '..', 'static');
const html = fs.readFileSync(path.join(STATIC, 'index.html'), 'utf8');
const appJs = fs.readFileSync(path.join(STATIC, 'app.js'), 'utf8');

// 注意：以前这里只截到「知识块查看抽屉」注释为止，导致抽屉（也就是 #app 的收尾部分）
// 从来没被检查过，结果抽屉被挪到 #app 外面都没发现——页面上就直接显示 {{ }} 原文。
// 现在改成从 <div id="app" 一直取到最后一个 </div>，保证整个 #app 都覆盖到。
const start = html.indexOf('<div id="app"');
const end = html.lastIndexOf('</div>') + '</div>'.length;
const template = html.slice(start, end);

let failed = 0;
function report(ok, name, detail = '') {
  console.log(`  ${ok ? '[OK]  ' : '[FAIL]'} ${name}${detail ? '  ' + detail : ''}`);
  if (!ok) failed += 1;
}

// ---------- 1. 标签配对 ----------
const VOID_TAGS = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
  'link', 'meta', 'param', 'source', 'track', 'wbr']);
const stack = [];
const tagErrors = [];
for (const m of template.matchAll(/<(\/?)([a-zA-Z][\w-]*)((?:"[^"]*"|'[^']*'|[^>"'])*?)(\/?)>/g)) {
  const [, closing, rawTag, , selfClose] = m;
  const tag = rawTag.toLowerCase();
  if (VOID_TAGS.has(tag)) continue;
  if (closing) {
    const top = stack.pop();
    if (top !== tag) tagErrors.push(`期望 </${top}>，实际 </${tag}>`);
  } else if (!selfClose) {
    stack.push(tag);
  }
}
if (stack.length) tagErrors.push('未闭合：' + stack.join(' > '));
report(tagErrors.length === 0, 'HTML 标签配对', tagErrors.join(' | ') || `(${template.length} 字符)`);

// ---------- 取出 app.js 暴露给模板的成员 ----------
const rIdx = appJs.lastIndexOf('return {');
const exposed = appJs
  .slice(rIdx + 'return {'.length, appJs.indexOf('};', rIdx))
  .split(',')
  .map((s) => s.trim())
  .filter((s) => /^[a-zA-Z_$][\w$]*$/.test(s));

// ---------- 收集模板内的局部变量（v-for / 插槽作用域），它们不算「未暴露」 ----------
const localVars = new Set(['$event', 'item', 'index']);
for (const m of template.matchAll(/v-for="\(?([^"]+?)\)?\s+in\s/g)) {
  m[1].split(',').forEach((v) => localVars.add(v.trim()));
}
for (const m of template.matchAll(/#default="\{([^}]*)\}"/g)) {
  m[1].split(',').forEach((v) => localVars.add(v.trim().split(':')[0].trim()));
}
for (const m of template.matchAll(/#default="([a-zA-Z_$][\w$]*)"/g)) localVars.add(m[1]);

// ---------- 2. 事件处理函数 ----------
const called = new Set();
for (const m of template.matchAll(/@(?:click|keydown|change|input)="([^"]+)"/g)) {
  for (const f of m[1].matchAll(/([a-zA-Z_$][\w$]*)\s*\(/g)) called.add(f[1]);
}
const missingFns = [...called].filter((fn) => !exposed.includes(fn));
report(missingFns.length === 0, '模板调用的方法都已暴露',
  missingFns.length ? '缺少：' + missingFns.join(', ') : `(${[...called].length} 个方法)`);

// ---------- 3. 模板引用的数据变量 ----------
const KEYWORDS = new Set(['true', 'false', 'null', 'undefined', 'in', 'of', 'new', 'typeof',
  'String', 'Number', 'Boolean', 'Object', 'Array', 'Math', 'JSON', 'Date']);
const exprSources = [];
for (const m of template.matchAll(/\{\{([^}]+)\}\}/g)) exprSources.push(m[1]);
for (const m of template.matchAll(/(?:v-if|v-else-if|v-show|v-model|v-loading|:key|:class|:title|:size|:disabled|:data|:model|:style)="([^"]*)"/g)) {
  exprSources.push(m[1]);
}

const referenced = new Set();
for (const expr of exprSources) {
  // 分两步清洗，顺序不能反：
  //   1. 先整段抠掉字符串字面量。否则 'rating-on-up' 会被拆成 rating / on / up，
  //      '#f2f7fe' 会被当成变量名——这类误报会让人不再信任检查结果，比不检查更糟。
  //   2. 再抠掉对象字面量的 key（{ open: true } 里的 open 不是变量）。
  const noStrings = expr
    .replace(/'(?:\\.|[^'\\])*'/g, ' ')
    .replace(/"(?:\\.|[^"\\])*"/g, ' ');
  const cleaned = noStrings.replace(/(?<![\w.$])([a-zA-Z_$][\w$]*)\s*:/g, ' ');

  for (const m of cleaned.matchAll(/(?<![\w.$-])([a-zA-Z_$][\w$]*)/g)) {
    const name = m[1];
    if (KEYWORDS.has(name) || localVars.has(name) || exposed.includes(name)) continue;
    if (/^[A-Z][A-Z_0-9]*$/.test(name)) continue;
    referenced.add(name);
  }
}
report(referenced.size === 0, '模板引用的数据变量都已暴露',
  referenced.size ? '可疑：' + [...referenced].join(', ') : `(检查了 ${exprSources.length} 处表达式)`);

// ---------- 4. 关键：#app 之外不能再有 Vue 模板语法 ----------
// 这是最容易踩的坑：组件写到 #app 结束标签外面，Vue 根本不会编译它，
// 页面上就会原样显示 {{ }} 和 v-if，看起来像「代码漏出来了」。
const afterApp = html.slice(end);
const leaked = [];
for (const m of afterApp.matchAll(/\{\{[^}]*\}\}/g)) leaked.push(m[0].trim());
for (const m of afterApp.matchAll(/\sv-(?:if|for|model|loading)[=\s]/g)) leaked.push(m[0].trim());
report(leaked.length === 0, '#app 之外没有残留模板语法',
  leaked.length ? '漏在 #app 外：' + leaked.slice(0, 5).join(', ') : `(${afterApp.trim().length} 字符尾段)`);

console.log('');
if (failed) {
  console.log(`前端检查：${failed} 项未通过`);
  process.exit(1);
}
console.log('前端检查：全部通过');
