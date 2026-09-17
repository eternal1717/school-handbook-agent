/* vendor 脚本加载诊断（不需要浏览器、不需要 npm 安装）。
 *
 * 用法：node tests/vendor_diag.js
 *
 * 复现 index.html 的脚本加载顺序，检查每个库有没有成功挂到全局变量上，
 * 用来定位「页面出现 {{ }} 原始文本」这类问题——本质是 Vue 没挂载成功。
 *
 * 注意：Element Plus 等库在加载时会访问 document，所以要先塞一个最小的
 * document 桩（stub）进去，否则在 Node 里必然报 "document is not defined"，
 * 那是环境缺的，不是文件坏了。
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const VENDOR = path.join(__dirname, '..', 'static', 'vendor');
// 与 index.html 中 <script> 的出现顺序保持一致
const ORDER = [
  'vue.global.prod.js',
  'element-plus.full.min.js',
  'element-plus-icons.iife.min.js',
  'element-plus-zh-cn.min.js',
  'marked.min.js',
];

// ---------- 最小 DOM 桩 ----------
function makeStubElement() {
  const el = {
    style: {},
    dataset: {},
    classList: { add() {}, remove() {}, contains: () => false },
    children: [],
    childNodes: [],
    firstChild: null,
    nodeType: 1,
    namespaceURI: 'http://www.w3.org/1999/xhtml',
    appendChild(c) { this.children.push(c); return c; },
    removeChild(c) { return c; },
    insertBefore(c) { return c; },
    setAttribute() {},
    removeAttribute() {},
    getAttribute: () => null,
    hasAttribute: () => false,
    querySelector: () => null,
    querySelectorAll: () => [],
    addEventListener() {},
    removeEventListener() {},
    cloneNode() { return makeStubElement(); },
    contains: () => false,
    focus() {},
    blur() {},
  };
  let _html = '';
  Object.defineProperty(el, 'innerHTML', {
    get: () => _html,
    set(v) { _html = String(v); },
  });
  Object.defineProperty(el, 'textContent', {
    get: () => '',
    set(v) { _html = String(v); },
  });
  return el;
}

const head = makeStubElement();
const body = makeStubElement();
const documentStub = {
  head,
  body,
  documentElement: makeStubElement(),
  createElement: () => makeStubElement(),
  createElementNS: () => makeStubElement(),
  createTextNode: (t) => ({ nodeType: 3, textContent: String(t) }),
  createComment: () => ({ nodeType: 8 }),
  querySelector: () => null,
  querySelectorAll: () => [],
  getElementById: () => null,
  getElementsByTagName: () => [],
  addEventListener() {},
  removeEventListener() {},
  activeElement: null,
};

const sandbox = { console, setTimeout, clearTimeout, Date, Math, JSON, Object, Array, Promise, document: documentStub };
sandbox.window = sandbox;
sandbox.self = sandbox;
sandbox.globalThis = sandbox;
sandbox.navigator = { userAgent: 'node' };
vm.createContext(sandbox);

let failed = 0;
for (const file of ORDER) {
  const full = path.join(VENDOR, file);
  if (!fs.existsSync(full)) {
    console.log(`  [FAIL] ${file} 不存在`);
    failed += 1;
    continue;
  }
  try {
    vm.runInContext(fs.readFileSync(full, 'utf8'), sandbox, { filename: file });
    console.log(`  [OK]   ${file}`);
  } catch (error) {
    console.log(`  [FAIL] ${file} -> ${error.message.split('\n')[0]}`);
    failed += 1;
  }
}

const CHECK = [
  ['Vue', 'Vue'],
  ['ElementPlus', 'Element Plus 组件库'],
  ['ElementPlusIconsVue', 'Element Plus 图标'],
  ['ElementPlusLocaleZhCn', 'Element Plus 中文语言包'],
  ['marked', 'Markdown 渲染'],
];
console.log('');
for (const [globalName, label] of CHECK) {
  const ok = typeof sandbox[globalName] !== 'undefined';
  console.log(`  ${ok ? '[OK]  ' : '[FAIL]'} 全局 ${globalName}  ${label}`);
  if (!ok) failed += 1;
}

console.log('');
if (failed) {
  console.log(`vendor 诊断：${failed} 项未通过`);
  process.exit(1);
}
console.log('vendor 诊断：全部通过');
