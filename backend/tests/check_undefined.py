"""静态检查：找出「被调用但从未定义」的全局名字。

## 为什么需要这个脚本

本项目的源文件曾被编辑器/同步进程抢占（Windows 上表现为写入被丢弃、
文件回滚到旧版本），结果就是**函数体里还留着调用，函数定义却没了**。
这类故障最坑的地方在于：Python 在 import 阶段完全不报错，直到那次调用
真的执行到那一行才抛 `NameError`——线上表现为「服务正常启动，一问就崩」。

所以这里用一个不依赖任何第三方库的办法提前兜住：
用标准库 `symtable` 把每个函数的自由变量（free/global 名）挖出来，
和「模块级已绑定的名字 + 内置名」对账，对不上的就是孤儿调用。

用法：
    python tests/check_undefined.py          # 只报错，退出码非 0 表示有问题
    python tests/check_undefined.py -v        # 连每个文件扫描到的名字一起打印
"""
from __future__ import annotations

import builtins
import sys
import symtable
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
VERBOSE = "-v" in sys.argv

# 这些名字由运行环境注入，源码里看不到定义
RUNTIME_PROVIDED = {"__name__", "__file__", "__doc__", "__package__", "__builtins__"}


def _bound_names(table: symtable.SymbolTable) -> set[str]:
    """模块/类作用域里「被绑定过」的名字：赋值、导入、def、class 都算。"""
    names = set()
    for sym in table.get_symbols():
        if sym.is_assigned() or sym.is_imported() or sym.is_namespace():
            names.add(sym.get_name())
    return names


def _scan(table: symtable.SymbolTable, module_names: set[str], known: set[str],
          out: list[tuple[str, int, str]]) -> None:
    """递归下探所有作用域，收集「读取了模块级名字」的引用。

    `known` 是「在外层作用域里已经绑定的名字」的累积集合——嵌套函数能闭包
    引用外层局部变量，那些名字不属于模块级，但也不是孤儿，必须排除。
    """
    for sym in table.get_symbols():
        name = sym.get_name()
        if name in known or name in RUNTIME_PROVIDED:
            continue
        # is_global() 为真 = 这个作用域里没绑定它，得往模块级/内置找
        if sym.is_global() and not sym.is_assigned() and not sym.is_imported():
            if name not in module_names and not hasattr(builtins, name):
                out.append((table.get_name(), table.get_lineno(), name))

    for child in table.get_children():
        # 子作用域能看到的：模块级全部 + 本作用域绑定的
        child_known = known | _bound_names(table)
        _scan(child, module_names, child_known, out)


def check_file(path: Path) -> list[tuple[str, int, str]]:
    source = path.read_text(encoding="utf-8")
    try:
        root = symtable.symtable(source, str(path), "exec")
    except SyntaxError as exc:
        return [(f"<语法错误>", exc.lineno or 0, str(exc.msg))]

    module_names = _bound_names(root)
    found: list[tuple[str, int, str]] = []
    for child in root.get_children():
        _scan(child, module_names, module_names, found)
    return found


def main() -> int:
    files = sorted(APP_DIR.rglob("*.py"))
    if not files:
        print(f"没找到任何 .py：{APP_DIR}")
        return 1

    total_problems = 0
    for path in files:
        problems = check_file(path)
        rel = path.relative_to(APP_DIR.parent)
        if problems:
            total_problems += len(problems)
            print(f"\n[!] {rel}")
            for scope, lineno, name in problems:
                print(f"    line {lineno:<5} {scope}() 用到未定义的名字: {name}")
        elif VERBOSE:
            print(f"[ok] {rel}")

    print()
    if total_problems:
        print(f"发现 {total_problems} 处未定义引用——这些就是「一问就崩」的定时炸弹。")
        return 1
    print(f"检查通过：{len(files)} 个文件，未发现未定义引用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
