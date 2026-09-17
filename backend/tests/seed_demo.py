"""演示数据整理：把调试期间攒下的重复会话去重，让「会话列表」看着像真实使用过。

背景
----
开发/评测期间同一条问题会被反复问（比如"本科最长可以读几年？"问了 7 次），
会话列表于是堆了上百条重复项。演示时这一片重复很掉价。

这个脚本不造假数据，只做**去重**：同一个标题只保留最新的一次，
连同它的 message / trace / feedback 一起留下，其余的整条删掉。

同时保留全部"有信息量"的分流样本 —— 越界提问（校园 WiFi 密码）、
注入攻击（忽略你之前的指令）、闲聊（谢谢你）、拒答类问题，
这些恰恰能证明系统有分流和防护能力，删了反而可惜。

用法
----
默认 **只报告不修改**，先看清楚会删什么：

    D:\\python\\python.exe tests\\seed_demo.py

确认无误后加 --apply 真正执行（执行前会自动备份数据库）：

    D:\\python\\python.exe tests\\seed_demo.py --apply

其他参数：

    --keep-dups       不去重，只统计（等价于纯预览）
    --user USER_ID    只处理某个用户的数据（默认全部用户）
    --backup-dir DIR  备份目录（默认 data/backup/）
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# 数据目录按脚本位置反推，避免依赖运行时工作目录
HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parent
DB_PATH = BACKEND_DIR / "data" / "school.db"

# 级联删除时跟着 conversation 一起走的三张表
CHILD_TABLES = ("message", "trace", "feedback")


def connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        print(f"[x] 找不到数据库：{db_path}")
        print("    先启动一次服务（run.py）让它自己建库，或确认路径对不对。")
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def plan_dedup(conn: sqlite3.Connection, user_id: str | None) -> tuple[list[str], list[str]]:
    """算出要保留和要删除的会话 id。

    规则：按 title 分组，每组保留 updated_at（缺省时用 created_at）最新的一条，
    其余全删。标题为空的会话单独一组，只保留最新一条。
    """
    sql = "select id, title, coalesce(updated_at, created_at) as ts from conversation"
    params: tuple = ()
    if user_id:
        sql += " where user_id = ?"
        params = (user_id,)
    sql += " order by ts desc"

    groups: dict[str, list[str]] = {}
    for row in conn.execute(sql, params):
        # 标题里可能有多余空格，归一化后再分组，避免"标题A "和"标题A"被当成两条
        title = (row["title"] or "").strip()
        groups.setdefault(title, []).append(row["id"])

    keep: list[str] = []
    drop: list[str] = []
    for _title, ids in groups.items():
        keep.append(ids[0])       # order by ts desc，第一条就是最新的
        drop.extend(ids[1:])
    return keep, drop


def count_children(conn: sqlite3.Connection, conv_ids: list[str]) -> dict[str, int]:
    """统计这些会话名下各有多少条子记录，好让用户知道删除的影响面。"""
    result: dict[str, int] = {}
    if not conv_ids:
        return result
    # sqlite 变量上限是 999，分批查避免超限
    for table in CHILD_TABLES:
        total = 0
        for i in range(0, len(conv_ids), 900):
            batch = conv_ids[i:i + 900]
            marks = ",".join("?" * len(batch))
            total += conn.execute(
                f"select count(*) from {table} where conversation_id in ({marks})", batch
            ).fetchone()[0]
        result[table] = total
    return result


def backup_db(db_path: Path, backup_dir: Path) -> Path:
    """删除前先把整个数据库复制一份，出问题能退回去。"""
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = backup_dir / f"school-{stamp}.db"
    shutil.copy2(db_path, dest)
    return dest


def apply_dedup(conn: sqlite3.Connection, drop: list[str]) -> dict[str, int]:
    """真正执行删除，返回各表删掉的条数。"""
    deleted: dict[str, int] = {}
    for i in range(0, len(drop), 900):
        batch = drop[i:i + 900]
        marks = ",".join("?" * len(batch))
        for table in CHILD_TABLES:
            cur = conn.execute(
                f"delete from {table} where conversation_id in ({marks})", batch
            )
            deleted[table] = deleted.get(table, 0) + cur.rowcount
        cur = conn.execute(
            f"delete from conversation where id in ({marks})", batch
        )
        deleted["conversation"] = deleted.get("conversation", 0) + cur.rowcount
    conn.commit()
    return deleted


# 带 user_id 归属字段的表（message 没有该字段，靠 conversation_id 关联）
OWNER_TABLES = ("conversation", "trace", "feedback", "long_term_memory")


def plan_unify(conn: sqlite3.Connection, target: str) -> dict[str, dict[str, int]]:
    """统计「统一归属」会改动多少行，按原用户分组，方便一眼看清散落情况。

    调试期不同脚本用了各自的 user_id（eval_user / verify_user / probe2 …），
    结果前端默认的 default_user 反而几乎看不到数据。这一步就是把它们收拢起来。
    """
    report: dict[str, dict[str, int]] = {}
    for table in OWNER_TABLES:
        try:
            rows = conn.execute(
                f"select user_id, count(*) from {table} where user_id <> ? group by user_id",
                (target,),
            ).fetchall()
        except sqlite3.OperationalError:
            continue  # 表不存在（比如老库还没建 long_term_memory）就跳过
        for user_id, n in rows:
            report.setdefault(user_id, {})[table] = n
    return report


def apply_unify(conn: sqlite3.Connection, target: str) -> dict[str, int]:
    """把归属真正改成 target，返回各表改动行数。"""
    moved: dict[str, int] = {}
    for table in OWNER_TABLES:
        try:
            cur = conn.execute(
                f"update {table} set user_id = ? where user_id <> ?", (target, target)
            )
        except sqlite3.OperationalError:
            continue
        moved[table] = cur.rowcount
    conn.commit()
    return moved


def show_kept(conn: sqlite3.Connection, keep: list[str], limit: int = 25) -> None:
    """把保留的会话列出来，让用户确认'留下来的确实是我想要的'。"""
    if not keep:
        return
    marks = ",".join("?" * len(keep))
    rows = conn.execute(
        f"select id, title, coalesce(updated_at, created_at) as ts "
        f"from conversation where id in ({marks}) order by ts desc",
        keep,
    ).fetchall()
    print(f"\n保留的 {len(rows)} 条会话（按时间倒序）：")
    for row in rows[:limit]:
        title = (row["title"] or "(无标题)").replace("\n", " ")[:36]
        print(f"  {row['ts'][:19]}  {title}")
    if len(rows) > limit:
        print(f"  ... 还有 {len(rows) - limit} 条")


def main() -> int:
    parser = argparse.ArgumentParser(description="把调试期间的重复会话去重，整理成演示状态")
    parser.add_argument("--apply", action="store_true", help="真正执行删除（默认只报告）")
    parser.add_argument("--keep-dups", action="store_true", help="不去重，只统计现状")
    parser.add_argument("--user", metavar="USER_ID", help="只处理某个用户的数据")
    parser.add_argument(
        "--unify-user",
        metavar="USER_ID",
        help="把所有会话/trace/feedback 的归属统一成这个用户（修掉调试期多用户散落的问题）",
    )
    parser.add_argument("--backup-dir", metavar="DIR", help="备份目录（默认 data/backup/）")
    args = parser.parse_args()

    conn = connect(DB_PATH)
    total_conv = conn.execute("select count(*) from conversation").fetchone()[0]
    total_msg = conn.execute("select count(*) from message").fetchone()[0]

    print("=" * 56)
    print("演示数据整理 · 会话去重")
    print("=" * 56)
    print(f"数据库    : {DB_PATH}")
    print(f"当前规模  : {total_conv} 个会话 / {total_msg} 条消息")

    if args.keep_dups:
        print("\n--keep-dups：只统计，不做任何修改。")
        for row in conn.execute(
            "select title, count(*) n from conversation group by title "
            "having n > 1 order by n desc limit 15"
        ):
            print(f"  {row['n']:3d} 次  {(row['title'] or '')[:40]}")
        conn.close()
        return 0

    keep, drop = plan_dedup(conn, args.user)
    children = count_children(conn, drop)
    unify_plan = plan_unify(conn, args.unify_user) if args.unify_user else {}

    print(f"\n去重结果  : 保留 {len(keep)} 条，删除 {len(drop)} 条重复")
    if children:
        print("连带删除  : " + "，".join(f"{k} {v} 条" for k, v in children.items()))
    show_kept(conn, keep)

    if unify_plan:
        print(f"\n归属统一到「{args.unify_user}」：")
        for old_user, tables in sorted(unify_plan.items(), key=lambda kv: -sum(kv[1].values())):
            detail = "，".join(f"{t} {n} 条" for t, n in tables.items())
            print(f"  {old_user!r:16s} → {detail}")

    if not drop and not unify_plan:
        print("\n数据库已经是干净的演示状态，无需改动。")
        conn.close()
        return 0

    if not args.apply:
        print("\n" + "-" * 56)
        print("这是预览模式，**什么都没有改**。")
        print("确认上面的清单没问题后，加 --apply 真正执行：")
        print("    D:\\python\\python.exe tests\\seed_demo.py --apply --unify-user default_user")
        conn.close()
        return 0

    # ---- 以下才是真正的写操作 ----
    backup_dir = Path(args.backup_dir) if args.backup_dir else BACKEND_DIR / "data" / "backup"
    dest = backup_db(DB_PATH, backup_dir)
    print(f"\n已备份    : {dest}")

    if drop:
        deleted = apply_dedup(conn, drop)
        print("已删除    : " + "，".join(f"{k} {v} 条" for k, v in deleted.items()))

    if args.unify_user:
        moved = apply_unify(conn, args.unify_user)
        detail = "，".join(f"{k} {v} 条" for k, v in moved.items() if v)
        print(f"已改归属  : {detail or '无'} → {args.unify_user}")

    left_conv = conn.execute("select count(*) from conversation").fetchone()[0]
    left_msg = conn.execute("select count(*) from message").fetchone()[0]
    print(f"整理后    : {left_conv} 个会话 / {left_msg} 条消息")

    print("\n当前各用户会话数：")
    for row in conn.execute(
        "select user_id, count(*) from conversation group by user_id order by 2 desc"
    ):
        print(f"  {row[0]:16s} {row[1]}")

    print("\n[OK] 完成。刷新浏览器（默认用户 default_user）就能看到完整演示数据。")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
