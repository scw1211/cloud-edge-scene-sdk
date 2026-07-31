#!/usr/bin/env python3
"""无需第三方依赖，采样统计一个命令及其所有子进程的 RSS 峰值。"""
import argparse
import json
import os
import subprocess
import time


def children(pid):
    # Linux /proc 暴露当前线程组的子进程列表，用它递归得到整棵进程树。
    try:
        with open("/proc/{}/task/{}/children".format(pid, pid)) as handle:
            return [int(value) for value in handle.read().split()]
    except IOError:
        return []


def rss_kb(pid):
    # VmRSS 是进程当前常驻内存；进程退出的瞬间文件可能消失，所以 IOError 视为 0。
    try:
        with open("/proc/{}/status".format(pid)) as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except IOError:
        pass
    return 0


def tree_pids(pid):
    # 广度/深度均可，这里用 pending 栈遍历所有仍存活的后代进程。
    result, pending = set(), [pid]
    while pending:
        current = pending.pop()
        if current in result:
            continue
        result.add(current)
        pending.extend(children(current))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0.02)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command or args.command[0] != "--" or len(args.command) == 1:
        parser.error("usage: measure_rss.py -- command [args...]")
    process = subprocess.Popen(args.command[1:])
    peak_kb, peak_pids = 0, []
    while process.poll() is None:
        # 按 interval 周期采样总 RSS，记录峰值和当时参与统计的 pid 列表。
        pids = tree_pids(process.pid)
        current = sum(rss_kb(pid) for pid in pids)
        if current > peak_kb:
            peak_kb, peak_pids = current, sorted(pids)
        time.sleep(args.interval)
    print(json.dumps({"exit_code": process.returncode, "peak_rss_kb": peak_kb, "peak_rss_mb": round(peak_kb / 1024.0, 3), "peak_pids": peak_pids}, sort_keys=True))
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
