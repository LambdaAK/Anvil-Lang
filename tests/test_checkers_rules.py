"""The rules in examples/checkers.anvil agree with an independent Python implementation of
American checkers on every position of a few hundred random games."""
import os
import re

from util import ROOT, run_native

DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]     # up-left, up-right, down-left, down-right


def rc(t):
    r = t // 4
    return r, 2 * (t % 4) + 1 - r % 2


def sq(r, c):
    return 4 * r + c // 2 if 0 <= r < 8 and 0 <= c < 8 else None


def jumps_from(b, s):
    out = []
    r, c = rc(s)
    for d, (dy, dx) in enumerate(DIRS):
        if b[s] <= 0 or (b[s] == 1 and dy > 0):
            continue
        over, to = sq(r + dy, c + dx), sq(r + 2 * dy, c + 2 * dx)
        if to is not None and b[over] < 0 and b[to] == 0:
            out.append((over, to))
    return out


def turned(a):
    return [-a[31 - t] for t in range(32)]


def successors(b, forced):
    """Every legal step as (board for the next mover, its forced piece, same player again)."""
    pieces = [forced] if forced >= 0 else [s for s in range(32) if b[s] > 0]
    jumps = [(s, over, to) for s in pieces for over, to in jumps_from(b, s)]
    out = []
    for s, over, to in jumps:
        a = list(b)
        crowned = a[s] == 1 and to < 4
        a[to], a[s], a[over] = (2 if a[s] == 2 or to < 4 else 1), 0, 0
        if not crowned and jumps_from(a, to):
            out.append((a, to, 1))
        else:
            out.append((turned(a), -1, 0))
    if jumps:
        return out
    for s in pieces:
        r, c = rc(s)
        for dy, dx in DIRS:
            n = sq(r + dy, c + dx)
            if b[s] > 0 and not (b[s] == 1 and dy > 0) and n is not None and b[n] == 0:
                a = list(b)
                a[n], a[s] = (2 if a[s] == 2 or n < 4 else 1), 0
                out.append((turned(a), -1, 0))
    return out


DUMP = '''
fn random_player(valid):
    u: [MOVES] ~ uniform(0, 1)
    return where(valid, u, -inf)

for g in range(GAMES_TO_CHECK):
    b = START
    forced = -1
    ply = 0
    while true:
        valid, next, next_forced, again = moves(b, forced)
        print("P", b, forced)
        for k in range(MOVES):
            if valid[k] > 0:
                print("N", next[k], next_forced[k], i32(again[k]))
        if sum(valid) == 0 or ply == 200:
            print("E")
            break
        m = argmax(random_player(valid))
        print("C", m)
        b = next[m]
        forced = next_forced[m]
        ply = ply + 1
'''


def nums(s):
    return [int(float(x)) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]


def test_rules_match_reference():
    path = os.path.join(ROOT, "examples", "checkers.anvil")
    rules = open(path).read().split("# " + "-" * 66 + " the player")[0]
    out, err, code, _ = run_native(rules + DUMP.replace("GAMES_TO_CHECK", "150"), path=path)
    assert code == 0, err
    positions = continuations = kings = 0
    expect = None
    for blk in re.split(r"^(?=[PNCE] ?)", out, flags=re.M):
        if not blk.strip():
            continue
        tag, v = blk[0], nums(blk[1:])
        if tag == "P":
            b, forced = v[:32], v[32]
            assert expect is None or (b, forced) == expect, "the chosen move led somewhere else"
            want = sorted((tuple(a), f, again) for a, f, again in successors(b, forced))
            got, expect = [], None
            positions += 1
            kings += any(abs(x) == 2 for x in b)
        elif tag == "N":
            got.append((tuple(v[:32]), v[32], v[33]))
        elif tag == "C":
            assert sorted(got) == want, f"moves differ in position {b} (forced {forced})"
            a, f, again = got[v[0]]
            expect = (list(a), f)
            continuations += again
        else:                                           # the game is over (or long enough)
            assert sorted(got) == want, f"moves differ in position {b} (forced {forced})"
            expect = None
    assert positions > 10000 and continuations > 100 and kings > 1000, (positions, continuations, kings)
