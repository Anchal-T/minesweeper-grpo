"""Minesweeper environment with text rendering, verifiable rewards, and an expert policy.

Board is rendered row-major; cells are separated by spaces:
  '.'  hidden
  '0'-'8' revealed, digit = adjacent mines
  '*'  mine (only shown for debugging / terminal states)
Coordinates are (row, col), 0-indexed.
"""
import random


class Minesweeper:
    def __init__(self, width=6, height=6, n_mines=6, seed=None):
        self.w, self.h, self.n_mines = width, height, n_mines
        rng = random.Random(seed)
        cells = [(r, c) for r in range(height) for c in range(width)]
        self.mines = set(rng.sample(cells, n_mines))
        self.revealed = set()
        # seed the game with one random safe cell + flood fill, so states are
        # mid-game positions rather than a fully hidden board
        safe = [c for c in cells if c not in self.mines]
        self._flood(*rng.choice(safe))

    def _neighbors(self, r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.h and 0 <= nc < self.w:
                    yield nr, nc

    def adjacent_mines(self, r, c):
        return sum((nr, nc) in self.mines for nr, nc in self._neighbors(r, c))

    def _flood(self, r, c):
        """Reveal (r,c); if it's a 0-cell, cascade like real minesweeper."""
        if (r, c) in self.revealed:
            return 0
        stack, gained = [(r, c)], 0
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in self.revealed or (cr, cc) in self.mines:
                continue
            self.revealed.add((cr, cc))
            gained += 1
            if self.adjacent_mines(cr, cc) == 0:
                stack.extend(self._neighbors(cr, cc))
        return gained

    def is_mine(self, r, c):
        return (r, c) in self.mines

    def click(self, r, c):
        """Returns (n_revealed, hit_mine). Illegal/off-board clicks count as a miss."""
        if not (0 <= r < self.h and 0 <= c < self.w):
            return 0, True
        if (r, c) in self.revealed:
            return 0, False  # no-op, wastes the turn
        if self.is_mine(r, c):
            return 0, True
        return self._flood(r, c), False

    def solved(self):
        n_safe = self.w * self.h - self.n_mines
        return len(self.revealed) >= n_safe

    def hidden_cells(self):
        return [(r, c) for r in range(self.h) for c in range(self.w)
                if (r, c) not in self.revealed]

    def render(self):
        rows = []
        for r in range(self.h):
            row = []
            for c in range(self.w):
                if (r, c) in self.revealed:
                    row.append(str(self.adjacent_mines(r, c)))
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)

    def prompt(self):
        return (f"Minesweeper {self.w}x{self.h} grid, {self.n_mines} mines. "
                f"'.' = hidden, digit = adjacent mines.\n"
                f"{self.render()}\n"
                f"Which hidden cell is safe? Answer with row,col (0-indexed). "
                f"Answer:")


def step_reward(board, r, c):
    """Dense verifiable reward for one move (non-mutating: the flood-fill gain
    is measured on a probe copy so all group rollouts score the same state).

    mine / illegal / off-board : -1.0
    safe click                 : 0.5 + 0.5 * (cells revealed / cells hidden before)
    clicking an already-open cell wastes the turn: -0.5
    """
    if not (0 <= r < board.h and 0 <= c < board.w):
        return -1.0, True
    if (r, c) in board.revealed:
        return -0.5, False
    if board.is_mine(r, c):
        return -1.0, True
    hidden_before = len(board.hidden_cells())
    probe = Minesweeper.__new__(Minesweeper)
    probe.w, probe.h, probe.n_mines = board.w, board.h, board.n_mines
    probe.mines = board.mines
    probe.revealed = set(board.revealed)
    n = probe._flood(r, c)
    return 0.5 + 0.5 * (n / hidden_before), False


def expert_move(board, rng=None):
    """Ground-truth expert: among safe hidden cells, pick the one whose flood
    fill reveals the most. Ties broken deterministically (first in row-major
    order) so SFT targets are consistent — the model must actually read the
    board instead of averaging over arbitrary tie choices."""
    best, best_gain = None, -1
    for (r, c) in board.hidden_cells():
        if board.is_mine(r, c):
            continue
        probe = Minesweeper.__new__(Minesweeper)
        probe.w, probe.h, probe.n_mines = board.w, board.h, board.n_mines
        probe.mines = board.mines
        probe.revealed = set(board.revealed)
        gain = probe._flood(r, c)
        if gain > best_gain:
            best, best_gain = (r, c), gain
    return best


def parse_move(text):
    """Parse the first 'row,col' pair out of a completion; returns (r, c) or None.

    Completions may repeat pairs ('2,1,1,4,') with no whitespace, so use a
    regex instead of token splitting."""
    import re
    m = re.search(r"(\d+)\s*,\s*(\d+)", text)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def play_episode(board, move_fn, max_moves=None):
    """Play a full game with greedy move_fn(prompt) -> text. Returns stats."""
    moves, hit_mine = 0, False
    max_moves = max_moves or board.w * board.h
    while not board.solved() and moves < max_moves:
        mv = parse_move(move_fn(board.prompt()))
        if mv is None:
            hit_mine = True  # unparseable = garbage move, treat as loss
            break
        _, mine = board.click(*mv)
        moves += 1
        if mine:
            hit_mine = True
            break
    return {"solved": board.solved() and not hit_mine,
            "cleared": len(board.revealed), "moves": moves,
            "total_safe": board.w * board.h - board.n_mines}
