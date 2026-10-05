"""Minesweeper environment with text rendering, verifiable rewards, and an expert policy.

Boards, rendering, and prompt match CAST (github.com/Wloner0809/CAST) so
results are comparable to its published table. The board is rendered with
right-justified column headers and 3-character cells:
  '.'  hidden
  '0'-'8' revealed, digit = adjacent mines
  '*'  mine (only shown for debugging / terminal states)
Coordinates are (row, col), 0-indexed.
"""
import random
from math import comb

# CAST draws its train and eval seeds from [0, 1_000_000). Training boards use
# seeds above that pool so a training board can never be an evaluation board.
TRAIN_SEED_START = 1_000_000


def generate_layout(seed, rows, cols, num_mines):
    """CAST's deterministic layout: random first click, 3x3 safe zone around it,
    then mines by rejection sampling from the same RNG stream. Returns
    (mine positions, first click). The RNG draw order must not change: it is what
    makes a seed reproduce the same board as the reference implementation."""
    rng = random.Random(seed)
    fc_r, fc_c = rng.randint(0, rows - 1), rng.randint(0, cols - 1)
    safe_3x3 = {(fc_r + dr, fc_c + dc)
                for dr in range(-1, 2) for dc in range(-1, 2)
                if 0 <= fc_r + dr < rows and 0 <= fc_c + dc < cols}
    total_cells = rows * cols
    safe_zone = safe_3x3
    if total_cells - len(safe_3x3) < num_mines:
        safe_zone = {(fc_r, fc_c)}
        num_mines = min(num_mines, total_cells - 1)
    mines = set()
    while len(mines) < num_mines:
        r, c = rng.randint(0, rows - 1), rng.randint(0, cols - 1)
        if (r, c) in mines or (r, c) in safe_zone:
            continue
        mines.add((r, c))
    return sorted(mines), (fc_r, fc_c)


class Minesweeper:
    def __init__(self, width=6, height=6, n_mines=7, seed=None,
                 mine_positions=None, first_click=None):
        self.w, self.h, self.n_mines = width, height, n_mines
        if mine_positions is None:
            mine_positions, first_click = generate_layout(
                seed if seed is not None else random.randrange(1 << 30),
                height, width, n_mines)
        self.mines = {(r, c) for r, c in mine_positions}
        self.revealed = set()
        # CAST auto-reveals the first click with flood fill, so states are
        # mid-game positions rather than a fully hidden board; a first click that
        # clears every safe cell leaves the board solved.
        self._flood(*first_click)

    def _clone(self, revealed=None):
        """Copy of this board: same mines, optionally a different revealed set."""
        clone = Minesweeper.__new__(Minesweeper)
        clone.w, clone.h, clone.n_mines = self.w, self.h, self.n_mines
        clone.mines = self.mines
        clone.revealed = set(self.revealed if revealed is None else revealed)
        return clone

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
        """CAST's layout: right-justified column header, 3-character cells."""
        lines = ["   " + " ".join(str(c).rjust(2) for c in range(self.w))]
        for r in range(self.h):
            row = f"{r:2} "
            for c in range(self.w):
                row += (f" {self.adjacent_mines(r, c)} "
                        if (r, c) in self.revealed else " . ")
            lines.append(row)
        return "\n".join(lines)

    def prompt(self):
        # No mine count: CAST never tells the agent how many mines are left.
        return (f"Minesweeper {self.w}x{self.h} grid with hidden mines. "
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
    n = board._clone()._flood(r, c)
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
        gain = board._clone()._flood(r, c)
        if gain > best_gain:
            best, best_gain = (r, c), gain
    return best


def mine_posterior(board):
    """Return exact mine probabilities for every hidden cell.

    Hidden layouts are weighted uniformly subject to the revealed clues and
    total mine count. Frontier assignments are enumerated with constraint
    bounds; unconstrained cells are accounted for with binomial weights.
    """
    hidden = board.hidden_cells()
    if not hidden:
        return {}
    hidden_set = set(hidden)
    clues = []
    frontier = set()
    for r, c in board.revealed:
        adjacent = [p for p in board._neighbors(r, c) if p in hidden_set]
        if adjacent:
            clues.append((adjacent, board.adjacent_mines(r, c)))
            frontier.update(adjacent)

    frontier = sorted(frontier)
    interior = sorted(hidden_set - set(frontier))
    f_index = {cell: i for i, cell in enumerate(frontier)}
    constraints = [(tuple(f_index[p] for p in cells), required)
                   for cells, required in clues]
    touching = [[] for _ in frontier]
    for ci, (indices, _) in enumerate(constraints):
        for i in indices:
            touching[i].append(ci)
    assigned = [-1] * len(frontier)
    sums = [0] * len(constraints)
    remaining = [len(indices) for indices, _ in constraints]
    ways_by_k = [0] * (len(frontier) + 1)
    mine_ways_by_cell_k = [[0] * (len(frontier) + 1) for _ in frontier]

    def visit(i, mines):
        if i == len(frontier):
            ways_by_k[mines] += 1
            for j, value in enumerate(assigned):
                if value:
                    mine_ways_by_cell_k[j][mines] += 1
            return
        for value in (0, 1):
            valid = True
            for ci in touching[i]:
                sums[ci] += value
                remaining[ci] -= 1
                target = constraints[ci][1]
                if sums[ci] > target or sums[ci] + remaining[ci] < target:
                    valid = False
            assigned[i] = value
            if valid:
                visit(i + 1, mines + value)
            for ci in touching[i]:
                sums[ci] -= value
                remaining[ci] += 1

    visit(0, 0)
    n_interior = len(interior)
    total_layouts = 0
    weighted_frontier = [0] * len(frontier)
    weighted_interior = 0
    for k, count in enumerate(ways_by_k):
        n_remaining = board.n_mines - k
        if count == 0 or not 0 <= n_remaining <= n_interior:
            continue
        weight = comb(n_interior, n_remaining)
        total_layouts += count * weight
        weighted_interior += count * comb(n_interior - 1, n_remaining - 1) if n_interior and n_remaining else 0
        for j in range(len(frontier)):
            weighted_frontier[j] += mine_ways_by_cell_k[j][k] * weight
    if total_layouts == 0:
        raise ValueError("board clues are inconsistent with the configured mine count")
    probabilities = {cell: weighted_frontier[i] / total_layouts
                     for i, cell in enumerate(frontier)}
    if interior:
        p = weighted_interior / total_layouts
        probabilities.update({cell: p for cell in interior})
    return probabilities


def posterior_move(board, rng=None):
    """Pick a minimum-risk hidden cell, breaking ties in row-major order.

    Passing rng breaks ties at random instead, which keeps generated training
    states from all funneling through the same tie-choice cells."""
    posterior = mine_posterior(board)
    if not posterior:
        return None
    p_min = min(posterior.values())
    candidates = sorted(cell for cell, p in posterior.items()
                        if abs(p - p_min) < 1e-12)
    # Exact expected flood-fill gain would enumerate hundreds of thousands of
    # full layouts on common boards; keep target selection inexpensive.
    return candidates[0] if rng is None else rng.choice(candidates)


def sample_state(width=6, height=6, n_mines=7, seed=None, rng=None):
    """A training position drawn uniformly from the states a game passes through.

    Starts from a CAST-style layout whose first click CAST already revealed and
    plays the posterior solver (random tie-break) until it would lose, keeping
    every position along the way. Sampling the whole game instead of only the
    opening is the point: evaluation plays full games, and the late positions
    with no provable safe cell are the ones a policy actually fails on. Unsolved
    states only, so the returned board always has a hidden cell to open.
    """
    if rng is None:
        rng = random.Random(seed)
    while True:
        board = Minesweeper(width, height, n_mines,
                            seed=rng.randrange(TRAIN_SEED_START, 1 << 30))
        states = [] if board.solved() else [set(board.revealed)]
        while not board.solved():
            move = posterior_move(board, rng)
            if move is None or board.is_mine(*move):
                break
            board.click(*move)
            if not board.solved():
                states.append(set(board.revealed))
        # An empty list means the first click cleared the board: draw another.
        if states:
            return board._clone(rng.choice(states))


def posterior_reward(board, r, c, mode="posterior", posterior=None):
    """Score a move from the visible state, without consulting hidden mines."""
    if not (0 <= r < board.h and 0 <= c < board.w):
        return -1.0
    if (r, c) in board.revealed:
        return -0.5
    posterior = mine_posterior(board) if posterior is None else posterior
    p = posterior[(r, c)]
    p_min = min(posterior.values())
    best = float(abs(p - p_min) < 1e-12)
    return best if mode == "vpr" else best - p


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
