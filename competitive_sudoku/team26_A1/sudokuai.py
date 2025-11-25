#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import copy
import math
import random
from typing import List, Optional, Set, Tuple
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Advanced Sudoku AI with "Expansion Denial" Strategy.

    Priorities:
    1. Score (Points)
    2. Expansion Denial (Block enemy from accessing new large areas)
    3. Direct Blocking (Take contested squares)
    4. Traps (Pending Points)
    """

    def __init__(self):
        super().__init__()
        self.my_player_id = 0
        self.opponent_id = 0

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Anytime algorithm. Uses Iterative Deepening Minimax.
        """
        self.my_player_id = game_state.current_player
        self.opponent_id = 3 - self.my_player_id

        # 1. Analyze the Board & Frontiers
        my_moves = self.get_legal_moves(game_state, self.my_player_id)
        if not my_moves:
            return

        # Identify Contested Squares (Squares both can reach)
        opp_reach = self.get_reachable_squares(game_state, self.opponent_id)

        # 2. Fast Sort with Expansion Denial Logic
        # We pass the opponent's reach to calculate "what we are denying"
        my_moves.sort(key=lambda m: self.quick_score(game_state, m, opp_reach), reverse=True)

        # Propose the best heuristic move immediately (Safety)
        self.propose_move(my_moves[0])

        # 3. Iterative Deepening Search
        depth = 1
        MAX_DEPTH = 50

        while depth <= MAX_DEPTH:
            try:
                best_move = self.minimax_root(game_state, depth, my_moves, opp_reach)
                if best_move:
                    self.propose_move(best_move)
            except TimeoutError:
                break
            depth += 1

    def quick_score(self, state: GameState, move: Move, opp_reach: Set[Tuple[int, int]]) -> int:
        """
        Heuristic to sort moves.
        Weights: Points >> Denial (Gateway Blocking) >> Direct Blocking >> Centrality
        """
        score = 0
        r, c = move.square

        # --- 1. POINTS (Highest Priority) ---
        pts = self.calculate_points_gained(state.board, r, c)
        score += pts * 20000

        # --- 2. EXPANSION DENIAL (The "2 Blocks Away" Logic) ---
        # Is this square contested? (i.e., can the opponent also take it?)
        if (r, c) in opp_reach:
            # Direct Block Bonus
            score += 1000

            # Expansion Calculation:
            # If opponent took this, how many *new* squares would they unlock?
            # We look at empty neighbors of (r,c) that are NOT currently in opp_reach.
            new_unlocks = 0
            N = state.board.N
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < N and 0 <= nc < N:
                        # If neighbor is empty and opponent couldn't reach it before...
                        if state.board.get((nr, nc)) == SudokuBoard.empty and (nr, nc) not in opp_reach:
                            new_unlocks += 1

            # Massive bonus for denying a "Gateway"
            # If taking this square blocks access to 2+ new squares (like (7,0)), reward heavily.
            score += new_unlocks * 2500

        # --- 3. CENTRALITY / EXPANSION (Tie-breaker) ---
        N = state.board.N
        center = N / 2.0
        dist = abs(r - center) + abs(c - center)
        score -= dist * 10

        return score

    def minimax_root(self, state: GameState, depth: int, moves: List[Move], opp_reach: Set[Tuple[int, int]]) -> \
    Optional[Move]:
        best_val = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf

        for move in moves:
            child_state = self.simulate_move(state, move)
            # Pass opp_reach to heuristic so we remember what we are blocking
            val = self.minimax(child_state, depth - 1, alpha, beta, False, opp_reach)

            if val > best_val:
                best_val = val
                best_move = move

            alpha = max(alpha, best_val)
            if beta <= alpha:
                break
        return best_move

    def minimax(self, state: GameState, depth: int, alpha: float, beta: float, is_maximizing: bool,
                initial_opp_reach: Set[Tuple[int, int]]) -> float:
        if depth == 0:
            return self.evaluate(state, initial_opp_reach)

        current_p = state.current_player
        legal_moves = self.get_legal_moves(state, current_p)

        if not legal_moves:
            return self.evaluate(state, initial_opp_reach)

        if is_maximizing:
            max_eval = -math.inf
            for move in legal_moves:
                child_state = self.simulate_move(state, move)
                eval_val = self.minimax(child_state, depth - 1, alpha, beta, False, initial_opp_reach)
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in legal_moves:
                child_state = self.simulate_move(state, move)
                eval_val = self.minimax(child_state, depth - 1, alpha, beta, True, initial_opp_reach)
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate(self, state: GameState, initial_opp_reach: Set[Tuple[int, int]]) -> float:
        """
        Heuristic Evaluation.
        """
        # --- 1. Score Difference ---
        my_idx = self.my_player_id - 1
        opp_idx = self.opponent_id - 1
        real_score_diff = (state.scores[my_idx] - state.scores[opp_idx])

        # --- 2. Mobility & Denial ---
        my_reach = self.get_reachable_squares(state, self.my_player_id)
        opp_reach_current = self.get_reachable_squares(state, self.opponent_id)

        # Mobility Score: "How many moves do I have vs him?"
        # Heavily penalize opponent moves (Denial Strategy)
        mobility_score = len(my_reach) - (3.0 * len(opp_reach_current))

        # Gateway Protection Score (Heuristic approximation):
        # Check if we successfully reduced the opponent's reach compared to start of turn
        # (This rewards the specific move that cut the graph)
        denial_bonus = len(initial_opp_reach) - len(opp_reach_current)

        # --- 3. Pending Points (Trap) ---
        pending_score = 0
        single_holes = self.find_single_holes(state.board)
        for (r, c) in single_holes:
            can_me = (r, c) in my_reach
            can_opp = (r, c) in opp_reach_current
            pts = self.calculate_points_gained(state.board, r, c)

            if can_opp:
                pending_score -= pts
            elif can_me:
                pending_score += pts

        # --- Weights ---
        W_SCORE = 20000.0
        W_PENDING = 15000.0
        W_MOBILITY = 200.0
        W_DENIAL_BONUS = 500.0  # Reward for every square we removed from their initial reach

        total = (real_score_diff * W_SCORE) + \
                (pending_score * W_PENDING) + \
                (mobility_score * W_MOBILITY) + \
                (denial_bonus * W_DENIAL_BONUS)

        return total

    # --- Helpers ---

    def get_reachable_squares(self, state: GameState, player_id: int) -> Set[Tuple[int, int]]:
        N = state.board.N
        if state.is_classic_game():
            count = sum(1 for i in range(N * N) if state.board.squares[i] == 0)
            return set(range(count))

        occupied = state.occupied_squares1 if player_id == 1 else state.occupied_squares2
        base_allowed = state.allowed_squares1 if player_id == 1 else state.allowed_squares2

        reachable = set()
        if base_allowed:
            for sq in base_allowed:
                if state.board.get(sq) == SudokuBoard.empty:
                    reachable.add(sq)

        for (r, c) in occupied:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < N and 0 <= nc < N:
                        if state.board.get((nr, nc)) == SudokuBoard.empty:
                            reachable.add((nr, nc))
        return reachable

    def get_legal_moves(self, state: GameState, player_id: int) -> List[Move]:
        moves = []
        board = state.board
        N = board.N
        candidates = self.get_reachable_squares(state, player_id)

        for (r, c) in candidates:
            for val in range(1, N + 1):
                if TabooMove((r, c), val) in state.taboo_moves: continue
                if self.is_valid_placement(board, r, c, val):
                    moves.append(Move((r, c), val))
        return moves

    def find_single_holes(self, board: SudokuBoard) -> List[Tuple[int, int]]:
        N = board.N
        m, n = board.m, board.n
        holes = set()

        def check_coords(coords):
            empties = [sq for sq in coords if board.get(sq) == SudokuBoard.empty]
            if len(empties) == 1:
                holes.add(empties[0])

        for r in range(N): check_coords([(r, c) for c in range(N)])
        for c in range(N): check_coords([(r, c) for r in range(N)])
        for br in range(0, N, m):
            for bc in range(0, N, n):
                check_coords([(br + i, bc + j) for i in range(m) for j in range(n)])
        return list(holes)

    def is_valid_placement(self, board: SudokuBoard, row: int, col: int, value: int) -> bool:
        N = board.N
        for c in range(N):
            if board.get((row, c)) == value: return False
        for r in range(N):
            if board.get((r, col)) == value: return False
        m, n = board.m, board.n
        sr, sc = (row // m) * m, (col // n) * n
        for i in range(m):
            for j in range(n):
                if board.get((sr + i, sc + j)) == value: return False
        return True

    def calculate_points_gained(self, board: SudokuBoard, r: int, c: int) -> int:
        regions = 0
        N = board.N
        if sum(1 for i in range(N) if board.get((r, i)) != 0) == N: regions += 1
        if sum(1 for i in range(N) if board.get((i, c)) != 0) == N: regions += 1
        m, n = board.m, board.n
        sr, sc = (r // m) * m, (c // n) * n
        if sum(1 for i in range(m) for j in range(n) if board.get((sr + i, sc + j)) != 0) == m * n: regions += 1
        return {0: 0, 1: 1, 2: 3, 3: 7}.get(regions, 0)

    def simulate_move(self, state: GameState, move: Move) -> GameState:
        new_state = copy.deepcopy(state)
        r, c, val = move.square[0], move.square[1], move.value
        p = new_state.current_player
        new_state.board.put((r, c), val)
        if p == 1:
            new_state.occupied_squares1.append((r, c))
        else:
            new_state.occupied_squares2.append((r, c))
        pts = self.calculate_points_gained(new_state.board, r, c)
        new_state.scores[p - 1] += pts
        new_state.current_player = 3 - p
        return new_state