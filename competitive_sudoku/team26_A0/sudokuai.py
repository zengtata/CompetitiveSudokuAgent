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
    Advanced Sudoku AI with Blocking, Trapping, and Expansion strategies.
    Uses Iterative Deepening Minimax with Alpha-Beta Pruning.
    """

    def __init__(self):
        super().__init__()
        self.my_player_id = 0
        self.opponent_id = 0

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Anytime algorithm. Iteratively deepens the search from depth 1 upwards
        until the time runs out (thread killed).
        """
        self.my_player_id = game_state.current_player
        self.opponent_id = 3 - self.my_player_id  # 1->2, 2->1

        # 1. Get immediate legal moves
        possible_moves = self.get_legal_moves(game_state, self.my_player_id)

        if not possible_moves:
            return

        # 2. Fast Fail-Safe: Propose a random move immediately so we don't timeout
        # Sort moves by a quick heuristic (capture center/points) to get a decent baseline
        possible_moves.sort(key=lambda m: self.quick_score(game_state, m), reverse=True)
        self.propose_move(possible_moves[0])

        # 3. Iterative Deepening Search
        depth = 1
        MAX_DEPTH = 50

        while depth <= MAX_DEPTH:
            try:
                best_move = self.minimax_root(game_state, depth, possible_moves)
                if best_move:
                    self.propose_move(best_move)
            except TimeoutError:
                break  # Should be handled by external killer, but good practice

            depth += 1

    def quick_score(self, state: GameState, move: Move) -> int:
        """Simple heuristic for initial move sorting: Points > Center > Random"""
        # Points
        pts = self.calculate_points_gained(state.board, move.square[0], move.square[1])
        score = pts * 100

        # Centrality (Encourage moving to middle)
        N = state.board.N
        center = N / 2
        dist = abs(move.square[0] - center) + abs(move.square[1] - center)
        score -= dist

        return score

    def minimax_root(self, state: GameState, depth: int, moves: List[Move]) -> Optional[Move]:
        best_val = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf

        for move in moves:
            child_state = self.simulate_move(state, move)
            # We passed move, so it's now opponent's turn (minimizing)
            val = self.minimax(child_state, depth - 1, alpha, beta, False)

            if val > best_val:
                best_val = val
                best_move = move

            alpha = max(alpha, best_val)
            if beta <= alpha:
                break

        return best_move

    def minimax(self, state: GameState, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        if depth == 0:
            return self.evaluate(state)

        # Determine whose turn it is in this state
        current_p = state.current_player
        legal_moves = self.get_legal_moves(state, current_p)

        if not legal_moves:
            # Game might not be over, but player must pass.
            # In this implementation, we just evaluate leaf if pass.
            return self.evaluate(state)

        if is_maximizing:
            max_eval = -math.inf
            for move in legal_moves:
                child_state = self.simulate_move(state, move)
                eval_val = self.minimax(child_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in legal_moves:
                child_state = self.simulate_move(state, move)
                eval_val = self.minimax(child_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate(self, state: GameState) -> float:
        """
        Complex Evaluation Function:
        1. Score Diff (High Priority)
        2. Pending Points (The Trap Strategy)
        3. Mobility/Blocking (Sealing Strategy)
        4. Expansion (Map Exploration)
        """

        # --- 1. Score Difference ---
        # If I am player 1, score[0] is mine.
        my_idx = self.my_player_id - 1
        opp_idx = self.opponent_id - 1

        real_score_diff = (state.scores[my_idx] - state.scores[opp_idx])

        # --- 2. Pending Points / Trap Logic ---
        # Scan for regions with exactly 1 empty square.
        # If opponent can reach it -> They WILL take it (Penalty for me)
        # If only I can reach it -> I WILL take it (Bonus for me)

        # We need the reachable squares for both players in this state
        my_reach = self.get_reachable_squares(state, self.my_player_id)
        opp_reach = self.get_reachable_squares(state, self.opponent_id)

        pending_score = 0

        # Helper to check regions (Row, Col, Block)
        N = state.board.N
        m, n = state.board.m, state.board.n

        # Check Rows, Cols, Blocks for single empty cells
        # (Optimization: This can be costly, do it efficiently)
        single_holes = self.find_single_holes(state.board)

        for (r, c) in single_holes:
            # Who can play here?
            can_me = (r, c) in my_reach
            can_opp = (r, c) in opp_reach

            # How many points is this hole worth? (1, 3, or 7?)
            # We simulate filling it to count completed regions
            pts = self.calculate_points_gained(state.board, r, c)

            if can_opp:
                # Opponent can take this. Bad for me.
                # "Wait" strategy: If I left this hole open, I get penalized here.
                pending_score -= pts
            elif can_me:
                # Only I can take it. Good. "Banked" points.
                pending_score += pts

        # --- 3. Mobility & Blocking (The "Seal" Logic) ---
        # Maximizing (My Moves - Opp Moves) blocks them and frees me.
        mobility_diff = len(my_reach) - len(opp_reach)

        # --- 4. Expansion / Centrality ---
        # Encourage moving towards the opponent's starting side
        expansion_score = 0
        center = N / 2.0

        # Get occupied squares for logic
        my_occupied = state.occupied_squares1 if self.my_player_id == 1 else state.occupied_squares2

        for (r, c) in my_occupied:
            # Column Centrality (stay away from side edges)
            col_val = -abs(c - center)

            # Row Progression (Go to other side)
            # If P1 (Top), we want High Row Index. If P2 (Bottom), Low Row Index.
            if self.my_player_id == 1:
                row_val = r
            else:
                row_val = (N - 1) - r

            expansion_score += (row_val * 1.5 + col_val)

        # --- Weighting ---
        W_SCORE = 10000.0
        W_PENDING = 8000.0  # Almost as important as real score
        W_MOBILITY = 50.0  # Blocking value
        W_EXPAND = 10.0  # Tie-breaker

        total = (real_score_diff * W_SCORE) + \
                (pending_score * W_PENDING) + \
                (mobility_diff * W_MOBILITY) + \
                (expansion_score * W_EXPAND)

        return total

    # --- Helper Methods ---

    def get_reachable_squares(self, state: GameState, player_id: int) -> Set[Tuple[int, int]]:
        """
        Re-implements the logic of calculating allowed squares based on occupied squares.
        Returns a SET of (r, c) tuples.
        """
        N = state.board.N
        occupied = state.occupied_squares1 if player_id == 1 else state.occupied_squares2
        base_allowed = state.allowed_squares1 if player_id == 1 else state.allowed_squares2

        reachable = set()

        # 1. Add Base Allowed (empty ones only)
        # Note: state.allowed_squaresX usually contains the starting rows/zones.
        if base_allowed:
            for sq in base_allowed:
                if state.board.get(sq) == SudokuBoard.empty:
                    reachable.add(sq)

        # 2. Add Neighbors of Occupied
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
        """Generates actual Move objects for the Minimax search."""
        moves = []
        board = state.board
        N = board.N

        # Use our helper to get candidate squares
        candidates = self.get_reachable_squares(state, player_id)

        # Optimization: In classic game (no restrictions), candidates is everything
        if state.is_classic_game():
            candidates = {(r, c) for r in range(N) for c in range(N) if board.get((r, c)) == 0}

        for (r, c) in candidates:
            # Try values 1..N
            for val in range(1, N + 1):
                # Check Taboo
                if TabooMove((r, c), val) in state.taboo_moves:
                    continue
                # Check Sudoku Rules
                if self.is_valid_placement(board, r, c, val):
                    moves.append(Move((r, c), val))
        return moves

    def find_single_holes(self, board: SudokuBoard) -> List[Tuple[int, int]]:
        """
        Scans the board for empty cells that are the LAST empty cell in their Row, Col, or Block.
        Returns a list of (r, c) coordinates.
        """
        # This can be optimized, but for A1 readability > optimized parsing
        N = board.N
        m, n = board.m, board.n
        holes = set()

        # Check Rows
        for r in range(N):
            empty_in_row = []
            for c in range(N):
                if board.get((r, c)) == SudokuBoard.empty:
                    empty_in_row.append((r, c))
            if len(empty_in_row) == 1:
                holes.add(empty_in_row[0])

        # Check Cols
        for c in range(N):
            empty_in_col = []
            for r in range(N):
                if board.get((r, c)) == SudokuBoard.empty:
                    empty_in_col.append((r, c))
            if len(empty_in_col) == 1:
                holes.add(empty_in_col[0])

        # Check Blocks
        for br in range(0, N, m):
            for bc in range(0, N, n):
                empty_in_block = []
                for i in range(m):
                    for j in range(n):
                        if board.get((br + i, bc + j)) == SudokuBoard.empty:
                            empty_in_block.append((br + i, bc + j))
                if len(empty_in_block) == 1:
                    holes.add(empty_in_block[0])

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
        # Row
        if sum(1 for i in range(N) if board.get((r, i)) != 0) == N: regions += 1
        # Col
        if sum(1 for i in range(N) if board.get((i, c)) != 0) == N: regions += 1
        # Block
        m, n = board.m, board.n
        sr, sc = (r // m) * m, (c // n) * n
        if sum(1 for i in range(m) for j in range(n) if board.get((sr + i, sc + j)) != 0) == m * n: regions += 1

        return {0: 0, 1: 1, 2: 3, 3: 7}.get(regions, 0)

    def simulate_move(self, state: GameState, move: Move) -> GameState:
        new_state = copy.deepcopy(state)
        r, c, val = move.square[0], move.square[1], move.value
        p = new_state.current_player

        new_state.board.put((r, c), val)

        # Update occupied list
        if p == 1:
            new_state.occupied_squares1.append((r, c))
        else:
            new_state.occupied_squares2.append((r, c))

        # Add points
        pts = self.calculate_points_gained(new_state.board, r, c)
        new_state.scores[p - 1] += pts

        # Switch turn
        new_state.current_player = 3 - p
        return new_state