from copy import deepcopy

class TikTacToeState:
    def __init__(self, board_state, parent=None, parent_action=None):
        self.board_state = board_state
        self.parent = parent
        self.parent_action = parent_action
        self.utility_value = None
        self.childrens = []

    def __str__(self):
        out = "+---+---+---+\n"
        for row in self.board_state:
            out += "|"
            for item in row:
                if item == 0:
                    out += "   |"
                elif item == 1:
                    out += " X |"
                elif item == -1:
                    out += " O |"
            out += "\n"
            out += "+---+---+---+\n"
        return out
    
    def __repr__(self):
        return self.__str__()
    
    def get_childrens(self, player):
        childrens = []
        for i in range(3):
            for j in range(3):
                if self.board_state[i][j] == 0:
                    new_board_state = deepcopy(self.board_state)
                    new_board_state[i][j] = player
                    childrens.append(TikTacToeState(new_board_state, self, (i, j)))
        self.childrens = childrens
        return childrens
    
    @staticmethod
    def _terminal_test(state, last_player):
        board_state = state.board_state
        zero_found = False
        if board_state[0][0] == last_player and board_state[1][1] == last_player and board_state[2][2] == last_player:
            state.utility_value = last_player
            return True
        if board_state[0][2] == last_player and board_state[1][1] == last_player and board_state[2][0] == last_player:
            state.utility_value = last_player
            return True
        for i in range(3):
            if board_state[i][0] == last_player and board_state[i][1] == last_player and board_state[i][2] == last_player:
                state.utility_value = last_player
                return True
            if board_state[0][i] == last_player and board_state[1][i] == last_player and board_state[2][i] == last_player:
                state.utility_value = last_player
                return True
            if 0 in board_state[i]:
                zero_found = True
        if not zero_found:
            state.utility_value = 0
            return True
        return False
    

class TikTacToeMinimaxSolver:    
    def _max_value(self, state): 
        if TikTacToeState._terminal_test(state, -1):
            return state.utility_value
        value = -2
        for child in state.get_childrens(player=1):
            value = max(value, self._min_value(child))
        state.utility_value = value
        return value
    
    def _min_value(self, state):
        if TikTacToeState._terminal_test(state, 1):
            return state.utility_value
        value = 2
        for child in state.get_childrens(player=-1):
            value = min(value, self._max_value(child))
        state.utility_value = value
        return value
    
    def get_next_move(self, state, player):
        if player == 1:
            value = self._max_value(state)
            for s in state.childrens:
                if s.utility_value == value:
                    return s
        else:
            value = self._min_value(state)
            for s in state.childrens:
                if s.utility_value == value:
                    return s
                
    def minimax(self, state):
        return self.get_next_move(state, 1).parent_action
    
class HumanPlayer:
    def get_next_move(self, state, player):
        out = "+-----+-----+-----+\n"
        for i in range(3):
            row = state.board_state[i]
            out += "|"
            for j in range(3):
                item = row[j]
                if item == 0:
                    out += f"({i},{j})|"
                elif item == 1:
                    out += "  X  |"
                elif item == -1:
                    out += "  O  |"
            out += "\n"
            out += "+-----+-----+-----+\n"
        print(out)
        next_move = eval(input("Your move [format : (row, col)]: "))
        while state.board_state[next_move[0]][next_move[1]] != 0:
            next_move = eval(input("Invalid move. Your move [format : (row, col)]: "))
        state.board_state[next_move[0]][next_move[1]] = player
        return state
    
class PlayTicTacToe:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
    
    def play(self):
        state = TikTacToeState([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        while True:
            if TikTacToeState._terminal_test(state, -1):
                print(f"\nGame Over {"player 2 won" if state.utility_value == -1 else "draw"}")
                print(state)
                break
            state = self.player1.get_next_move(state, 1)
            if TikTacToeState._terminal_test(state, 1):
                print(f"\nGame Over {'player 1 won' if state.utility_value == 1 else 'draw'}")
                print(state)
                break
            state = self.player2.get_next_move(state, -1)

    
if __name__ == "__main__":
    state = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    root = TikTacToeState(state)
    play = PlayTicTacToe(HumanPlayer(), TikTacToeMinimaxSolver())
    play.play()