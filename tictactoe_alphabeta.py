from copy import deepcopy
import math
import matplotlib.pyplot as plt

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
    

class TikTacToeAlphaBetaSolver:    
    def _max_value(self, state, alpha, beta): 
        if TikTacToeState._terminal_test(state, -1):
            return state.utility_value
        value = -2
        for child in state.get_childrens(player=1):
            value = max(value, self._min_value(child, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        state.utility_value = value
        return value
    
    def _min_value(self, state, alpha, beta):
        if TikTacToeState._terminal_test(state, 1):
            return state.utility_value
        value = 2
        for child in state.get_childrens(player=-1):
            value = min(value, self._max_value(child, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        state.utility_value = value
        return value
    
    def get_next_move(self, state, player):
        if player == 1:
            value = self._max_value(state, alpha=-math.inf, beta=math.inf)
            for s in state.childrens:
                if s.utility_value == value:
                    return s
        else:
            value = self._min_value(state, alpha=-math.inf, beta=math.inf)
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
    
    def play(self, terminal=True):
        if terminal:
            self._play_terminal()
        else:
            self._play_graph()

    def _play_terminal(self):
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

    def _initial_plot(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 3)
        self.ax.set_ylim(0, 3)
        self.ax.set_aspect('equal')
        for i in range(1, 3):
            self.ax.axhline(i, color='black', linewidth=2)
            self.ax.axvline(i, color='black', linewidth=2)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        plt.pause(0.1)

    def _update_plot(self, state):
        self.ax.clear()
        self.ax.set_xlim(0, 3)
        self.ax.set_ylim(0, 3)
        self.ax.set_aspect('equal')
        self.ax.axis(False)
        for i in range(1, 3):
            self.ax.axhline(i, color='black', linewidth=2)
            self.ax.axvline(i, color='black', linewidth=2)
        for i in range(3):
            for j in range(3):
                if state.board_state[i][j] == 1:
                    self.ax.text(j+0.5, 3-(i+0.5), f'X', ha='center', fontsize=50, va='center', color='red')
                elif state.board_state[i][j] == -1:
                    self.ax.text(j+0.5, 3-(i+0.5), f'O', ha='center', fontsize=50, va='center', color='blue')
        self.ax.set_title("Tic-Tac-Toe")
        self.state = state
        plt.pause(0.5)
    
    def _on_click(self, event):
        if self.human_move:
            row = math.floor(3 - event.ydata)
            col = math.floor(event.xdata)
            if self.state.board_state[row][col] == 0:
                self.state.board_state[row][col] = 1 if self.human_first else -1
                self.fig.canvas.mpl_disconnect(self.cid)
                self.human_move = False

    def _final_plot(self, state, result):
        self._update_plot(state)
        self.ax.set_title(result)
        if "Human" in result:
            self.fig.set_facecolor("g")
        elif "Computer" in result:
            self.fig.set_facecolor("b")
        else:
            self.fig.set_facecolor("g")
        plt.pause(0.5)

    def _play_graph(self):
        state = TikTacToeState([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        if isinstance(self.player1, HumanPlayer) and isinstance(self.player2, HumanPlayer):
            raise Exception("Both players are human")
        elif isinstance(self.player1, HumanPlayer):
            self.human_move = True
            self.human_first = True
        elif isinstance(self.player2, HumanPlayer):
            self.human_move = False
            self.human_first = False
        else:
            raise Exception("Human player not found")
        self._initial_plot()
        if self.human_move:
            while True:
                self._update_plot(state)
                if TikTacToeState._terminal_test(state, -1):
                    self._final_plot(state, f"\nGame Over {'Computer won' if self.state.utility_value == -1 else 'draw'}")
                    break
                self.cid = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
                state = self.state
                self._update_plot(state)
                if TikTacToeState._terminal_test(state, 1):
                    self._final_plot(state, f"\nGame Over {'Human won' if self.state.utility_value == 1 else 'draw'}")
                    break
                if self.human_move is False:
                    state = self.player2.get_next_move(self.state, -1)
                    self.human_move = True
        else:
            while True:
                self._update_plot(state)
                if TikTacToeState._terminal_test(state, -1):
                    self._final_plot(state, f"\nGame Over {'Human won' if self.state.utility_value == -1 else 'draw'}")
                    break
                if self.human_move is False:
                    state = self.player1.get_next_move(self.state, 1)
                    self.human_move = True
                self._update_plot(state)
                if TikTacToeState._terminal_test(state, 1):
                    self._final_plot(state, f"\nGame Over {'Computer won' if self.state.utility_value == 1 else 'draw'}")
                    break
                self.cid = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
                print(self.state.parent_action)
                state = self.state
        plt.show()


    
if __name__ == "__main__":
    from time import perf_counter
    start = perf_counter()
    state = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    root = TikTacToeState(state)
    play = PlayTicTacToe(HumanPlayer(), TikTacToeAlphaBetaSolver())
    play.play(terminal=False)
    end = perf_counter()
    print("It took total of ", end - start, " seconds")