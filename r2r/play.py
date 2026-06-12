"""Play against agents from the command line.

The Module 0 boss fight:

    python -m r2r.play boss

You move by typing a column number. The boss thinks by imagining ~3000
random futures per move. It has never been taught Connect Four. Good luck.
"""

import sys

from r2r.builtin import MCTSAgent, MinimaxAgent, RandomAgent
from r2r.envs.games import ConnectFour, TicTacToe


class HumanAgent:
    def select_move(self, game, state):
        legal = game.legal_moves(state)
        while True:
            try:
                raw = input(f"your move {legal}: ").strip()
            except EOFError:
                print("\n(goodbye)")
                sys.exit(0)
            try:
                move = int(raw)
            except ValueError:
                print("type a number")
                continue
            if move in legal:
                return move
            print("illegal move")


OPPONENTS = {
    "boss": (ConnectFour(), lambda: MCTSAgent(n_simulations=3000)),
    "mcts": (ConnectFour(), lambda: MCTSAgent(n_simulations=400)),
    "random": (ConnectFour(), lambda: RandomAgent()),
    "ttt-perfect": (TicTacToe(), lambda: MinimaxAgent()),
    "ttt-random": (TicTacToe(), lambda: RandomAgent()),
}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "boss"
    if name not in OPPONENTS:
        print(f"unknown opponent {name!r}. options: {', '.join(OPPONENTS)}")
        return
    game, make_agent = OPPONENTS[name]
    agent = make_agent()
    human_player = 1
    if len(sys.argv) > 2 and sys.argv[2] == "second":
        human_player = -1

    if name == "boss":
        print("=" * 46)
        print("  MODULE 0: THE BET")
        print("=" * 46)
        print("This is the machine you will build.")
        print("Every line of it. By the end of this course.")
        print("First: lose to it.\n")

    state = game.initial_state()
    human = HumanAgent()
    print(game.render(state) + "\n")
    while game.winner(state) is None:
        player = game.current_player(state)
        if player == human_player:
            move = human.select_move(game, state)
        else:
            print("thinking...")
            move = agent.select_move(game, state)
            print(f"opponent plays {move}")
        state = game.next_state(state, move)
        print(game.render(state) + "\n")

    w = game.winner(state)
    if w == 0:
        print("Draw.")
    elif w == human_player:
        print("You won. (Try 'boss' with more simulations... or proceed to Module 1.)")
    else:
        print("You lost.")
        if name == "boss":
            print("\nRemember this feeling. In eleven modules, this machine")
            print("will exist because you built it — and it will have taught")
            print("itself, starting from nothing but a reward signal.")
            print("Proceed to modules/01-the-coin.")


if __name__ == "__main__":
    main()
