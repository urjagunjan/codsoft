import math

# Represents the game board
board = [' ' for _ in range(9)]

def print_board():
    """Prints the Tic-Tac-Toe board."""
    print()
    for i in range(0, 9, 3):
        print(f" {board[i]} | {board[i+1]} | {board[i+2]} ")
        if i < 6:
            print("---|---|---")
    print()

def is_winner(bo, le):
    """Checks if a player has won."""
    return ((bo[0] == le and bo[1] == le and bo[2] == le) or
            (bo[3] == le and bo[4] == le and bo[5] == le) or
            (bo[6] == le and bo[7] == le and bo[8] == le) or
            (bo[0] == le and bo[3] == le and bo[6] == le) or
            (bo[1] == le and bo[4] == le and bo[7] == le) or
            (bo[2] == le and bo[5] == le and bo[8] == le) or
            (bo[0] == le and bo[4] == le and bo[8] == le) or
            (bo[2] == le and bo[4] == le and bo[6] == le))

def is_board_full(bo):
    """Checks if the board is full."""
    return ' ' not in bo

def minimax(bo, depth, is_maximizing):
    """Minimax algorithm implementation."""
    if is_winner(bo, 'O'):
        return 1
    elif is_winner(bo, 'X'):
        return -1
    elif is_board_full(bo):
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if bo[i] == ' ':
                bo[i] = 'O'
                score = minimax(bo, depth + 1, False)
                bo[i] = ' '
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(9):
            if bo[i] == ' ':
                bo[i] = 'X'
                score = minimax(bo, depth + 1, True)
                bo[i] = ' '
                best_score = min(score, best_score)
        return best_score

def get_best_move():
    """Finds the best move for the AI."""
    best_score = -math.inf
    move = 0
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O'
            score = minimax(board, 0, False)
            board[i] = ' '
            if score > best_score:
                best_score = score
                move = i
    return move

def main():
    """Main game loop."""
    print("Welcome to Tic-Tac-Toe!")
    print("You are X, and the AI is O.")
    print_board()

    while True:
        # Player's move
        while True:
            try:
                player_move = int(input("Enter your move (1-9): ")) - 1
                if 0 <= player_move <= 8 and board[player_move] == ' ':
                    board[player_move] = 'X'
                    break
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 9.")

        print_board()

        if is_winner(board, 'X'):
            print("Congratulations! You won!")
            break
        elif is_board_full(board):
            print("It's a draw!")
            break

        # AI's move
        print("AI is thinking...")
        ai_move = get_best_move()
        board[ai_move] = 'O'
        print(f"AI chose position {ai_move + 1}")
        print_board()

        if is_winner(board, 'O'):
            print("AI wins! Better luck next time.")
            break
        elif is_board_full(board):
            print("It's a draw!")
            break

if __name__ == "__main__":
    main()
