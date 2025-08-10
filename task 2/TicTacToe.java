import java.util.Scanner;

public class TicTacToe {

    private static char[] board = {' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '};
    private static char player = 'X';
    private static char ai = 'O';

    public static void printBoard() {
        System.out.println();
        for (int i = 0; i < 9; i += 3) {
            System.out.println(" " + board[i] + " | " + board[i + 1] + " | " + board[i + 2] + " ");
            if (i < 6) {
                System.out.println("---|---|---");
            }
        }
        System.out.println();
    }

    private static boolean isWinner(char[] bo, char le) {
        return (bo[0] == le && bo[1] == le && bo[2] == le) ||
               (bo[3] == le && bo[4] == le && bo[5] == le) ||
               (bo[6] == le && bo[7] == le && bo[8] == le) ||
               (bo[0] == le && bo[3] == le && bo[6] == le) ||
               (bo[1] == le && bo[4] == le && bo[7] == le) ||
               (bo[2] == le && bo[5] == le && bo[8] == le) ||
               (bo[0] == le && bo[4] == le && bo[8] == le) ||
               (bo[2] == le && bo[4] == le && bo[6] == le);
    }

    private static boolean isBoardFull(char[] bo) {
        for (char c : bo) {
            if (c == ' ') {
                return false;
            }
        }
        return true;
    }

    private static int minimax(char[] newBoard, boolean isMaximizing) {
        if (isWinner(newBoard, ai)) {
            return 1;
        } else if (isWinner(newBoard, player)) {
            return -1;
        } else if (isBoardFull(newBoard)) {
            return 0;
        }

        if (isMaximizing) {
            int bestScore = Integer.MIN_VALUE;
            for (int i = 0; i < 9; i++) {
                if (newBoard[i] == ' ') {
                    newBoard[i] = ai;
                    int score = minimax(newBoard, false);
                    newBoard[i] = ' ';
                    bestScore = Math.max(score, bestScore);
                }
            }
            return bestScore;
        } else {
            int bestScore = Integer.MAX_VALUE;
            for (int i = 0; i < 9; i++) {
                if (newBoard[i] == ' ') {
                    newBoard[i] = player;
                    int score = minimax(newBoard, true);
                    newBoard[i] = ' ';
                    bestScore = Math.min(score, bestScore);
                }
            }
            return bestScore;
        }
    }

    private static int getBestMove() {
        int bestScore = Integer.MIN_VALUE;
        int move = 0;
        for (int i = 0; i < 9; i++) {
            if (board[i] == ' ') {
                board[i] = ai;
                int score = minimax(board, false);
                board[i] = ' ';
                if (score > bestScore) {
                    bestScore = score;
                    move = i;
                }
            }
        }
        return move;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Welcome to Tic-Tac-Toe!");
        System.out.println("You are X, and the AI is O.");
        printBoard();

        while (true) {
            // Player's move
            int playerMove;
            while (true) {
                System.out.print("Enter your move (1-9): ");
                try {
                    playerMove = scanner.nextInt() - 1;
                    if (playerMove >= 0 && playerMove < 9 && board[playerMove] == ' ') {
                        board[playerMove] = player;
                        break;
                    } else {
                        System.out.println("Invalid move. Try again.");
                    }
                } catch (Exception e) {
                    System.out.println("Invalid input. Please enter a number between 1 and 9.");
                    scanner.next(); // clear the invalid input
                }
            }
            printBoard();

            if (isWinner(board, player)) {
                System.out.println("Congratulations! You won!");
                break;
            } else if (isBoardFull(board)) {
                System.out.println("It's a draw!");
                break;
            }

            // AI's move
            System.out.println("AI is thinking...");
            int aiMove = getBestMove();
            board[aiMove] = ai;
            System.out.println("AI chose position " + (aiMove + 1));
            printBoard();

            if (isWinner(board, ai)) {
                System.out.println("AI wins! Better luck next time.");
                break;
            } else if (isBoardFull(board)) {
                System.out.println("It's a draw!");
                break;
            }
        }
        scanner.close();
    }
}
