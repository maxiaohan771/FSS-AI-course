"""
Five-in-a-Row (Gomoku) - Human vs Human
Two players take turns placing pieces on a 15x15 board.
First to get 5 consecutive pieces wins.
"""

# Game constants
BOARD_SIZE = 15
EMPTY = '+'  # Middle dot for empty cells
PLAYER_1 = '●'  # Black circle
PLAYER_2 = '○'  # White circle

# Directions to check for wins
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def create_board():
    """Create an empty game board."""
    return [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def print_board(board):
    """Display the board with coordinates."""
    # Print column numbers
    print("\n   ", end="")
    for col in range(BOARD_SIZE):
        print(f"{col:2}", end=" ")
    print()
    
    # Print top border
    print("   +" + "---" * BOARD_SIZE)
    
    # Print each row
    for row in range(BOARD_SIZE):
        print(f"{row:2} |", end=" ")
        for col in range(BOARD_SIZE):
            print(board[row][col], end="  ")
        print()
    
    print()


def is_valid_move(board, row, col):
    """Check if a move is valid."""
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board[row][col] == EMPTY


def check_win(board, row, col, player):
    """
    Check if the last move at (row, col) creates a win for 'player'.
    Returns True if 5 or more consecutive pieces exist in any direction.
    """
    for dr, dc in DIRECTIONS:
        count = 1  # Start with the piece just placed
        
        # Check in positive direction
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
            count += 1
            r += dr
            c += dc
        
        # Check in negative direction
        r, c = row - dr, col - dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
            count += 1
            r -= dr
            c -= dc
        
        if count >= 5:
            return True
    
    return False


def is_board_full(board):
    """Check if the board has no empty spaces left."""
    for row in board:
        if EMPTY in row:
            return False
    return True


def get_player_symbol(player_num):
    """Return the symbol for a given player number."""
    return PLAYER_1 if player_num == 1 else PLAYER_2


def get_player_name(player_num):
    """Return the display name for a player."""
    return f"Player {player_num} ({get_player_symbol(player_num)})"


def clear_screen():
    """Clear the terminal screen."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def show_instructions():
    """Display game instructions."""
    print("=" * 60)
    print("            FIVE-IN-A-ROW (GOMOKU)")
    print("                  Human vs Human")
    print("=" * 60)
    print("\n📖 RULES:")
    print("   • Player 1 uses", PLAYER_1, "(black)")
    print("   • Player 2 uses", PLAYER_2, "(white)")
    print("   • Take turns placing your piece on an empty cell")
    print("   • First to get 5 pieces in a row wins!")
    print("   • Rows can be horizontal, vertical, or diagonal")
    print("\n🎮 HOW TO PLAY:")
    print(f"   • Board size: {BOARD_SIZE}x{BOARD_SIZE}")
    print("   • Coordinates range from 0 to", BOARD_SIZE - 1)
    print("   • Enter row and column numbers separated by space")
    print("   • Example: '7 7' places a piece in the center")
    print("   • Type 'quit' to exit the game")
    print("\n💡 TIPS:")
    print("   • Control the center of the board")
    print("   • Create multiple threats")
    print("   • Block your opponent's potential lines")
    print("=" * 60)
    input("\nPress Enter to start the game...")


def play_game():
    """Main game loop for two human players."""
    board = create_board()
    current_player = 1
    game_over = False
    
    while not game_over:
        clear_screen()
        print_board(board)
        
        # Show current player's turn
        player_name = get_player_name(current_player)
        print(f"\n🔴 {player_name}'s TURN 🔴")
        
        # Get player's move
        while True:
            move = input("Enter row and column (or 'quit'): ").strip().lower()
            
            if move == 'quit':
                print("\n👋 Thanks for playing! Goodbye!")
                return False
            
            try:
                row, col = map(int, move.split())
                
                if is_valid_move(board, row, col):
                    # Make the move
                    symbol = get_player_symbol(current_player)
                    board[row][col] = symbol
                    
                    # Check for win
                    if check_win(board, row, col, symbol):
                        clear_screen()
                        print_board(board)
                        print("\n" + "=" * 60)
                        print(f"   🎉🎉🎉  {player_name} WINS!  🎉🎉🎉")
                        print("=" * 60)
                        game_over = True
                        break
                    
                    # Check for draw
                    if is_board_full(board):
                        clear_screen()
                        print_board(board)
                        print("\n" + "=" * 60)
                        print("           📍 IT'S A DRAW! 📍")
                        print("           The board is completely filled.")
                        print("=" * 60)
                        game_over = True
                        break
                    
                    # Switch to other player
                    current_player = 2 if current_player == 1 else 1
                    break
                    
                else:
                    print("❌ Invalid move! Cell is either out of bounds or already occupied.")
                    
            except ValueError:
                print("❌ Invalid input! Please enter two numbers separated by space (e.g., '7 7')")
    
    # Ask for rematch
    while True:
        choice = input("\n🎮 Play again? (y/n): ").strip().lower()
        if choice == 'y':
            return True
        elif choice == 'n':
            print("\n👋 Thanks for playing! Goodbye!")
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no")


def main():
    """Main program entry point."""
    show_instructions()
    
    playing = True
    while playing:
        playing = play_game()


if __name__ == "__main__":
    main()