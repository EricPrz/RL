import chess
import chess.engine

board = chess.Board()
engine = chess.engine.SimpleEngine.popen_uci("/home/eric/Downloads/dragon/dragon_05e2a7/Linux/dragon-linux-avx2")

while not board.is_game_over():
    result = engine.play(board, chess.engine.Limit(time=0.1))
    board.push(result.move)
    print(board)

engine.quit()
