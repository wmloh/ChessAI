### GENERAL CONSTANTS ### (should be moved to another constants file later)
BOARD_DIM = (8, 8)
ACTION_DEPTH = 2

### PIECE REPRESENTATION ###
STATE_DEPTH = 13

PAWN_W = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
BISH_W = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0)
KNHT_W = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)
ROOK_W = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
QUEN_W = (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)
KING_W = (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)

PAWN_B = (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)
BISH_B = (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
KNHT_B = (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
ROOK_B = (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
QUEN_B = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
KING_B = (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

EMPTY = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

### PGN CONSTANTS ###
WHITE_WIN = '1-0'
BLACK_WIN = '0-1'
BOTH_DRAW = '1/2-1/2'

### MAPPINGS ###
MAP_SYMBOL = {
    '.': EMPTY,
    'P': PAWN_W,
    'B': BISH_W,
    'N': KNHT_W,
    'R': ROOK_W,
    'Q': QUEN_W,
    'K': KING_W,
    'p': PAWN_B,
    'b': BISH_B,
    'n': KNHT_B,
    'r': ROOK_B,
    'q': QUEN_B,
    'k': KING_B
}

FLIP_MAP_SYMBOL = {
    '.': EMPTY,
    'p': PAWN_W,
    'b': BISH_W,
    'n': KNHT_W,
    'r': ROOK_W,
    'q': QUEN_W,
    'k': KING_W,
    'P': PAWN_B,
    'B': BISH_B,
    'N': KNHT_B,
    'R': ROOK_B,
    'Q': QUEN_B,
    'K': KING_B
}

# assumes one-to-one correspondence
INVERSE_MAP_SYMBOL = {v: k for k, v in MAP_SYMBOL.items()}

# FEN default options allowing white next turn, all sides castling, no enpassant, and no 50-move limit
FEN_DEFAULT = ' w KQkq - 0 0'