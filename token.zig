pub const TokenKind = enum {
    LEFT_PAREN,
    RIGHT_PAREN,
    PLUS,
    MINUS,
    EQUAL
};

pub const Token = struct {
    type:       TokenKind,
    literal:    []const u8,
};
