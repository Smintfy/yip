const std = @import("std");

pub const TokenKind = enum {
    LEFT_PAREN,
    RIGHT_PAREN,
    LEFT_SQUAB,
    RIGHT_SQUAB,
    COMMA,

    BANG,
    EQUAL,
    NOT_EQUAL,
    GREATER,
    GREATER_EQUAL,
    LESS,
    LESS_EQUAL,

    PLUS,
    MINUS,
    STAR,
    SLASH,

    NUMBER,
    STRING,
    IDENTIFIER,

    SET,
    TRUE,
    FALSE,
    IF,
    DECLARATION,
    END_DECLARATION,

    EOF
};

pub const keywords = std.StaticStringMap(TokenKind).initComptime(.{
    .{ "set", TokenKind.SET },
    .{ "end", TokenKind.END_DECLARATION },
    .{ "true", TokenKind.TRUE },
    .{ "false", TokenKind.FALSE },
    .{ "if", TokenKind.IF }
});

pub const Token = struct {
    type: TokenKind,
    literal: []const u8,

    pub fn toString(self: Token) ?[]const u8 {
        return std.enums.tagName(TokenKind, self.type);
    }
};

test "keywords" {
    try std.testing.expectEqual(TokenKind.SET, keywords.get("set").?);
}
