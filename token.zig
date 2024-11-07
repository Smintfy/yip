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
    ASSIGN,
    END,
    THEN,

    ERROR,
    EOF
};

pub const keywords = std.StaticStringMap(TokenKind).initComptime(.{
    .{ "set", TokenKind.SET },
    .{ "end", TokenKind.END },
    .{ "true", TokenKind.TRUE },
    .{ "false", TokenKind.FALSE },
    .{ "if", TokenKind.IF },
    .{ "then", TokenKind.THEN },
    .{ "fn", TokenKind.IF },
});

pub const Token = struct {
    type: TokenKind,
    literal: []const u8,

    pub fn toString(self: Token) ?[]const u8 {
        return std.enums.tagName(TokenKind, self.type);
    }
};
