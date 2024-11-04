const std = @import("std");

pub const TokenKind = enum {
    LEFT_PAREN,
    RIGHT_PAREN,
    PLUS,
    MINUS,
    EQUAL
};

pub const Token = struct {
    type: TokenKind,
    literal: []const u8,

    pub fn toString(self: Token) ?[]const u8 {
        return std.enums.tagName(TokenKind, self.type);
    }
};
