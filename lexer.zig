const std = @import("std");
const tk = @import("token.zig");

pub const Lexer = struct {
    source: []const u8,
    tokens: std.ArrayList(tk.Token),
    current: u32,

    pub fn new(allocator: std.mem.Allocator, source: []const u8) Lexer {
        return Lexer{
            .source = source,
            .tokens = std.ArrayList(tk.Token).init(allocator),
            .current = 0
        };
    }

    pub fn tokenize(self: *Lexer) ![]tk.Token {
        for (self.source) |c| {
            try switch (c) {
                '(' => self.tokens.append(tk.Token{ .type = tk.TokenKind.LEFT_PAREN, .literal = "(" }),
                ')' => self.tokens.append(tk.Token{ .type = tk.TokenKind.RIGHT_PAREN, .literal = ")" }),
                '+' => self.tokens.append(tk.Token{ .type = tk.TokenKind.PLUS, .literal = "+" }),
                '-' => self.tokens.append(tk.Token{ .type = tk.TokenKind.MINUS, .literal = "-" }),
                '=' => self.tokens.append(tk.Token{ .type = tk.TokenKind.EQUAL, .literal = "=" }),
                else => {
                    std.log.err("Unexpected character found: {c}", .{c});
                    std.process.exit(1);
                }
            };
        }
        return self.tokens.toOwnedSlice();
    }

    fn advance(self: *Lexer) []const u8 {
        self.current += 1;
        return self.source[self.current - 1];
    }
};

test "tokens" {
    const allocator = std.testing.allocator;
    const source = "()";

    var lexer = Lexer.new(allocator, source);
    defer lexer.tokens.deinit();
    const tokens = try lexer.tokenize();
    defer allocator.free(tokens);

    const expected_tokens = [_]tk.Token{
        .{ .type = tk.TokenKind.LEFT_PAREN, .literal = "(" },
        .{ .type = tk.TokenKind.RIGHT_PAREN, .literal = ")" }
    };

    try std.testing.expectEqualSlices(tk.Token, &expected_tokens ,tokens);
}
