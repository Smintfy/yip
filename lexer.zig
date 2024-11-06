const std = @import("std");
const tk = @import("token.zig");

pub const Lexer = struct {
    source: []const u8,
    tokens: std.ArrayList(tk.Token),
    current: usize,

    pub fn new(allocator: std.mem.Allocator, source: []const u8) Lexer {
        return Lexer {
            .source = source,
            .tokens = std.ArrayList(tk.Token).init(allocator),
            .current = 0,
        };
    }

    pub fn tokenize(self: *Lexer) ![]tk.Token {
        while (self.advance()) |c| {
            try switch (c) {
                '(' => self.tokens.append(.{ .type = .LEFT_PAREN, .literal = &[_]u8{c} }),
                ')' => self.tokens.append(.{ .type = .RIGHT_PAREN, .literal = &[_]u8{c} }),
                '[' => self.tokens.append(.{ .type = .LEFT_SQUAB, .literal = &[_]u8{c} }),
                ']' => self.tokens.append(.{ .type = .RIGHT_SQUAB, .literal = &[_]u8{c} }),
                ',' => self.tokens.append(.{ .type = .COMMA, .literal = &[_]u8{c} }),
                '+' => self.tokens.append(.{ .type = .PLUS, .literal = &[_]u8{c} }),
                '-' => self.tokens.append(.{ .type = .MINUS, .literal = &[_]u8{c} }),
                '*' => self.tokens.append(.{ .type = .STAR, .literal = &[_]u8{c} }),
                '/' => self.tokens.append(.{ .type = .SLASH, .literal = &[_]u8{c} }),
                '=' => self.tokens.append(.{ .type = .EQUAL, .literal = &[_]u8{c} }),
                ':' => {
                    if (self.peekChar() == ':') {
                        _ = self.advance();
                        try self.tokens.append(.{
                            .type = .DECLARATION,
                            .literal = self.source[self.current - 2..self.current]
                        });
                    }
                },
                '!' => {
                    if (self.match('=')) {
                        try self.tokens.append(.{
                            .type = .NOT_EQUAL,
                            .literal = self.source[self.current - 2..self.current]
                        });
                    } else {
                        try self.tokens.append(.{
                            .type = .BANG,
                            .literal = &[_]u8{c},
                        });
                    }
                },
                '>' => {
                    if (self.match('=')) {
                        try self.tokens.append(.{
                            .type = .GREATER_EQUAL,
                            .literal = self.source[self.current - 2..self.current]
                        });
                    } else {
                        try self.tokens.append(.{
                            .type = .GREATER,
                            .literal = &[_]u8{c},
                        });
                    }
                },
                '<' => {
                    if (self.match('=')) {
                        try self.tokens.append(.{
                            .type = .LESS_EQUAL,
                            .literal = self.source[self.current - 2..self.current]
                        });
                    } else {
                        try self.tokens.append(.{
                            .type = .LESS,
                            .literal = &[_]u8{c},
                        });
                    }
                },
                'a'...'z', 'A'...'Z' => {
                    const start = self.current - 1;
                    while (self.peekChar()) |ch| {
                        if ('a' <= ch and ch <= 'z' or 'A' <= ch and ch <= 'Z') {
                            _ = self.advance();
                        } else {
                            break;
                        }
                    }

                    const key = self.source[start..self.current];
                    if (tk.keywords.get(key)) |token_type| {
                        try self.tokens.append(.{ .type = token_type, .literal = key });
                    } else {
                        try self.tokens.append(.{ .type = .IDENTIFIER, .literal = key });
                    }
                },
                '0'...'9' => {
                    const start = self.current - 1;
                    while (self.peekChar()) |ch| {
                        if ('0' <= ch and ch <= '9') {
                            _ = self.advance();
                        } else {
                            break;
                        }
                    }
                    try self.tokens.append(.{
                        .type = .NUMBER,
                        .literal =  self.source[start..self.current]
                    });
                },
                else => {
                    if (std.ascii.isWhitespace(c)) continue;
                    std.log.err("Unexpected character found: {c}", .{c});
                    std.process.exit(1);
                }
            };
        }
        try self.tokens.append(.{
            .type = .EOF,
            .literal = "EOF"
        });
        return self.tokens.toOwnedSlice();
    }

    fn match(self: *Lexer, expected: u8) bool {
        if (self.current >= self.source.len) return false;
        if (self.peekChar() != expected) return false;
        self.current += 1;
        return true;
    }

    fn advance(self: *Lexer) ?u8 {
        if (self.current >= self.source.len) return null;
        self.current += 1;
        return self.source[self.current - 1];
    }

    fn peekChar(self: *Lexer) ?u8 {
        if (self.current >= self.source.len) return null;
        return self.source[self.current];
    }

    pub fn deinit(self: *Lexer) void {
        self.tokens.deinit();
    }

    fn charToSlice(c: u8) []const u8 {
        return &[_]u8{c};
    }
};

// zig expectEqualSlices() compares the expected pointer and actual pointer
// if (expected.ptr == actual.ptr and expected.len == actual.len) {
//    return;
// }
//
// This is why we cannot directly compare arrays of slices to their actual
// form without the slices like the example below:
//
// const S = "A...Z";
//
// const Sa = hw[0..n];
// const Sb = hw[m..k];
//
// const A = [_][]const u8{hello, world};
// const E = [_][]const u8{"Hello", "World"};
//
// try std.testing.expectEqualSlices(&A, &E);
//
// This is due to the items of expected and actual having differen memory addresses.

test "symbols" {
    const allocator = std.testing.allocator;
    const source = "()[]+-/*";

    var lexer = Lexer.new(allocator, source);
    defer lexer.deinit();
    const tokens = try lexer.tokenize();
    defer allocator.free(tokens);

    const expected_tokens = [_]tk.Token{
        .{ .type = .LEFT_PAREN, .literal = "(" },
        .{ .type = .RIGHT_PAREN, .literal = ")" },
        .{ .type = .LEFT_SQUAB, .literal = "[" },
        .{ .type = .RIGHT_SQUAB, .literal = "]" },
        .{ .type = .PLUS, .literal = "+" },
        .{ .type = .MINUS, .literal = "-" },
        .{ .type = .SLASH, .literal = "/" },
        .{ .type = .STAR, .literal = "*" },
    };

    for (expected_tokens, 0..) |expected, i| {
        try std.testing.expectEqualStrings(expected.literal, tokens[i].literal);
    }
}

test "more_symbols" {
    const allocator = std.testing.allocator;
    const source =
        \\ true = false
        \\ true != false
        \\ 5 < 2
        \\ 6 <= 8
        \\ 7 > 5
        \\ 1 >= 3
    ;

    var lexer = Lexer.new(allocator, source);
    defer lexer.deinit();
    const tokens = try lexer.tokenize();
    defer allocator.free(tokens);

    const expected_tokens = [_]tk.Token{
        .{ .type = .TRUE, .literal = "true" },
        .{ .type = .EQUAL, .literal = "=" },
        .{ .type = .FALSE, .literal = "false" },
        // ============ //
        .{ .type = .TRUE, .literal = "true" },
        .{ .type = .NOT_EQUAL, .literal = "!=" },
        .{ .type = .FALSE, .literal = "false" },
        // ============ //
        .{ .type = .NUMBER, .literal = "5" },
        .{ .type = .GREATER, .literal = "<" },
        .{ .type = .NUMBER, .literal = "2" },
        // ============ //
        .{ .type = .NUMBER, .literal = "6" },
        .{ .type = .GREATER_EQUAL, .literal = "<=" },
        .{ .type = .NUMBER, .literal = "8" },
        // ============ //
        .{ .type = .NUMBER, .literal = "7" },
        .{ .type = .LESS, .literal = ">" },
        .{ .type = .NUMBER, .literal = "5" },
        // ============ //
        .{ .type = .NUMBER, .literal = "1" },
        .{ .type = .LESS_EQUAL, .literal = ">=" },
        .{ .type = .NUMBER, .literal = "3" },
    };

    for (expected_tokens, 0..) |expected, i| {
        try std.testing.expectEqualStrings(expected.literal, tokens[i].literal);
    }
}

test "identifiers" {
    const allocator = std.testing.allocator;
    const source =
        \\ set x :: 42
        \\ set y :: 3 * (2 + 1)
        \\ set z :: [69, 420]
    ;

    var lexer = Lexer.new(allocator, source);
    defer lexer.deinit();
    const tokens = try lexer.tokenize();
    defer allocator.free(tokens);

    const expected_tokens = [_]tk.Token{
        .{ .type = .SET, .literal = "set" },
        .{ .type = .IDENTIFIER, .literal = "x" },
        .{ .type = .DECLARATION, .literal = "::" },
        .{ .type = .NUMBER, .literal = "42" },
        // ============ //
        .{ .type = .SET, .literal = "set" },
        .{ .type = .IDENTIFIER, .literal = "y" },
        .{ .type = .DECLARATION, .literal = "::" },
        .{ .type = .NUMBER, .literal = "3" },
        .{ .type = .STAR, .literal = "*" },
        .{ .type = .LEFT_PAREN, .literal = "(" },
        .{ .type = .NUMBER, .literal = "2" },
        .{ .type = .PLUS, .literal = "+" },
        .{ .type = .NUMBER, .literal = "1" },
        .{ .type = .RIGHT_PAREN, .literal = ")" },
        // ============ //
        .{ .type = .SET, .literal = "set" },
        .{ .type = .IDENTIFIER, .literal = "z" },
        .{ .type = .DECLARATION, .literal = "::" },
        .{ .type = .LEFT_SQUAB, .literal = "[" },
        .{ .type = .NUMBER, .literal = "69" },
        .{ .type = .COMMA, .literal = "," },
        .{ .type = .NUMBER, .literal = "420" },
        .{ .type = .RIGHT_SQUAB, .literal = "]" },
    };

    for (expected_tokens, 0..) |expected, i| {
        try std.testing.expectEqualStrings(expected.literal, tokens[i].literal);
    }
}

// test "functions" {
//     const source =
//         \\ fn fib (n) ::
//         \\     if (n < 2)
//         \\         then n
//         \\     else
//         \\         fib (n - 1) + fib (n - 2)
//         \\ end
//         \\
//         \\ fn main ::
//         \\     print (fib(2))
//         \\ end
//     ;
// }
