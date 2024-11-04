const std = @import("std");
const tk = @import("token.zig");

const Lexer = struct {
    source: []const u8,
    tokens: std.ArrayList(tk.Token),
    current: u32,

    fn new(allocator: std.mem.Allocator, source: []const u8) Lexer {
        return Lexer{
            .source = source,
            .tokens = std.ArrayList(tk.Token).init(allocator),
            .current = 0
        };
    }

    fn tokenize(self: *Lexer) !std.ArrayList((tk.Token)) {
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
        return self.tokens;
    }

    fn advance(self: *Lexer) []const u8 {
        self.current += 1;
        return self.source[self.current - 1];
    }
};

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const source = "()+-=";

    var lexer = Lexer.new(allocator, source);
    defer lexer.tokens.deinit();
    const tokens = try lexer.tokenize();

    for (tokens.items) |t| {
        std.debug.print("{s} {?s}\n", .{t.literal, std.enums.tagName(tk.TokenKind, t.type)});
    }
}
