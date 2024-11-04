const std = @import("std");
const lex = @import("lexer.zig");
const Lexer = lex.Lexer;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const source = "()+-=";

    var lexer = Lexer.new(allocator, source);
    defer lexer.tokens.deinit();
    const tokens = try lexer.tokenize();
    defer allocator.free(tokens);

    for (tokens) |t| {
        std.debug.print("{s} {?s}\n", .{t.literal, t.toString()});
    }
}
