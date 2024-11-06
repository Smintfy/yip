const std = @import("std");
const lex = @import("lexer.zig");
const tk = @import("token.zig");
const Lexer = lex.Lexer;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const source = "set x :: 9 _";

    var lexer = Lexer.new(allocator, source);
    defer lexer.deinit();
    const tokens = try lexer.tokenize();
    defer allocator.free(tokens);

    for (tokens) |t| {
        std.debug.print("{s} {?s}\n", .{t.literal, t.toString()});
    }
}
