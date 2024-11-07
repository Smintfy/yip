const std = @import("std");
const tk = @import("token.zig");
const lex = @import("lexer.zig");
const ast = @import("ast.zig");

pub const Parser = struct {
    tokens: []tk.Token,
    current: usize,
    allocator: std.mem.Allocator,

    pub fn new(allocator: std.mem.Allocator, tokens: []tk.Token) Parser {
        return .{
            .tokens = tokens,
            .current = 0,
            .allocator = allocator
        };
    }

    pub fn parse(self: *Parser) ![]*ast.Node {
        var statements = std.ArrayList(*ast.Node).init(self.allocator);
        defer statements.deinit();
        while (self.peek().type != tk.TokenKind.EOF) {
            switch (self.peek().type) {
                .SET => {
                    try statements.append(@constCast(&(try self.parseSet())));
                },
                else => {
                    std.log.warn("Unhandled token: {s}", .{self.peek().literal});
                    _ = self.advance();
                }
            }
        }
        return statements.toOwnedSlice();
    }

    fn parseSet(self: *Parser) !ast.Node {
        _ = self.advance();
        const identifier = try self.consume(tk.TokenKind.IDENTIFIER);
        _ = try self.consume(tk.TokenKind.ASSIGN);
        const value = try self.consume(tk.TokenKind.NUMBER);
        return ast.Node{
            .Set = .{
                .name = @constCast(&ast.Node{ .Variable = identifier }),
                .value = @constCast(&ast.Node{ .Number = (try std.fmt.parseFloat(f64, value.literal)) })
            }
        };
    }

    fn peek(self: *Parser) tk.Token {
        return self.tokens[self.current];
    }

    fn consume(self: *Parser, token_type: tk.TokenKind) !tk.Token {
        if (self.peek().type == token_type) {
            return self.advance();
        } else {
            std.log.err("Unhandled error consume()", .{});
            return tk.Token{ .type = .ERROR, .literal = "" };
        }
    }

    fn advance(self: *Parser) tk.Token {
        self.current += 1;
        return self.tokens[self.current - 1];
    }
};

test "parser" {
    const allocator = std.testing.allocator;
    const source = "set x :: 9";

    var lexer = lex.Lexer.new(allocator, source);
    defer lexer.deinit();
    const tokens = try lexer.tokenize();
    defer allocator.free(tokens);

    var parser = Parser.new(allocator, tokens);
    const statements = try parser.parse();
    defer allocator.free(statements);

    for (statements) |stmt| {
        std.debug.print("{s}\n", .{@tagName(stmt.*)});
    }
}
