const std = @import("std");
const tk = @import("token.zig");

pub const Node = union(enum) {
    Number: f64,
    String: []const u8,
    Variable: tk.Token,
    Set: struct { name: *Node, value: *Node },
    Function: struct { name: *Node, args: []*Node, body: []*Node },
    Call: struct { function: *Node, args: []*Node },
    BinaryOp: struct { left: *Node, op: tk.Token, right: *Node },
    UnaryOp: struct { op: tk.Token, expr: *Node },
};

test "ast" {
    const set_tree = Node {
        .Set = .{
            .name = @constCast(&Node{
                .Variable = tk.Token { .type = .IDENTIFIER, .literal = "foo" }
            }),
            .value = @constCast(&Node{ .Number =  5.0 })
        }
    };

    try std.testing.expectEqual("foo", set_tree.Set.name.Variable.literal);
    try std.testing.expectEqual(@as(f64, 5.0), set_tree.Set.value.Number);

    const function_tree = Node {
        .Function = .{
            .name = @constCast(&Node{
                .Variable = tk.Token { .type = .IDENTIFIER, .literal = "add" }
            }),
            .args = @constCast(&[_]*Node{
                    @constCast(&Node{
                        .Variable = tk.Token { .type = .IDENTIFIER, .literal = "x" }
                    }),
                    @constCast(&Node{
                        .Variable = tk.Token { .type = .IDENTIFIER, .literal = "y" }
                    }),
                },
            ),
            .body = @constCast(&[_]*Node{
                    @constCast(&Node {
                        .BinaryOp = .{
                            .left = @constCast(&Node{
                                .Variable = tk.Token { .type = .IDENTIFIER, .literal = "x" }
                            }),
                            .op = tk.Token { .type = .PLUS, .literal = "+" },
                            .right = @constCast(&Node{
                                .Variable = tk.Token { .type = .IDENTIFIER, .literal = "y" }
                            }),
                        },
                    }),
                },
            )
        },
    };

    try std.testing.expectEqual("add", function_tree.Function.name.Variable.literal);
    try std.testing.expectEqual("x", function_tree.Function.args[0].Variable.literal);
    try std.testing.expectEqual("y", function_tree.Function.args[1].Variable.literal);
    try std.testing.expectEqual("x", function_tree.Function.body[0].BinaryOp.left.Variable.literal);
    try std.testing.expectEqual("+", function_tree.Function.body[0].BinaryOp.op.literal);
    try std.testing.expectEqual("y", function_tree.Function.body[0].BinaryOp.right.Variable.literal);
}
