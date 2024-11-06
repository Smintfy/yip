const std = @import("std");
const tk = @import("token.zig");

const Node = union(enum) {
    Identifier: []const u8,
    Number: f64,
    Set: struct {
        name: *Node,
        value: *Node,
    },
    Function: struct {
        name: *Node,
        params: [](*Node),
        body: [](*Node),
    },
    BinaryOp: struct {
        left: *Node,
        op: tk.Token,
        right: *Node,
    },
};

test "ast" {
    const set_tree = Node {
        .Set = .{
            .name = @constCast(&Node{ .Identifier = "foo" }),
            .value = @constCast(&Node{ .Number =  5.0 })
        }
    };

    try std.testing.expectEqual("foo", set_tree.Set.name.Identifier);
    try std.testing.expectEqual(@as(f64, 5.0), set_tree.Set.value.Number);

    const function_tree = Node {
        .Function = .{
            .name = @constCast(&Node{ .Identifier = "add" }),
            .params = @constCast(&[_]*Node{
                    @constCast(&Node{ .Identifier = "x" }),
                    @constCast(&Node{ .Identifier = "y" }),
                },
            ),
            .body = @constCast(&[_]*Node{
                    @constCast(&Node {
                        .BinaryOp = .{
                            .left = @constCast(&Node{ .Identifier = "x" }),
                            .op = tk.Token { .type = .PLUS, .literal = "+" },
                            .right = @constCast(&Node{ .Identifier = "y" }),
                        },
                    }),
                },
            )
        },
    };

    try std.testing.expectEqual("add", function_tree.Function.name.Identifier);
    try std.testing.expectEqual("x", function_tree.Function.params[0].Identifier);
    try std.testing.expectEqual("y", function_tree.Function.params[1].Identifier);
    try std.testing.expectEqual("x", function_tree.Function.body[0].BinaryOp.left.Identifier);
    try std.testing.expectEqual("+", function_tree.Function.body[0].BinaryOp.op.literal);
    try std.testing.expectEqual("y", function_tree.Function.body[0].BinaryOp.right.Identifier);
}
