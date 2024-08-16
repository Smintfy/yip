from __future__ import annotations
from typing import List, Any
from enum import Enum
import argparse
import sys


class TokenType(Enum):
    # Single character
    LEFT_PAREN  = "LEFT_PAREN"
    RIGHT_PAREN = "RIGHT_PAREN"
    PLUS = "PLUS"
    MINUS = "MINUS"
    STAR = "STAR"
    SLASH = "SLASH"
    SEMICOLON = "SEMICOLON"

    # One or two character
    BANG = "BANG"
    BANG_EQUAL = "BANG_EQUAL"
    EQUAL = "EQUAL"
    EQUAL_EQUAL = "EQUAL_EQUAL"
    GREATER = "GREATER"
    GREATER_EQUAL = "GREATER_EQUAL"
    LESS = "LESS"
    LESS_EQUAL = "LESS_EQUAL"

    # Literals
    IDENTIFIER = "IDENTIFIER"
    STRING = "STRING"
    NUMBER = "NUMBER"

    # Keywords
    SET = "SET"
    WRITE = "WRITE"
    PROC = "PROC"
    IF = "IF"
    ELSE = "ELSE"
    TRUE = "TRUE"
    FALSE = "FALSE"
    OR = "OR"
    AND = "AND"
    NONE = "NONE"

    # End-of-line
    EOF = "EOF"


keywords = {
    "set": TokenType.SET,
    "write": TokenType.WRITE,
    "proc": TokenType.PROC,
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE,
    "or": TokenType.OR,
    "and": TokenType.AND
}


class Token:
    def __init__(self, token_type: TokenType, lexeme: str, literal: Any, line: int):
        self.token_type = token_type
        self.lexeme = lexeme
        self.literal = literal
        self.line = line

    def __str__(self) -> str:
        return f"{self.token_type.value} {self.lexeme} {self.literal}"


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.tokens = []
        self.start = 0
        self.current = 0
        self.line = 1

    def tokenize(self) -> List[Token]:
        while not self.is_at_end():
            self.start = self.current
            self.scan_token()
        self.tokens.append(Token(TokenType.EOF, "", None, self.line))
        return self.tokens

    def scan_token(self):
        char = self.advance()
        match char:
            case "(":
                self.add_token(TokenType.LEFT_PAREN)
            case ")":
                self.add_token(TokenType.RIGHT_PAREN)
            case "+":
                self.add_token(TokenType.PLUS)
            case "-":
                self.add_token(TokenType.MINUS)
            case "*":
                self.add_token(TokenType.STAR)
            case "/":
                self.add_token(TokenType.SLASH)
            case "!":
                if self.match("="):
                    self.add_token(TokenType.BANG_EQUAL)
                else:
                    self.add_token(TokenType.BANG)
            case "=":
                if self.match("="):
                    self.add_token(TokenType.EQUAL_EQUAL)
                else:
                    self.add_token(TokenType.EQUAL)
            case "<":
                if self.match("="):
                    self.add_token(TokenType.LESS_EQUAL)
                else:
                    self.add_token(TokenType.LESS)
            case ">":
                if self.match("="):
                    self.add_token(TokenType.GREATER_EQUAL)
                else:
                    self.add_token(TokenType.GREATER)
            case "\n":
                self.line += 1
            case ";":
                self.comment()
            case '"':
                self.string()
            case _:
                if char.isdigit():
                    self.number()
                elif char.isalpha():
                    self.identifier()
                elif char.isspace():
                    pass
                else:
                    raise SyntaxError(f"Unexpected character: {char} in line: {self.line}")

    def advance(self) -> str:
        self.current += 1
        return self.source[self.current - 1]

    def is_at_end(self) -> bool:
        return (self.current) >= len(self.source)

    def peek(self) -> str:
        if self.is_at_end():
            return "\0"
        return self.source[self.current]
    
    def peek_next(self) -> str:
        if self.current + 1 >= len(self.source):
            return "\0"
        return self.source[self.current + 1] 
    
    def add_token(self, token_type: TokenType, literal: Any=None):
        text = self.source[self.start:self.current]
        self.tokens.append(Token(token_type, text, literal, self.line))

    def match(self, expected: str) -> bool:
        if self.is_at_end(): return False
        if self.peek() != expected:
            return False
        self.current += 1
        return True

    def string(self):
        while self.peek() != '"' and not self.is_at_end():
            self.advance()
        if self.is_at_end():
            raise ValueError(f"Unterminated string in line: {self.line}")
        self.advance()

        value = self.source[(self.start + 1):(self.current - 1)]
        self.add_token(TokenType.STRING, value)

    def number(self):
        while self.peek().isdigit():
            self.advance()
        
        if (self.peek() == "." and self.peek_next().isdigit):
            self.advance()
            while self.peek().isdigit():
                self.advance()
        self.add_token(TokenType.NUMBER, float(self.source[self.start:self.current]))

    def identifier(self):
        while self.peek().isalnum():
            self.advance()
        key = self.source[self.start:self.current]
        token_type = keywords.get(key)

        if token_type is None:
            token_type = TokenType.IDENTIFIER
        self.add_token(token_type)

    def comment(self):
        while self.peek() != "\n" and not self.is_at_end():
            self.advance()


class ExprVisitor:
    def visit_literal_expr(self, expr: Expr): raise NotImplementedError
    def visit_unary_expr(self, expr: Expr): raise NotImplementedError
    def visit_binary_expr(self, expr: Expr): raise NotImplementedError
    def visit_variable_expr(self, expr: Expr): raise NotImplementedError


class Expr:
    def accept(visitor: ExprVisitor): raise NotImplementedError


class Binary(Expr):
    def __init__(self, left: Expr, operator: Token, right: Expr):
        self.left = left
        self.operator = operator
        self.right = right

    def accept(self, visitor: ExprVisitor):
        return visitor.visit_binary_expr(self)


class Unary(Expr):
    def __init__(self, operator: Token, right: Expr):
        self.operator = operator
        self.right = right

    def accept(self, visitor: ExprVisitor):
        return visitor.visit_unary_expr(self)


class Literal(Expr):
    def __init__(self, value: Any):
        self.value = value

    def accept(self, visitor: ExprVisitor):
        return visitor.visit_literal_expr(self)
    

class Variable(Expr):
    def __init__(self, name: Token):
        self.name = name

    def accept(self, visitor: ExprVisitor):
        return visitor.visit_variable_expr(self)
    

class StmtVisitor:
    def visit_expression_stmt(self, stmt: Stmt): raise NotImplementedError
    def visit_print_stmt(self, stmt: Stmt): raise NotImplementedError
    def visit_set_stmt(self, stmt: Stmt): raise NotImplementedError


class Stmt:
    def accept(visitor: StmtVisitor): raise NotImplementedError


class Expression(Stmt):
    def __init__(self, expression: Expr):
        self.expression = expression

    def accept(self, visitor: StmtVisitor):
        return visitor.visit_expression_stmt(self)
    

class Print(Stmt):
    def __init__(self, expressions: List[Expr]):
        self.expressions = expressions

    def accept(self, visitor: StmtVisitor):
        return visitor.visit_print_stmt(self)
    

class Set(Stmt):
    def __init__(self, name: Token, initializer: Expr):
        self.name = name
        self.initializer = initializer

    def accept(self, visitor: StmtVisitor):
        return visitor.visit_set_stmt(self)


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0

    def parse(self) -> List[Stmt]:
        statements = []
        while not self.is_at_end():
            statements.append(self.declaration())
        return statements
    
    def declaration(self) -> Stmt:
        self.consume(TokenType.LEFT_PAREN, "Expect '(' before expression.")
        if self.match(TokenType.SET): return self.set_statement()
        return self.statement()
    
    def statement(self) -> Stmt:
        if self.match(TokenType.WRITE):
            return self.print_statement()
        return self.expression_statement()
    
    def set_statement(self) -> Stmt:
        name = self.consume(TokenType.IDENTIFIER, "Expect variable name.")
        initializer = self.expression()
        self.consume(TokenType.RIGHT_PAREN, "Expect ')' after expression.")
        return Set(name, initializer)
    
    def print_statement(self) -> Stmt:
        expressions = []
        while not self.check(TokenType.RIGHT_PAREN):
            expressions.append(self.expression())
        self.consume(TokenType.RIGHT_PAREN, "Expect ')' after expression.")
        return Print(expressions)
   
    def expression_statement(self) -> Stmt:
        expr = self.expression()
        self.consume(TokenType.RIGHT_PAREN, "Expect ')' after expression.")
        return Expression(expr)

    def expression(self) -> Expr:
        return self.binary()
    
    def binary(self) -> Expr:
        if self.match(TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH, 
                      TokenType.BANG_EQUAL, TokenType.EQUAL_EQUAL, TokenType.AND, TokenType.OR,
                      TokenType.LESS, TokenType.LESS_EQUAL, TokenType.GREATER, TokenType.GREATER_EQUAL):
            operator = self.previous()
            left = self.unary()
            while not self.check(TokenType.RIGHT_PAREN) and not self.is_at_end():
                right = self.unary()
                left = Binary(left, operator, right)
            return left
        return self.unary()

    #TODO: allow single unary expression like (-3) or (-(expr))
    def unary(self) -> Expr:
        if self.match(TokenType.MINUS, TokenType.BANG):
            operator = self.previous()
            right = self.unary()
            return Unary(operator, right)
        return self.primary()
   
    def primary(self) -> Expr:
        if self.match(TokenType.TRUE):
            return Literal(True)
        elif self.match(TokenType.FALSE):
            return Literal(False)
        elif self.match(TokenType.NUMBER, TokenType.STRING):
            return Literal(self.previous().literal)
        elif self.match(TokenType.IDENTIFIER):
            return Variable(self.previous())
        elif self.match(TokenType.LEFT_PAREN):
            expr = self.expression()
            self.consume(TokenType.RIGHT_PAREN, "Expect ')' after expression.")
            return expr
        raise RuntimeError(f"Unknown Expression: {self.peek().lexeme}")

    def match(self, *types: TokenType) -> bool:
        for type in types:
            if self.check(type):
                self.advance()
                return True
        return False

    def check(self, type: TokenType) -> bool:
        if self.is_at_end():
            return False
        return self.peek().token_type == type

    def advance(self) -> Token:
        if not self.is_at_end():
            self.current += 1
        return self.previous()

    def is_at_end(self) -> bool:
        return self.peek().token_type == TokenType.EOF

    def peek(self) -> Token:
        return self.tokens[self.current]

    def previous(self) -> Token:
        return self.tokens[self.current - 1]

    def consume(self, type: TokenType, message: str) -> Token:
        if self.check(type):
            return self.advance()
        raise RuntimeError(message)


class Environment:
    def __init__(self):
        self.values = {}

    def set(self, name: str, value: Any):
        self.values[name] = value

    def get(self, name: Token):
        try:
            return self.values[name.lexeme]
        except KeyError:
            raise RuntimeError(name, f"Undefined variable '{name.lexeme}'.")


class Interpreter(ExprVisitor, StmtVisitor):
    def __init__(self, eval=False):
        self.environment = Environment()
        self.eval = eval

    def interpret(self, statements: List[Stmt]):
        results = []
        for stmt in statements:
            if self.eval and isinstance(stmt, Expression):
                results.append(self.execute(stmt))
            else:
                self.execute(stmt)
        return results if self.eval else None

    def execute(self, stmt: Stmt):
        statement = stmt.accept(self)
        return statement if self.eval is True else None

    def evaluate(self, expr: Expr):
        return expr.accept(self)
    
    @staticmethod
    def is_truthy(object: Any):
        return bool(object)
    
    @staticmethod
    def is_equal(left: Any, right: Any):
        return left == right
    
    @staticmethod
    def stringify(obj: Any) -> str:
        if isinstance(obj, str):
            return obj
        elif obj is None:
            return 'None'
        elif isinstance(obj, bool):
            return str(obj).lower()
        elif isinstance(obj, float):
            return str(int(obj)) if obj.is_integer() else str(obj)
        return str(obj)
    
    def check_number_operands(self, left: Any, operator: Token, right: Any):
        if isinstance(left, float) and isinstance(right, float): return
        raise RuntimeError(operator,"Operands must be a type of number")
    
    def check_number_operand(self, operator: Token, right: Any):
        if isinstance(right, float): return
        raise RuntimeError(operator,"Operands must be type of number")
    
    def visit_expression_stmt(self, stmt: Expression):
        return self.evaluate(stmt.expression) if self.eval else None
    
    def visit_print_stmt(self, stmt: Print):
        values = [self.stringify(self.evaluate(expr)) for expr in stmt.expressions]
        print("".join(values))
        return None
    
    def visit_set_stmt(self, stmt: Set):
        if stmt.initializer is not None:
            value = self.evaluate(stmt.initializer)
        self.environment.set(stmt.name.lexeme, value)
        return None

    def visit_variable_expr(self, expr: Variable):
        return self.environment.get(expr.name)
    
    def visit_literal_expr(self, expr: Literal):
        return expr.value
    
    def visit_unary_expr(self, expr: Unary):
        right = self.evaluate(expr.right)
        operator_type = expr.operator.token_type

        match operator_type:
            case TokenType.MINUS:
                self.check_number_operand(expr.operator, right)
                return -right
            case TokenType.BANG:
                return not self.is_truthy(right)
        raise RuntimeError(f"Unknown operator {expr.operator.lexeme} in line {expr.operator.line}")

    def visit_binary_expr(self, expr: Binary):
        left = self.evaluate(expr.left)
        right = self.evaluate(expr.right)
        operator_type = expr.operator.token_type

        match operator_type:
            case TokenType.PLUS:
                if expr.operator.token_type == TokenType.PLUS:
                    if isinstance(left, float) and isinstance(right, float):
                        return left + right
                    elif isinstance(left, str) and isinstance(right, str):
                        return left + right
                    else:
                        raise TypeError(f"Mismatch operation type between {type(left)} and {type(right)}")
            case TokenType.MINUS:
                self.check_number_operands(left, expr.operator, right)
                return left - right
            # TODO: multiply string with a number
            case TokenType.STAR:
                self.check_number_operands(left, expr.operator, right)
                return left * right
            case TokenType.SLASH:
                self.check_number_operands(left, expr.operator, right)
                return left / right
            case TokenType.EQUAL_EQUAL:
                return self.is_equal(left, right)
            case TokenType.BANG_EQUAL:
                return not self.is_equal(left, right)
            case TokenType.OR:
                return (self.is_truthy(left) or self.is_truthy(right))
            case TokenType.AND:
                return (self.is_truthy(left) and self.is_truthy(right))
            # TODO: allow grouped comparison such as (< a b c ...)
            case TokenType.LESS:
                self.check_number_operands(left, expr.operator, right)
                return left < right
            case TokenType.LESS_EQUAL:
                self.check_number_operands(left, expr.operator, right)
                return left <= right
            case TokenType.GREATER:
                self.check_number_operands(left, expr.operator, right)
                return left > right
            case TokenType.GREATER_EQUAL:
                self.check_number_operands(left, expr.operator, right)
                return left >= right
        raise RuntimeError(f"Unknown operator {expr.operator.lexeme} in line {expr.operator.line}")


class AstPrinter:
    def print(self, statements: List[Stmt]):
        print("Abstract Syntax Tree:")
        for stmt in statements:
            self._print_stmt(stmt)
        print()

    def _print_stmt(self, stmt: Stmt, depth=0):
        indent = "   " * depth
        stmt_type = type(stmt).__name__
        print(indent + f"{stmt_type}(")
    
        if isinstance(stmt, Expression):
            self._print_expr(stmt.expression, depth + 1)
        elif isinstance(stmt, Print):
            self._print_expr(stmt.expression, depth + 1)
        elif isinstance(stmt, Set):
            print(indent + "  ", f"Name={stmt.name}")
            self._print_expr(stmt.initializer, depth + 1, "set")
        print(indent + ")")

    def _print_expr(self, expr: Expr, depth=0, instance=None):
        # TODO: idk, improve the indentation? I mean it's good already. (If not lazy)
        indent = "  " * depth

        if isinstance(expr, Literal):
            if instance == "set":
                print(indent, f"Initializer=", end="")
                print(f"Literal(value={expr.value})", end="")
            else:
                print(indent + f"Literal(value={expr.value})", end="") if depth == 1 else print(f"Literal(value={expr.value})", end="")
        elif isinstance(expr, Unary):
            print(indent + f"Unary(") if depth == 1 else print(f"Unary(")
            print(indent + "  ", f"Op={expr.operator.token_type.value}({expr.operator.lexeme})")
            print(indent + "  ", f"Right=", end="")
            self._print_expr(expr.right, depth + 2)
            print(")", end="")
        elif isinstance(expr, Variable):
            print(indent + f"Variable(") if depth == 1 else print(f"Unary(")
            print(indent + "  ", f"Name={expr.name}")
            print(")", end="")
        elif isinstance(expr, Binary):
            print(indent + f"Binary(") if depth == 1 else print(f"Binary(")
            print(indent + "  ", f"Op={expr.operator.token_type.value}({expr.operator.lexeme}),")
            print(indent + "  ", "Left=", end="")
            self._print_expr(expr.left, depth + 2)
            print(",")
            print(indent + "  ", "Right=", end="")
            self._print_expr(expr.right, depth + 2)
            print(")", end="")


def run(source: str, debug_source=False, tokenize=False, eval=False, ast=False, interactive=False):
    if debug_source:
        print("Source:\n", 'START"""\n', source, '\n"""EOF\n', sep="")

    tokens = Lexer(source).tokenize()
    if tokenize:
        print("Tokens:")
        for token in tokens:
            print(token)
        print()

    statements = Parser(tokens).parse()
    if ast:
        printer = AstPrinter()
        printer.print(statements)
    
    interpreter = Interpreter(eval)
    if eval:
        results = interpreter.interpret(statements)
        if results:
            output = "Results:" if not interactive else ""
            print(output, results)
    else:
        interpreter.interpret(statements)


def run_file(fpath: str, debug_source: bool, tokenize: bool, eval: bool, ast: bool):
    with open(fpath, "r", encoding="utf-8") as file:
        source = file.read()
    run(source, debug_source, tokenize, eval, ast)


def run_interactive():
    while True:
        line = input(">>> ")
        if line.lower() == "quit":
            break
        run(line, eval=True, interactive=True)


def main():
    if len(sys.argv) >= 2:
        # TODO: clean these up (make it nicer) before it's too late.
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument("--source", action=argparse.BooleanOptionalAction)
        arg_parser.add_argument("--tokenize", action=argparse.BooleanOptionalAction)
        arg_parser.add_argument("--eval", action=argparse.BooleanOptionalAction)
        arg_parser.add_argument("--ast", action=argparse.BooleanOptionalAction)
        arg_parser.add_argument("file_path")
        args = arg_parser.parse_args()

        if not args.file_path.endswith(".yip"):
            raise NameError(f"Invalid file extension: {args.file_path}")
        run_file(args.file_path, args.source, args.tokenize, args.eval, args.ast)
    else:
        run_interactive()

if __name__ == "__main__":
    main()