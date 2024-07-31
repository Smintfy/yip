from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any
from enum import Enum
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
    PUT = "PUT"
    PROC = "PROC"
    IF = "IF"
    ELSE = "ELSE"
    TRUE = "TRUE"
    FALSE = "FALSE"
    NONE = "NONE"

    # End-of-line
    EOF = "EOF"


keywords = {
    "set": TokenType.SET,
    "put": TokenType.PUT,
    "proc": TokenType.PROC,
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE
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


class ExprVisitor(ABC):
    @abstractmethod
    def visit_literal_expr(self, expr: Expr) -> Any: raise NotImplementedError

    @abstractmethod
    def visit_unary_expr(self, expr: Expr) -> Any: raise NotImplementedError

    @abstractmethod
    def visit_binary_expr(self, expr: Expr) -> Any: raise NotImplementedError

    @abstractmethod
    def visit_group_expr(self, expr: Expr) -> Any: raise NotImplementedError


class Expr(ABC):
    @abstractmethod
    def accept(visitor: ExprVisitor) -> Any: raise NotImplementedError


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
    

class Group(Expr):
    def __init__(self, expressions: List[Expr]):
        self.expressions = expressions

    def accept(self, visitor: ExprVisitor):
        return visitor.visit_group_expr(self)
    
    def __iter__(self):
        return iter(self.expressions)


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0

    def parse(self) -> List[Expr]:
        expressions = []
        while not self.is_at_end():
            expressions.append(self.expression())
        return Group(expressions)

    def expression(self) -> Expr:
        return self.binary()

    def binary(self) -> Expr:
        if self.match(TokenType.LEFT_PAREN):
            operator = self.advance()
            if operator.token_type in [TokenType.MINUS, TokenType.BANG] and ((self.tokens[self.current + 1].token_type == TokenType.RIGHT_PAREN) or (self.peek().token_type == TokenType.LEFT_PAREN)):
                right = self.expression()
                self.consume(TokenType.RIGHT_PAREN, "Expect ')' after expression.")
                return Unary(operator, right)
            left = self.expression()
            while not self.check(TokenType.RIGHT_PAREN) and not self.is_at_end():
                right = self.expression()
                left = Binary(left, operator, right)
            self.consume(TokenType.RIGHT_PAREN, "Expect ')' after expression.")
            return left
        return self.unary()

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
        elif self.match(TokenType.LEFT_PAREN):
            expr = self.expression()
            self.consume(TokenType.RIGHT_PAREN, "Expect ')' after expression.")
            return expr
        raise RuntimeError(f"Expect Expression: {self.peek().lexeme}")

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



class Interpreter(ExprVisitor):
    def interpret(self, expr: Expr) -> Any:
        return expr.accept(self)
    
    @staticmethod
    def is_truthy(object: Any):
        return bool(object)
    
    @staticmethod
    def is_equal(left: Any, right: Any):
        return left == right
    
    def check_number_operands(self, left: Any, operator: Token, right: Any):
        if isinstance(left, float) and isinstance(right, float):
            return
        raise RuntimeError(operator,"Operands must be a type of number")
    
    def check_number_operand(self, operator: Token, right: Any):
        if isinstance(right, float):
            return
        raise RuntimeError(operator,"Operands must be type of number")
    
    def visit_literal_expr(self, expr: Literal) -> Any:
        return expr.value
    
    def visit_unary_expr(self, expr: Unary) -> Any:
        right = expr.right.accept(self)
        operator_type = expr.operator.token_type

        match operator_type:
            case TokenType.MINUS:
                self.check_number_operand(expr.operator, right)
                return -right
            case TokenType.BANG:
                return not self.is_truthy(right)
        raise RuntimeError(f"Unknown operator {expr.operator.lexeme} in line {expr.operator.line}")

    def visit_binary_expr(self, expr: Binary) -> Any:
        left = expr.left.accept(self)
        right = expr.right.accept(self)
        operator_type = expr.operator.token_type

        match operator_type:
            case TokenType.PLUS:
                if expr.operator.token_type == TokenType.PLUS:
                    if isinstance(left, float) and isinstance(right, float):
                        return left + right
                    if isinstance(left, str) and isinstance(right, str):
                        return left + right
            case TokenType.MINUS:
                self.check_number_operands(left, expr.operator, right)
                return left - right
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

    def visit_group_expr(self, expr: Group) -> Any:
        return [e.accept(self) for e in expr.expressions]


def print_ast(expression: Expr, level=0):
    if isinstance(expression, Binary):
        print("  "*level, expression.operator)
        print_ast(expression.left, level + 1)
        print_ast(expression.right, level + 1)
    elif isinstance(expression, Unary):
        print("  "*level, expression.operator)
        print_ast(expression.right, level + 1)
    elif isinstance(expression, Literal):
        print("  "*level, str(expression.value))


def ast_depth(expression: Expr):
    if isinstance(expression, Binary):
        return max(ast_depth(expression.left), ast_depth(expression.right)) + 1
    elif isinstance(expression, Unary):
        return ast_depth(expression.right) + 1
    elif isinstance(expression, Literal):
        return 1


def main():
    if len(sys.argv) < 2:
        raise ValueError("File path is not provided.")

    fpath = sys.argv[1]
    extension = fpath.split(".")[-1]
    
    if extension != "yip":
        raise NameError(f"Invalid file: {fpath}")

    source = open(fpath, "r", encoding="utf-8").read()
    print(source)
    print()

    tokens = Lexer(source)
    tokens = tokens.tokenize()
    for token in tokens:
        print(token)
    print()

    parser = Parser(tokens)
    expression = parser.parse()
    print(expression)

    print()

    for expr in expression:
        print(f"Max Depth: {ast_depth(expr)}")
        print_ast(expr)

    print()

    interpreter = Interpreter()
    result = interpreter.interpret(expression)
    print("Evaluation result: ", result)

if __name__ == "__main__":
    main()