from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from enum import Enum
from dataclasses import dataclass
import sys

class TokenType(Enum):
    # Single characters
    LEFT_PAREN  = "LEFT_PAREN"
    RIGHT_PAREN = "RIGHT_PAREN"
    PLUS = "PLUS"
    MINUS = "MINUS"
    STAR = "STAR"
    SLASH = "SLASH"
    SEMICOLON = "SEMICOLON"

    # Literals
    IDENTIFIER = "IDENTIFIER"
    STRING = "STRING"
    NUMBER = "NUMBER"
    CONSTANT = "CONSTANT"

    # Keywords
    SET = "SET"
    PUT = "PUT"

    # End-of-line
    EOF = "EOF"

keywords = {
    "set": TokenType.SET,
    "put": TokenType.PUT
}

class Token:
    def __init__(self, token_type: TokenType, lexeme: str, literal: object, line: int):
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
            case "(": self.add_token(TokenType.LEFT_PAREN)
            case ")": self.add_token(TokenType.RIGHT_PAREN)
            case "+": self.add_token(TokenType.PLUS)
            case "-": self.add_token(TokenType.MINUS)
            case "*": self.add_token(TokenType.STAR)
            case "/": self.add_token(TokenType.SLASH)
            case ";":
                while self.peek() != "\n" and not self.is_at_end():
                    self.advance()
            case '"':
                self.string()
            case "\n":
                self.line += 1
            case _:
                if char.isdigit():
                    self.number()
                elif char.isalpha():
                    self.identifier()
                elif char.isspace():
                    pass
                else:
                    raise ValueError(f"Unexpected character: {char} in line: {self.line}")

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
    
    def add_token(self, token_type: TokenType, literal: object=None):
        text = self.source[self.start:self.current]
        self.tokens.append(Token(token_type, text, literal, self.line))

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

@dataclass
class BinaryExpr:
    left: Expr
    operator: Token
    right: Expr

    def accept(self, visitor: ExprVisitor) -> object:
        return visitor.visit_binary_expr(self)

@dataclass
class UnaryExpr:
    operator: Token
    right: Expr

    def accept(self, visitor: ExprVisitor) -> object:
        return visitor.visit_unary_expr(self)

@dataclass
class LiteralExpr:
    value: object

    def accept(self, visitor: ExprVisitor) -> object:
        return visitor.visit_literal_expr(self)
    
@dataclass
class GroupExpr:
    expressions: List[Expr]

    def accept(self, visitor: ExprVisitor) -> object:
        return visitor.visit_group_expr(self)
    
    def __iter__(self):
        return iter(self.expressions)

Expr = BinaryExpr | UnaryExpr | LiteralExpr | GroupExpr

class ExprVisitor:
    @abstractmethod
    def visit_literal_expr(self, expr: LiteralExpr) -> object:
        pass

    @abstractmethod
    def visit_unary_expr(self, expr: UnaryExpr) -> object:
        pass

    @abstractmethod
    def visit_binary_expr(self, expr: BinaryExpr) -> object:
        pass

    @abstractmethod
    def visit_group_expr(self, expr: GroupExpr) -> object:
        pass

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0

    def parse(self):
        expressions = []
        while not self.is_at_end():
            if self.check(TokenType.LEFT_PAREN):
                expressions.append(self.expression())
            else:
                raise RuntimeError(f"Unexpected token {self.peek().lexeme} in line: {self.peek().line}. Expect expression to be enclosed in parentheses.")
        return GroupExpr(expressions)
    
    def expression(self) -> Expr:
        if self.match(TokenType.NUMBER):
            return LiteralExpr(self.previous().literal)
        if self.match(TokenType.MINUS):
            operator = self.previous()
            right = self.expression()
            return UnaryExpr(operator, right)
        if self.match(TokenType.LEFT_PAREN):
            operator = self.peek()
            self.advance()
            left = self.expression()
            while not self.check(TokenType.RIGHT_PAREN):
                right = self.expression()
                left = BinaryExpr(left, operator, right)
            self.consume(TokenType.RIGHT_PAREN, "Expect ')' after expression.")
            return left 
        raise RuntimeError(f"Expected Expression: {self.peek().lexeme}")

    def is_at_end(self) -> bool:
        return self.peek().token_type == TokenType.EOF

    def peek(self) -> Token:
        return self.tokens[self.current]
    
    def previous(self) -> Token:
        return self.tokens[self.current - 1]
    
    def advance(self) -> Token:
        if not self.is_at_end():
            self.current += 1
        return self.previous()

    def check(self, token_type: TokenType) -> bool:
        if self.is_at_end():
            return False
        return self.peek().token_type == token_type

    def match(self, *token_types: TokenType) -> bool:
        for t in token_types:
            if self.check(t):
                self.advance()
                return True
        return False

    def consume(self, token_type: TokenType, msg: str) -> Token:
        if self.check(token_type):
            return self.advance()
        raise RuntimeError(msg)
    
class Interpreter(ExprVisitor):
    def interpret(self, expr: Expr) -> object:
        return expr.accept(self)
    
    def visit_literal_expr(self, expr: LiteralExpr) -> object:
        return expr.value
    
    def visit_unary_expr(self, expr: UnaryExpr) -> object:
        right = expr.right.accept(self)
        if expr.operator.lexeme == "-":
            return -right
        raise RuntimeError(f"Unknown operator {expr.operator.lexeme} in line {expr.operator.line}")

    def visit_binary_expr(self, expr: BinaryExpr) -> object:
        left = expr.left.accept(self)
        right = expr.right.accept(self)
        if expr.operator.lexeme == "+":
            return left + right
        elif expr.operator.lexeme == "-":
            return left - right
        elif expr.operator.lexeme == "*":
            return left * right
        elif expr.operator.lexeme == "/":
            return left / right
        raise RuntimeError(f"Unknown operator {expr.operator.lexeme} in line {expr.operator.line}")

    def visit_group_expr(self, expr: GroupExpr) -> object:
        return [e.accept(self) for e in expr.expressions]
    
def print_ast(expression: Expr, level=0):
    if isinstance(expression, BinaryExpr):
        print("  "*level, expression.operator)
        print_ast(expression.left, level + 1)
        print_ast(expression.right, level + 1)
    elif isinstance(expression, UnaryExpr):
        print("  "*level, expression.operator)
        print_ast(expression.right, level + 1)
    elif isinstance(expression, LiteralExpr):
        print("  "*level, str(expression))

def ast_depth(expression: Expr):
    if isinstance(expression, BinaryExpr):
        return max(ast_depth(expression.left), ast_depth(expression.right)) + 1
    elif isinstance(expression, UnaryExpr):
        return ast_depth(expression.right) + 1
    elif isinstance(expression, LiteralExpr):
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