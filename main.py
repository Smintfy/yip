from typing import List
from functools import reduce
import sys
import math

class Token:
    def __init__(self, token_type, literal: str) -> None:
        self.token_type = token_type
        self.literal = literal

    def __str__(self) -> str:
        return f"{self.token_type} {self.literal}"
    
class Scanner:
    def __init__(self, source: str) -> None:
        self.source: str = source
        self.tokens = []
        self.current = 0

    def tokenize(self) -> List[Token]:
        while not self.is_at_end():
            char = self.advance()
            match char:
                case "(":
                    self.tokens.append(Token("LEFT_PAREN", char))
                case ")":
                    self.tokens.append(Token("RIGHT_PAREN", char))
                case "+" | "-" | "*" | "/":
                    if char == "-" and self.source[self.current].isdigit():
                        self.number(char)
                    else:
                        self.tokens.append(Token("OPERATOR", char))
                case "P":
                    if self.source[self.current] == "I":
                        self.advance()
                        self.tokens.append(Token("CONSTANT", math.pi))
                case "E":
                    # f(x): e^x
                    if self.match("XP"):
                        self.tokens.append(Token("EXPONENT", "exp"))
                    else:
                        self.tokens.append(Token("CONSTANT", math.e))
                case "S":
                    # square root
                    if self.match("QRT"):
                        self.tokens.append(Token("SQUARE_ROOT", "sqrt"))
                case _ if char.isdigit():
                    self.number(char)
        return self.tokens

    def advance(self) -> str:
        self.current += 1
        return self.source[self.current - 1]

    def is_at_end(self) -> bool:
        return self.current >= len(self.source)
    
    def number(self, char: str) -> None:
        num = char
        while self.source[self.current].isdigit() or self.source[self.current] == '.':
            # append the sequence of negative number and float
            num += self.advance()
        self.tokens.append(Token("OPERAND", float(num)))

    def match(self, expected: str) -> bool:
        if self.source[self.current:self.current + len(expected)] == expected:
            self.current += len(expected)
            return True
        return False

    
class Parser:
    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.current = 0

    def expression(self):
        # (ope opr opr)
        # (+ 2 3)
        if self.check("LEFT_PAREN"):
            self.advance()
            operator = self.tokens[self.current]
            operands = []
            self.advance()
            while not self.check("RIGHT_PAREN"):
                # recursive to represent a nested block
                operands.append(self.expression())
            self.expect("RIGHT_PAREN")
            return (operator.literal, operands)
        elif self.check("OPERAND"):
            return float(self.advance().literal)
        elif self.check("CONSTANT"):
            return self.advance().literal

    def advance(self) -> Token:
        if not self.is_at_end():
            self.current += 1
        return self.tokens[self.current - 1]

    def check(self, token_type) -> bool:
        return self.tokens[self.current].token_type == token_type
    
    def expect(self, token_type) -> None:
        token = self.tokens[self.current]
        if token.token_type == token_type:
            self.advance()
    
    def is_at_end(self) -> bool:
        return self.current >= len(self.tokens)
    
def evaluate(ast):
    if isinstance(ast, tuple):
        operator_map = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b
        }
        operator, operands = ast
        if operator in operator_map:
            func = operator_map[operator]
            return reduce(func, (evaluate(op) for op in operands))
    else:
        return ast

def main() -> None:
    fpath = sys.argv[1]
    f = open(fpath)

    tokens = Scanner(f.read())
    tokens = tokens.tokenize()

    temp = Parser(tokens)
    temp = temp.expression()
    print("Abstract Syntax Tree: ", temp)

    result = evaluate(temp)
    print("Output:", result)

if __name__ == "__main__":
    main()