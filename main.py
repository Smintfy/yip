from typing import List, Tuple
from enum import Enum
import sys

class TokenType(Enum):
    # Single characters
    LEFT_PAREN  = "LEFT_PAREN"
    RIGHT_PAREN = "RIGHT_PAREN"
    OPERATOR = "OPERATOR"
    SEMICOLON = "SEMICOLON"

    # Literals
    # IDENTIFIER = "IDENTIFIER"
    # STRING = "STRING"
    NUMBER = "NUMBER"
    CONSTANT = "CONSTANT"
    MATH_FUNC = "MATH_FUNC"

    # Logical statement
    # NOT = "NOT"
    # AND = "AND"
    # OR = "OR"
    # NAND = "NAND"
    # NOR = "NOR"

    # End-of-line
    EOF = "EOF"


class Constant(Enum):
    PI = 3.141592653589793
    EULER_NUMBER = 2.718281828459045
    SQRT_2 = 1.414213562373095

class Token:
    def __init__(self, token_type: TokenType, literal):
        self.token_type = token_type
        self.literal = literal

    def __str__(self) -> str:
        return f"{self.token_type.value} {self.literal}"
    
class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.tokens = []
        self.current = 0

    def tokenize(self) -> List[Token]:
        while not self.is_at_end():
            self.scan_token()
        self.add_token(TokenType.EOF, "")
        return self.tokens

    def scan_token(self):
        char = self.advance()
        match char:
            case "(":
                self.add_token(TokenType.LEFT_PAREN, char)
            case ")":
                self.add_token(TokenType.RIGHT_PAREN, char)
            case "+" | "-" | "*" | "/":
                if char == "-" and self.source[self.current].isdigit():
                    self.number(char)
                else:
                    self.add_token(TokenType.OPERATOR, char)
            case ";":
                while self.source[self.current] != "\n":
                    self.advance()
            case "p":
                if self.source[self.current] == "i":
                    self.advance()
                    self.add_token(TokenType.CONSTANT, Constant.PI.value)
                if self.match("ow"):
                     self.add_token(TokenType.MATH_FUNC, "pow")
            case "e":
                # f(x): e^x
                if self.match("xp"):
                    self.add_token(TokenType.MATH_FUNC, "exp")
                else:
                    self.add_token(TokenType.CONSTANT, Constant.EULER_NUMBER.value)
            case "s":
                # square root
                if self.match("qrt"):
                    self.add_token(TokenType.MATH_FUNC, "sqrt")
            case "π":
                self.add_token(TokenType.CONSTANT, Constant.PI.value)
            case _:
                if char.isdigit():
                    self.number(char)
                elif char.isspace():
                    pass
                else:
                    raise ValueError(f"Unexpected character: {char}")

    def advance(self) -> str:
        self.current += 1
        return self.source[self.current - 1]

    def is_at_end(self) -> bool:
        return self.current >= len(self.source)
    
    def number(self, char: str):
        num = char
        while self.source[self.current].isdigit() or self.source[self.current] == '.':
            # append the sequence of negative number and float
            num += self.advance()
        self.add_token(TokenType.NUMBER, float(num))

    def match(self, expected: str) -> bool:
        if self.source[self.current:self.current + len(expected)] == expected:
            self.current += len(expected)
            return True
        return False
    
    def add_token(self, token_type: TokenType, literal):
        self.tokens.append(Token(token_type, literal))
    
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0

    def parse(self) -> List[Tuple]:
        expressions = []
        while not self.is_at_end():
            expressions.append(self.expression())
        return expressions

    def expression(self):
        # (OPERATOR OPERAND OPERAND ...)
        if self.match(TokenType.LEFT_PAREN):
            operator = self.consume(TokenType.OPERATOR, TokenType.MATH_FUNC)
            operands = []
            while not self.check(TokenType.RIGHT_PAREN):
                # recursive to represent a nested block
                operands.append(self.expression())
            self.consume(TokenType.RIGHT_PAREN)
            return (operator.literal, operands)
        elif self.match(TokenType.NUMBER):
            return float(self.prev().literal)
        elif self.match(TokenType.CONSTANT):
            return self.prev().literal
    
    def advance(self) -> Token:
        self.current += 1
        return self.prev()
    
    def peek(self) -> Token:
        return self.tokens[self.current]
    
    def prev(self) -> Token:
        return self.tokens[self.current - 1]

    def match(self, token_type) -> bool:
        if self.check(token_type):
            self.advance()
            return True
        return False

    def check(self, token_type: TokenType) -> bool:
        return self.peek().token_type == token_type
    
    def consume(self, *token_types) -> Token:
        for t in token_types:
            if self.check(t):
                return self.advance()
    
    def is_at_end(self) -> bool:
        return self.peek().token_type == TokenType.EOF
    
def evaluate(ast):
    if isinstance(ast, tuple):
        operator_map = {
            "+": lambda x, y: x + y,
            "-": lambda x, y: x - y,
            "*": lambda x, y: x * y,
            "/": lambda x, y: x / y
        }
        operator, operands = ast
        if operator in operator_map:
            func = operator_map[operator]
            return reduce(func, (evaluate(op) for op in operands))
        elif operator == "sqrt":
            if len(operands) > 1:
                raise ValueError("Square root can only compute one operand.")
            if operands[0] == 2:
                return Constant.SQRT_2.value
            return sqrt(evaluate(operands[0]))
        elif operator == "pow":
            if len(operands) < 2:
                return evaluate(operands[0])
            if len(operands) > 4:
                raise ValueError("Amount of chained power cannot exceed 4.")
            return reduce(lambda x, y: x**y, (evaluate(op) for op in operands))
        elif operator == "exp":
            if len(operands) > 1:
                raise ValueError("Exponential function can only compute one operand.")
            return Constant.EULER_NUMBER.value**evaluate(operands[0])
        else:
            raise RuntimeError(f"Unknown operator: {operator}")
    else:
        return ast
    
def reduce(function, sequence):
    it = iter(sequence)
    res = next(it)
    for element in it:
        res = function(res, element)
    return res

def sqrt(x, tolerance=1e-15):
    if x < 0:
        raise ValueError("Cannot compute the square root of a negative number.")
    low, high = 0, x
    guess = (low + high) / 2
    while abs(guess * guess - x) > tolerance:
        if guess * guess < x:
            low = guess
        else:
            high = guess
        guess = (low + high) / 2
    return round(guess, 10)

def main():
    if len(sys.argv) < 2:
        raise ValueError("File path is not provided.")

    fpath = sys.argv[1]
    extension = fpath.split(".")[-1]
    
    if extension != "yip":
        raise NameError(f"Invalid file: {fpath}")

    file = open(fpath, "r", encoding="utf-8").read()
    tokens = Lexer(file.lower())
    tokens = tokens.tokenize()
    
    for token in tokens:
        print(token)

    temp = Parser(tokens) 
    temp = temp.parse()
    print("Abstract Syntax Tree: ", temp)

    result = list(map(evaluate, temp))
    print("Output:", result)

if __name__ == "__main__":
    main()