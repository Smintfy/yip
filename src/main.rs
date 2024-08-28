use core::fmt;

#[derive(Debug)]
enum TokenType {
    LeftParen,
    RightParen,
    Plus,
    Minus,
    Number
}

struct Token {
    token_type: TokenType,
    lexeme: String,
    literal: Option<String>,
}

impl Token {
    fn new(token_type: TokenType, lexeme: String, literal: Option<String>) -> Self {
        Token {
            token_type,
            lexeme,
            literal
        }
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

fn parse(source: &str) {
    for c in source.chars() {
        let token = match c {
            '(' => TokenType::LeftParen,
            _ => continue,
        };

        println!("{:?}", token)
    }
}

fn main() {
    let source = "(+ 3 4)";
    
    parse(source)
}