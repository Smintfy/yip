# yip
yip is a toy interpreted programming language.

### Example
1. Square root approximation with binary search
```
(set x 2) ; find square root of two

(set tolerance 0.000000000000001) ; 15 digit accuracy
(set low 0)
(set high x)
(set guess (/ (+ low high) 2))

(while (> (abs (- (* guess guess) x)) tolerance)
    (do
        (if (< (* guess guess) x)
            (swap low guess)
            (swap high guess)) ; else
        (swap guess (/ (+ low high) 2))))

(write guess)

; print out 1.414213562373095
``` 

### Roadmap

1. Lexical Analysis
    - [x] Token Type
        - try using enum or mapping
    - [x] Tokenization
        - [x] Operator
        - [x] Number
        - [x] Keyword

2. Parsing
    - [ ] Generate AST
        - not fully done
    - [ ] Pretty print the grammar and syntax

3. Evaluate
    - [ ] Math experssion
        - [x] Real numbers
        - [x] Basic operation
        - [ ] Constant
            - like pi, e, etc
            - only pi and e
        - [ ] Math function
            - like square root and etc. Square root is implemented
        - [ ] Complex math
            - [ ] imaginary number
    - [ ] Logical expression
        - [x] Where input is already set by the user
            - e.g A = True B = False
        - [ ] Generate a truth table
            - generate all possible input. 2^N
    - [ ] Variable
        - variable is declared using `set` keyword. No constant.
    - [ ] Functions

4. Error
    - [ ] idk what to put here yet

5. Testing
