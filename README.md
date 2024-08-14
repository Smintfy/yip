# yip
yip is a toy interpreted programming language.

### Example
1. Basic operations
   
    ```lisp
    (+ 2 3)

    ; output 5
    ```
2. Equality
   
    ```lisp
    (== false true)

    ; output false
    ```

3. Comparison
   
    ```lisp
    (> 7 5)

    ; output true
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
