## yip
____
yip is a toy interpreted programming language.

### Example
1. Simple basic operations
    ```lisp
    (+ 2 3)

    ;output 5
    ```

2. Complex basic operations
    ```lisp
    (+ (- 5 3)
    (+ 4.2 -2)
    (- (+ 7 (- 8.5 3.1)) (+ 3 2.8))
    (- -4 (- 1 0.5))
    (+ 2.5 (- 1 1.5))
    (+ (- (+ 6 -3) (- 9 1.2)) 2)
    (+ (* 2
        (- 3
            (/ 8 4.0)))
    (/ (+ 5.5
            (* 2
                (- 10 3)))
        7)
    (- 15
        (* 2.5
            (/ 20
                (+ 3
                (- 10 5)))))
    (* -1.2
        (+ 4
            (/ 6
                (- 9 3))))))
    ```
    ```shell
    python main.py test/test.yip

    # output 13.035714285714285
    ```

### Roadmap

1. Lexical Analysis
    - [ ] Token Type
        - try using enum or mapping
    - [ ] Tokenization
        - [x] Operator
        - [x] Number
        - [ ] Keyword

2. Parsing
    - [ ] Generate AST
        - not fully done
    - [ ] Pretty print the grammar and syntax

3. Evaluate
    - [ ] Math experssion
        - [x] Real numbers
        - [x] Basic operation
        - [x] Constant
            - like pi, e, etc
            - only pi and e
        - [ ] Math function
            - like square root and etc. Square root is implemented
        - [ ] Complex math
            - [ ] imaginary number
    - [ ] Logical expression
        - [ ] Where input is already set by the user
            - e.g A = True B = False
        - [ ] Generate a truth table
            - generate all possible input. 2^N
    - [ ] Variable
        - variable is declared using `set` keyword. No constant.
    - [ ] Functions

4. Error
    - [ ] idk what to put here yet

5. Testing