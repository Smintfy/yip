# yip
yip is a toy interpreted functional programming language.

### Example
1. Square root approximation with binary search
```clj
; Find the square root of a number using binary search
(fn sqrt [x]
    (do
        (set tolerance 1e-15) ; up to 15 digit accuracy.
        (set low 0)
        (set high x)
        (set guess (/ (+ low high) 2))
        (while (> (abs (- (* guess guess) x)) tolerance)
            (do
                (if (< (* guess guess) x)
                    (swap low guess)
                    (swap high guess)) ; else
                (swap guess (/ (+ low high) 2))))
        (write "The square root of " x " is: " guess)))

(sqrt [2]) ; print out 1.414213562373095
``` 
