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
