#Fibonacci series is a series of numbers in which each number is the sum of the two preceding numbers.

def Fibonacci(n):  
   if n <= 1:  
       return n  
   else:  
       return(Fibonacci(n-1) + Fibonacci(n-2))  
 
 
nterms = int(input("Enter the terms? "))  # take input from the user
  
if nterms <= 0:  # check if the number is valid 
   print("Please enter a positive integer")  
else:  
   print("Fibonacci sequence:")  
   for i in range(nterms):  
       print(Fibonacci(i))
