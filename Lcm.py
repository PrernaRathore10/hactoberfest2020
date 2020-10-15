# Python Program to find the L.C.M. of two input number

def lcm():
    print("Enter two numbers : ")
    x  = int(input("num1 : "))
    y  = int(input("num2 : "))
    # choose the greater number
    if x > y:
        greater = x
    else:
        greater = y

    while(True):
        if((greater % x == 0) and (greater % y == 0)):
            lcm = greater
            break
        greater += 1

    return lcm

print("The L.C.M. is", lcm())
