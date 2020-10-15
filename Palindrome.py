# Program to check if a string is palindrome or not

def isPalindrome():
    string = str(input("Enter text : "))
    # make it suitable for caseless comparison
    string = string.casefold()
    # check if the string is equal to its reverse
    if string == string[::-1]:
        result = True
    else:
        result = False
    return result
 
print(isPalindrome())
