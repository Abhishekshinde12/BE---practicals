import Pyro4

string_concatenator = Pyro4.Proxy("PYRONAME:object")
str1 = input("Enter the first string: ")
str2 = input("Enter the second string: ")
result = string_concatenator.concateStrings(str1, str2)
print(result)