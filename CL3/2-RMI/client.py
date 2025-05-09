import Pyro4
# get the object registered on the name server
# we registered the class with this uri
string_concatenator = Pyro4.Proxy("PYRONAME:string.concatenator")
# getting the strings input
str1 = input("Enter the first string: ")
str2 = input("Enter the second string: ")
# actually calling the remote function here
result = string_concatenator.concateStrings(str1, str2)
# print result
print("Concatenated string is: ",result)