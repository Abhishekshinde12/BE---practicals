# import the client
from xmlrpc.client import ServerProxy
# setup the url for proxy
# client proxy - takes request and sends to server
'''
Without a proxy, you'd have to:
  Manually format XML requests
  Open a network socket
  Send HTTP POST requests
  Parse the XML response
'''
# it makes the remote function call as if it is a local function
proxy = ServerProxy("http://localhost:4000/")
# take the number
num = int(input("Enter a number: "))
# call the remote function and store result and print it
result = proxy.factorial(num)
print("Factorial of", num, "is", result)