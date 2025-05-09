# importing the server
from xmlrpc.server import SimpleXMLRPCServer

# factorial function
def factorial(n):
  if (n == 0):
    return 1
  return n * factorial(n-1)

# create server
server = SimpleXMLRPCServer(("localhost", 4000))
# register the function
server.register_function(factorial, "factorial")
print("Server started..........")
# start the server
server.serve_forever()