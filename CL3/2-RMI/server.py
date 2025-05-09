# install - Pyro4
'''
In RMI the actual object is transferred from the server to the client
Unlike RPC where parameters are transferred from client to server
'''

'''
RUNNING THE CODE
1. python -m Pyro4.naming - running the pyro4 sensor
2. then run the server.py
3. then run the client.py
'''

import Pyro4

# we need to define the class and then use this decorator to
# makes the class methods available for remote calls
@Pyro4.expose
class StringConcatenation:
  # actual concatenation function
  def concateStrings(self, str1, str2):
    return str1 + str2

# like intermediatory
# it acts like mini-server
# listen for remote procedure calls for client
daemon = Pyro4.Daemon()
# create an object of the class
server = StringConcatenation()
# register the object with daemon
# universal resource identifier for the objects
# PYRO:objectid@location 
uri = daemon.register(server)
# name server
# here we take the uri and register it with a more easily understandable name to the pyro server
# as it's not easy to remember the uri for each of the object
ns = Pyro4.locateNS()
# printing the uri
print(f"Server URI: {uri}")
# registering the uri with the more easily identifiable name
ns.register("string.concatenator", uri)
print("Server is running . . . . ")
# start the server
daemon.requestLoop()