import time


class LoadBalancer:
  # storing a list of servers and the current index of the server to be used
  def __init__(self, serversList):
    self.servers = serversList
    self.currentIndex = 0

  # get the server at the current index and increment the index
  # then return the server
  def distribute_request(self):
    server = self.servers[self.currentIndex]
    self.currentIndex = (self.currentIndex + 1) % len(self.servers)
    return server


# Server class that has a name and a method to process a request
class Server:
  # storing the name of the server
  def __init__(self, name):
    self.name = name

  # Simulating the processing of a request by sleeping for 1 second
  def process_request(self):
    print(f"Server {self.name} is processing the request.")
    time.sleep(1)
    print(f"Server {self.name} has finished processing the request.")


# Create Servers
server1 = Server("Server 1")
server2 = Server("Server 2")
server3 = Server("Server 3")

# Create Load Balancer with the list of servers
loadBalancer = LoadBalancer([server1, server2, server3])

# Distribute requests to servers
for i in range(10):
  # get the server to process the request
  server = loadBalancer.distribute_request()
  # call the process_request method on the server
  server.process_request()
