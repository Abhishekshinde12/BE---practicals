import Pyro4

@Pyro4.expose
class StringConcatenation:
  def concateStrings(self, str1, str2):
    return str1 + str2

daemon = Pyro4.Daemon()
object = StringConcatenation()
uri = daemon.register(object)
ns = Pyro4.locateNS()
print(f"Server URI: {uri}")
ns.register("object", uri)
print("Server is running . . . . ")
# start the server
daemon.requestLoop()