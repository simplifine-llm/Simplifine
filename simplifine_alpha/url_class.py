from dataclasses import dataclass

@dataclass
class url_config:
    url: str = "http://34.196.116.1:5000"   # The URL of the server, default is the L4 server
    url_a100 = "http://18.189.220.250:5000"  # The URL of the A100 server
    url_l4 = "http://34.196.116.1:5000"     # The URL of the L4 server