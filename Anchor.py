class Anchor:
    def __init__(self,name,x,y,z):
       self.name = name
       self.x = x
       self.y = y
       self.z = z
       self._rssi = -9999
       self._angle_x_mod = 0
       self._angle_x_mus = 0
       self._angle_y = 0

    def getName(self):
        return self.name

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getZ(self):
        return self.z

    def set_rssi(self, x):
        self._rssi = x

    def get_rssi(self):
        return self._rssi

    def set_angle_x_mod(self, x):
        self._angle_x_mod = x

    def get_angle_x_mod(self):
        return self._angle_x_mod

    def set_angle_x_mus(self, x):
        self._angle_x_mus = x

    def get_angle_x_mus(self):
        return self._angle_x_mus

    def set_angle_y(self, x):
        self._angle_y = x

    def get_angle_y(self):
        return self._angle_y